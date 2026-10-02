"""SQLite connection management.

Streamlit re-executes the whole script on every widget interaction, so a
module-level connection is unsafe. Every operation therefore acquires a
short-lived connection through a context manager, with WAL journalling enabled
to permit concurrent readers alongside a writer.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

import pandas as pd

from freshsense.config import SETTINGS
from freshsense.exceptions import DatabaseError
from freshsense.logging_config import get_logger
from freshsense.paths import PATHS

LOG = get_logger(__name__)


class Database:
    """Thin, dependency-injectable wrapper around a SQLite database file."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path: Path = db_path or PATHS.database
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._cfg = SETTINGS.section("database")

    # ── connection ────────────────────────────────────────────────────
    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Yield a configured connection, committing on success.

        Yields:
            An open :class:`sqlite3.Connection` with ``Row`` factory.

        Raises:
            DatabaseError: If the connection or the transaction fails.
        """
        conn = sqlite3.connect(
            self.db_path,
            timeout=float(self._cfg.get("busy_timeout_ms", 5000)) / 1000.0,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        try:
            conn.execute(f"PRAGMA journal_mode={self._cfg.get('journal_mode', 'WAL')}")
            if self._cfg.get("foreign_keys", True):
                conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA synchronous=NORMAL")
            yield conn
            conn.commit()
        except sqlite3.Error as exc:                      # pragma: no cover
            conn.rollback()
            LOG.error("Database error: %s", exc)
            raise DatabaseError(str(exc)) from exc
        finally:
            conn.close()

    # ── schema ────────────────────────────────────────────────────────
    def initialise(self, schema_path: Path | None = None) -> None:
        """Create every table, index and view. Idempotent."""
        schema_path = schema_path or PATHS.schema_file
        if not schema_path.is_file():
            raise DatabaseError(f"Schema file not found: {schema_path}")

        with self.connect() as conn:
            conn.executescript(schema_path.read_text(encoding="utf-8"))
        LOG.info("Database initialised at %s", PATHS.relative(self.db_path))

    def drop_all(self) -> None:
        """Remove the database file entirely. Used by ``--reset`` flows."""
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(str(self.db_path) + suffix)
            if candidate.exists():
                candidate.unlink()
        LOG.warning("Database removed: %s", PATHS.relative(self.db_path))

    @property
    def exists(self) -> bool:
        return self.db_path.is_file()

    # ── queries ───────────────────────────────────────────────────────
    def query(self, sql: str, params: Sequence[Any] | dict[str, Any] = ()) -> pd.DataFrame:
        """Run a SELECT and return the result as a DataFrame."""
        with self.connect() as conn:
            return pd.read_sql_query(sql, conn, params=params)

    def query_one(
        self, sql: str, params: Sequence[Any] | dict[str, Any] = ()
    ) -> dict[str, Any] | None:
        """Run a SELECT expected to return at most one row."""
        with self.connect() as conn:
            row = conn.execute(sql, params).fetchone()
        return dict(row) if row else None

    def scalar(self, sql: str, params: Sequence[Any] = (), default: Any = 0) -> Any:
        """Return the first column of the first row, or ``default``."""
        with self.connect() as conn:
            row = conn.execute(sql, params).fetchone()
        return default if row is None or row[0] is None else row[0]

    def execute(self, sql: str, params: Sequence[Any] | dict[str, Any] = ()) -> int:
        """Run a single write statement and return the affected row count."""
        with self.connect() as conn:
            cursor = conn.execute(sql, params)
            return cursor.rowcount

    def execute_many(self, sql: str, rows: Sequence[Sequence[Any]]) -> int:
        """Run a batched write statement."""
        if not rows:
            return 0
        with self.connect() as conn:
            cursor = conn.executemany(sql, rows)
            return cursor.rowcount

    def insert_returning_id(
        self, sql: str, params: Sequence[Any] | dict[str, Any] = ()
    ) -> int:
        """Run an INSERT and return the new ``rowid``."""
        with self.connect() as conn:
            cursor = conn.execute(sql, params)
            return int(cursor.lastrowid or 0)

    # ── bulk load ─────────────────────────────────────────────────────
    def write_frame(
        self,
        frame: pd.DataFrame,
        table: str,
        *,
        if_exists: str = "append",
        chunksize: int = 1000,
    ) -> int:
        """Write a DataFrame to ``table``.

        Only columns that exist in the destination table are written, so an
        enriched analytical frame can be persisted without schema drift.
        """
        if frame.empty:
            LOG.warning("write_frame: '%s' received an empty frame", table)
            return 0

        columns = self.table_columns(table)
        payload = frame[[c for c in frame.columns if c in columns]].copy()

        for column in payload.columns:
            if pd.api.types.is_datetime64_any_dtype(payload[column]):
                payload[column] = payload[column].dt.strftime("%Y-%m-%d")

        with self.connect() as conn:
            payload.to_sql(
                table, conn, if_exists=if_exists, index=False, chunksize=chunksize
            )
        LOG.info("Wrote %d rows x %d cols -> %s", len(payload), payload.shape[1], table)
        return len(payload)

    def table_columns(self, table: str) -> list[str]:
        """Return the column names of ``table``."""
        with self.connect() as conn:
            return [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]

    def table_exists(self, table: str) -> bool:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table','view') "
                "AND name = ?",
                (table,),
            ).fetchone()
        return row is not None

    def row_count(self, table: str) -> int:
        if not self.table_exists(table):
            return 0
        return int(self.scalar(f"SELECT COUNT(*) FROM {table}"))

    def truncate(self, *tables: str) -> None:
        """Delete every row from the named tables, preserving structure."""
        with self.connect() as conn:
            for table in tables:
                conn.execute(f"DELETE FROM {table}")
        LOG.info("Truncated: %s", ", ".join(tables))

    def summary(self) -> pd.DataFrame:
        """Row counts for every user table — used by the Settings page."""
        with self.connect() as conn:
            names = [
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name NOT LIKE 'sqlite_%' ORDER BY name"
                )
            ]
            rows = [
                {"table": n, "rows": conn.execute(f"SELECT COUNT(*) FROM {n}").fetchone()[0]}
                for n in names
            ]
        return pd.DataFrame(rows)


_DEFAULT_DB: Database | None = None


def get_database() -> Database:
    """Return the process-wide :class:`Database` singleton."""
    global _DEFAULT_DB
    if _DEFAULT_DB is None:
        _DEFAULT_DB = Database()
    return _DEFAULT_DB


__all__ = ["Database", "get_database"]