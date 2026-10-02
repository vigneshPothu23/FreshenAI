"""Persistence service: pipeline frames into the normalised database.

The missing layer. The pipeline produces correct analytical frames and the
schema is a correct operational model, but nothing ever carried one into the
other. This module does, driven entirely by the declarative contract in
:mod:`freshsense.db.mapping` — adding a table means adding a ``TableSpec``, not
editing this file.

Four problems it has to solve, none of which a bare ``to_sql`` handles:

**Ordering.** ``orders`` references ``batches`` references ``sellers``. Loading
out of order fails a foreign-key constraint, so the order is derived
topologically from the specs rather than hardcoded.

**Orphans.** Cleaning legitimately removes rows — a batch dropped for an
impossible quantity, a seller dropped as a duplicate. Orders referencing them
are now orphans. SQLite would abort the entire transaction on the first one, so
orphans are detected in Python, dropped, and *reported*. Silently discarding
them would misstate revenue; failing the whole load over them would make the
pipeline hostage to a handful of rows.

**Surrogate keys.** ``items`` has an autoincrement primary key that ``batches``
references. ``INSERT OR REPLACE`` would reissue those ids on every reload and
orphan every child row, so surrogate-key tables use ``INSERT OR IGNORE`` and
their ids are resolved back onto dependants after insert.

**Atomicity.** The whole load runs in one transaction. A partial load — sellers
present, batches missing — is worse than no load, because every downstream
component would read it as real.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from freshsense.db.mapping import (SCHEMA_MAP, SPECS_BY_TABLE, TableSpec,
                                   load_order, project, verify_against_schema)
from freshsense.db.database import Database, get_database
from freshsense.exceptions import DatabaseError, DataValidationError
from freshsense.logging_config import get_logger
from freshsense.paths import PATHS

LOG = get_logger(__name__)

REPLACE, UPSERT = "replace", "upsert"


# ══════════════════════════════════════════════════════════════════════════
@dataclass
class TableLoadResult:
    """What happened to one table."""

    table: str
    rows_source: int = 0
    rows_projected: int = 0
    rows_written: int = 0
    rows_dropped_orphan: int = 0
    rows_dropped_duplicate: int = 0
    orphan_detail: dict[str, int] = field(default_factory=dict)
    latency_ms: int = 0
    status: str = "ok"
    error: str = ""

    @property
    def rows_dropped(self) -> int:
        return self.rows_dropped_orphan + self.rows_dropped_duplicate

    def as_dict(self) -> dict[str, Any]:
        return {
            "table": self.table,
            "source_rows": self.rows_source,
            "projected": self.rows_projected,
            "written": self.rows_written,
            "dropped_orphan": self.rows_dropped_orphan,
            "dropped_duplicate": self.rows_dropped_duplicate,
            "latency_ms": self.latency_ms,
            "status": self.status,
            "detail": self.error or (
                "; ".join(f"{k}={v}" for k, v in self.orphan_detail.items())
                or "clean"
            ),
        }


@dataclass
class LoadReport:
    """Outcome of a whole load."""

    results: list[TableLoadResult] = field(default_factory=list)
    mode: str = REPLACE
    duration_ms: int = 0
    verified: dict[str, list[str]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return all(r.status == "ok" for r in self.results) and not self.verified

    @property
    def total_written(self) -> int:
        return sum(r.rows_written for r in self.results)

    @property
    def total_dropped(self) -> int:
        return sum(r.rows_dropped for r in self.results)

    def frame(self) -> pd.DataFrame:
        if not self.results:
            return pd.DataFrame(
                columns=["table", "source_rows", "projected", "written",
                         "dropped_orphan", "dropped_duplicate", "latency_ms",
                         "status", "detail"]
            )
        return pd.DataFrame([r.as_dict() for r in self.results])

    def summary(self) -> str:
        return (
            f"{self.total_written:,} row(s) written across "
            f"{len(self.results)} table(s) in {self.duration_ms} ms "
            f"({self.mode} mode)"
            + (f"; {self.total_dropped:,} row(s) dropped for referential "
               f"integrity" if self.total_dropped else "")
        )


# ══════════════════════════════════════════════════════════════════════════
class DatabaseLoader:
    """Loads projected pipeline frames into the normalised schema."""

    def __init__(self, db: Database | None = None) -> None:
        self.db = db or get_database()

    # ── preflight ─────────────────────────────────────────────────────
    def verify_contract(self) -> dict[str, list[str]]:
        """Check the mapping against the live schema before touching data."""
        return verify_against_schema(self.db)

    def ensure_schema(self) -> None:
        """Create the schema if it is absent, so a first run needs no setup step."""
        if not self.db.table_exists("batches"):
            LOG.info("Schema not present — initialising")
            self.db.initialise()

    # ── load ──────────────────────────────────────────────────────────
    def load(
        self,
        tables: Mapping[str, pd.DataFrame],
        *,
        mode: str = REPLACE,
        strict: bool = False,
    ) -> LoadReport:
        """Persist every mapped table inside a single transaction.

        Args:
            tables: Pipeline frames keyed by source name (``inventory``,
                ``sales``, ``orders``, ``buyers``, ``sellers``).
            mode: ``"replace"`` clears the mapped tables first, giving a
                deterministic reload. ``"upsert"`` merges on the natural key,
                preserving rows the current extract does not mention.
            strict: Raise when any row is dropped for referential integrity,
                instead of reporting it.

        Returns:
            A :class:`LoadReport` with per-table counts and drop reasons.

        Raises:
            DatabaseError: If the mapping contract does not hold, or the
                transaction fails. Nothing is committed in either case.
            DataValidationError: If ``strict`` and rows were dropped.
        """
        if mode not in (REPLACE, UPSERT):
            raise ValueError(f"mode must be '{REPLACE}' or '{UPSERT}', got {mode!r}")

        self.ensure_schema()

        problems = self.verify_contract()
        if problems:
            raise DatabaseError(
                "Schema mapping contract is broken; refusing to load. "
                + "; ".join(f"{t}: {', '.join(v)}" for t, v in problems.items())
            )

        started = time.perf_counter()
        report = LoadReport(mode=mode)
        ordered = load_order()

        # One connection, one transaction. A partially loaded database is worse
        # than an empty one: every downstream component would read it as real.
        conn = sqlite3.connect(self.db.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("BEGIN")

            if mode == REPLACE:
                self._clear(conn, ordered)

            for spec in ordered:
                report.results.append(self._load_table(conn, spec, tables))

            conn.commit()
        except Exception as exc:
            conn.rollback()
            LOG.error("Load failed and was rolled back: %s", exc)
            raise DatabaseError(f"Database load failed: {exc}") from exc
        finally:
            conn.close()

        report.duration_ms = int((time.perf_counter() - started) * 1000)

        if strict and report.total_dropped:
            raise DataValidationError(
                f"{report.total_dropped} row(s) dropped for referential "
                f"integrity: "
                + "; ".join(
                    f"{r.table}({r.rows_dropped})"
                    for r in report.results if r.rows_dropped
                )
            )

        LOG.info("Load complete — %s", report.summary())
        return report

    # ── internals ─────────────────────────────────────────────────────
    @staticmethod
    def _clear(conn: sqlite3.Connection, ordered: Sequence[TableSpec]) -> None:
        """Empty the mapped tables in reverse dependency order.

        Children first: deleting ``sellers`` before ``batches`` would trip the
        foreign key even though both are about to be emptied.
        """
        for spec in reversed(list(ordered)):
            conn.execute(f"DELETE FROM {spec.table}")
        LOG.info("Cleared %d mapped table(s) for a deterministic reload",
                 len(ordered))

    def _load_table(
        self,
        conn: sqlite3.Connection,
        spec: TableSpec,
        tables: Mapping[str, pd.DataFrame],
    ) -> TableLoadResult:
        """Project, integrity-filter and insert one table."""
        started = time.perf_counter()
        result = TableLoadResult(table=spec.table)

        source = tables.get(spec.source)
        if source is None:
            result.status = "skipped"
            result.error = f"no source frame named '{spec.source}' was supplied"
            LOG.warning("Table '%s' skipped: %s", spec.table, result.error)
            return result

        result.rows_source = len(source)

        # Measure duplicates against the *derived* frame, not the raw source.
        # ``items`` collapses 1,425 batch rows into one row per product by
        # design; counting that as 1,380 duplicates would report a deliberate
        # aggregation as data loss.
        derived = spec.derive(source) if spec.derive is not None else source
        frame = project(source, spec)
        result.rows_projected = len(frame)
        result.rows_dropped_duplicate = max(0, len(derived) - len(frame))

        if frame.empty:
            result.status = "empty"
            result.latency_ms = int((time.perf_counter() - started) * 1000)
            return result

        frame = self._resolve_surrogates(conn, spec, frame)
        frame, dropped = self._drop_orphans(conn, spec, frame)
        result.rows_dropped_orphan = sum(dropped.values())
        result.orphan_detail = dropped

        if frame.empty:
            result.status = "empty"
            result.error = "every row was dropped for referential integrity"
            result.latency_ms = int((time.perf_counter() - started) * 1000)
            return result

        result.rows_written = self._insert(conn, spec, frame)
        result.latency_ms = int((time.perf_counter() - started) * 1000)

        LOG.info("[%-16s] %5d source -> %5d written%s (%d ms)",
                 spec.table, result.rows_source, result.rows_written,
                 f", {result.rows_dropped_orphan} orphan(s) dropped"
                 if result.rows_dropped_orphan else "",
                 result.latency_ms)
        return result

    @staticmethod
    def _resolve_surrogates(
        conn: sqlite3.Connection, spec: TableSpec, frame: pd.DataFrame
    ) -> pd.DataFrame:
        """Attach parent surrogate ids that the pipeline cannot know.

        ``batches.item_id`` points at the autoincrement key of ``items``, which
        only exists once ``items`` has been inserted. Resolving it by product
        name here is what makes the normalised link real rather than nominal —
        a null ``item_id`` would leave the catalogue unjoinable.
        """
        if spec.table != "batches" or "product_name" not in frame.columns:
            return frame

        rows = conn.execute("SELECT item_id, name FROM items").fetchall()
        if not rows:
            return frame

        lookup = {r["name"]: r["item_id"] for r in rows}
        out = frame.copy()
        out["item_id"] = out["product_name"].map(lookup)

        unresolved = int(out["item_id"].isna().sum())
        if unresolved:
            LOG.warning("batches: %d row(s) reference a product missing from the "
                        "items catalogue; item_id left null", unresolved)
        out["item_id"] = out["item_id"].astype("Int64")
        return out

    @staticmethod
    def _drop_orphans(
        conn: sqlite3.Connection, spec: TableSpec, frame: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict[str, int]]:
        """Remove rows whose foreign keys have no parent.

        Detected in Python because SQLite aborts the whole transaction on the
        first violation, which would make a complete load hostage to a handful
        of rows that cleaning legitimately removed upstream.
        """
        dropped: dict[str, int] = {}
        out = frame

        for local, (parent_table, parent_column) in spec.foreign_keys.items():
            if local not in out.columns:
                continue

            rows = conn.execute(
                f"SELECT {parent_column} FROM {parent_table}"
            ).fetchall()
            parents = {r[parent_column] for r in rows}

            present = out[local].notna()
            # A null foreign key is permitted by the schema — an order whose
            # originating batch was never listed is still a real order.
            valid = ~present | out[local].isin(parents)

            invalid = int((~valid).sum())
            if invalid:
                dropped[f"{local}->{parent_table}"] = invalid
                LOG.warning("%s: %d row(s) reference a missing %s; dropped",
                            spec.table, invalid, parent_table)
            out = out[valid]

        return out.reset_index(drop=True), dropped

    @staticmethod
    def _insert(
        conn: sqlite3.Connection, spec: TableSpec, frame: pd.DataFrame
    ) -> int:
        """Insert with the conflict strategy the table's key type demands.

        Surrogate-key tables use ``INSERT OR IGNORE``: replacing a row would
        reissue its autoincrement id and silently orphan every child that
        references it. Natural-key tables use ``INSERT OR REPLACE``, which makes
        a reload idempotent.
        """
        columns = list(frame.columns)
        placeholders = ", ".join("?" * len(columns))
        conflict = "IGNORE" if spec.surrogate_key else "REPLACE"
        sql = (
            f"INSERT OR {conflict} INTO {spec.table} "
            f"({', '.join(columns)}) VALUES ({placeholders})"
        )

        payload = frame.astype(object).where(pd.notna(frame), None)
        rows = [tuple(record) for record in payload.itertuples(index=False, name=None)]

        cursor = conn.executemany(sql, rows)
        return int(cursor.rowcount if cursor.rowcount >= 0 else len(rows))

    # ── reporting ─────────────────────────────────────────────────────
    def row_counts(self) -> pd.DataFrame:
        """Row count per mapped table, for post-load verification."""
        return pd.DataFrame([
            {"table": spec.table, "rows": self.db.row_count(spec.table)}
            for spec in load_order()
        ])

    def integrity_check(self) -> pd.DataFrame:
        """Confirm no orphan survived the load.

        Runs after commit as an independent assertion. A load that reports
        success but leaves dangling references has not actually succeeded.
        """
        rows: list[dict[str, Any]] = []
        for spec in SCHEMA_MAP:
            for local, (parent_table, parent_column) in spec.foreign_keys.items():
                orphans = self.db.scalar(
                    f"SELECT COUNT(*) FROM {spec.table} c "
                    f"WHERE c.{local} IS NOT NULL AND NOT EXISTS ("
                    f"  SELECT 1 FROM {parent_table} p "
                    f"  WHERE p.{parent_column} = c.{local})"
                )
                rows.append({
                    "child": spec.table,
                    "column": local,
                    "parent": parent_table,
                    "orphans": int(orphans),
                    "status": "ok" if not orphans else "VIOLATION",
                })
        return pd.DataFrame(rows)


def load_processed_data(
    tables: Mapping[str, pd.DataFrame] | None = None,
    *,
    mode: str = REPLACE,
    db: Database | None = None,
) -> LoadReport:
    """Load pipeline output into the database.

    Reads the processed CSVs when no frames are supplied, so the loader can be
    run independently of the pipeline.
    """
    if tables is None:
        from freshsense.data.loaders import ProcessedDataLoader

        tables = ProcessedDataLoader().load_all()
    return DatabaseLoader(db).load(tables, mode=mode)


__all__ = [
    "DatabaseLoader", "LoadReport", "TableLoadResult", "load_processed_data",
    "REPLACE", "UPSERT",
]