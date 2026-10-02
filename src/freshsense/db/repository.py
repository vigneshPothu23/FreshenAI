"""
Repository layer.

This module isolates all SQL operations from the rest of the application.
Every feature (RAG, Recommendation, Monitoring, UI) interacts with SQLite
through repository classes instead of writing SQL directly.

Milestone 2 note — schema alignment
-----------------------------------
Three repositories previously targeted tables that do not exist in
``schema.sql``: ``inventory``, ``sales`` and ``metrics``. Those names belong to
the pre-normalisation analytical vocabulary the Sprint 1 pipeline emits; the
database normalises them into ``batches``, ``daily_item_stats`` and
``model_metrics``/``business_metrics``. Every call on those classes raised
``no such table``. The class names are unchanged — only the tables they address
and the columns they select. ``TABLE`` is retained on each class so any existing
caller reading it keeps working.

Two design rules the layer now enforces:

* **A page reports the total it was drawn from.** Rows without that total make a
  paged view lie: fifty of six hundred at-risk batches looks identical to all
  fifty that exist, and an operator under-reacts to the first.
* **Read-then-write runs in one transaction.** Streamlit re-executes the script
  on every interaction, so concurrent writers are the normal case. Two separate
  connections around a stock adjustment lose updates.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Sequence

import pandas as pd

from freshsense.db.database import Database
from freshsense.exceptions import DatabaseError
from freshsense.logging_config import get_logger

LOG = get_logger(__name__)


def _now() -> str:
    """Timestamp in the format every ``created_at`` column in the schema uses."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ══════════════════════════════════════════════════════════════════════════
@dataclass
class Page:
    """One page of rows, with the total the page was drawn from.

    Returning a slice without its total is what makes a paged UI misleading:
    the caller cannot tell a complete result from a truncated one.
    """

    rows: pd.DataFrame = field(default_factory=pd.DataFrame)
    total: int = 0
    page: int = 1
    page_size: int = 50

    @property
    def pages(self) -> int:
        return max(1, -(-self.total // max(self.page_size, 1)))

    @property
    def has_next(self) -> bool:
        return self.page < self.pages

    @property
    def has_previous(self) -> bool:
        return self.page > 1

    @property
    def is_truncated(self) -> bool:
        return self.total > len(self.rows)

    def summary(self) -> str:
        if self.total == 0:
            return "no matching rows"
        start = (self.page - 1) * self.page_size + 1
        return (f"showing {start}-{start + len(self.rows) - 1} of "
                f"{self.total:,} (page {self.page} of {self.pages})")


class BaseRepository:
    """Base repository providing common database helpers."""

    #: Destination table. Subclasses override.
    TABLE: str = ""

    def __init__(self, database: Database | None = None):
        self.db = database or Database()

    # ── passthrough helpers (unchanged public surface) ────────────────
    def execute(self, sql: str, params: tuple | dict = ()):
        return self.db.execute(sql, params)

    def query(self, sql: str, params: tuple | dict = ()):
        return self.db.query(sql, params)

    def query_one(self, sql: str, params: tuple | dict = ()):
        return self.db.query_one(sql, params)

    def scalar(self, sql: str, params: tuple = (), default: Any = None):
        return self.db.scalar(sql, params, default)

    # ── shared generic operations ─────────────────────────────────────
    def count(self) -> int:
        """Total rows in this repository's table."""
        return int(self.scalar(f"SELECT COUNT(*) FROM {self.TABLE}", default=0) or 0)

    def load_dataframe(self) -> pd.DataFrame:
        """Every row, as a DataFrame."""
        return self.query(f"SELECT * FROM {self.TABLE}")

    def insert_dataframe(self, frame: pd.DataFrame) -> int:
        """Bulk-insert a frame already using this table's column names.

        Pipeline output must go through the Milestone 1 ``DatabaseLoader``
        instead: it applies the schema mapping. This method assumes the frame is
        already in the database's own vocabulary.
        """
        return self.db.write_frame(frame, self.TABLE)

    def truncate(self) -> None:
        """Delete every row, preserving the table."""
        self.db.truncate(self.TABLE)

    def exists(self) -> bool:
        """Whether this repository's table is present in the schema."""
        return self.db.table_exists(self.TABLE)

    def columns(self) -> list[str]:
        return self.db.table_columns(self.TABLE)

    # ── internals shared by CRUD implementations ──────────────────────
    def _filtered_payload(self, record: dict[str, Any],
                          *, drop: Iterable[str] = ()) -> dict[str, Any]:
        """Keep only keys that are real columns, warning about the rest.

        Silently dropping unknown keys hides a caller's typo; raising on them
        makes a superset dictionary unusable. Warning does both jobs.
        """
        valid = set(self.columns())
        excluded = set(drop)
        payload = {k: v for k, v in record.items()
                   if k in valid and k not in excluded}
        unknown = sorted(set(record) - valid - excluded)
        if unknown:
            LOG.warning("%s: ignoring column(s) not in the schema: %s",
                        self.TABLE, unknown)
        return payload

    def _insert(self, record: dict[str, Any], *, conflict: str = "",
                drop: Iterable[str] = ()) -> int:
        payload = self._filtered_payload(record, drop=drop)
        if not payload:
            raise DatabaseError(
                f"Cannot insert into '{self.TABLE}': the record shares none of "
                f"its columns. Expected keys such as {sorted(self.columns())[:5]}."
            )
        clause = f" OR {conflict}" if conflict else ""
        placeholders = ", ".join(["?"] * len(payload))
        return self.db.execute(
            f"INSERT{clause} INTO {self.TABLE} ({', '.join(payload)}) "
            f"VALUES ({placeholders})",
            list(payload.values()),
        )

    def _update(self, key_column: str, key_value: Any,
                updates: dict[str, Any]) -> int:
        payload = self._filtered_payload(updates, drop=[key_column])
        if not payload:
            return 0
        assignment = ", ".join(f"{k} = ?" for k in payload)
        return self.db.execute(
            f"UPDATE {self.TABLE} SET {assignment} WHERE {key_column} = ?",
            [*payload.values(), key_value],
        )

    def _delete(self, key_column: str, key_value: Any) -> int:
        return self.db.execute(
            f"DELETE FROM {self.TABLE} WHERE {key_column} = ?", (key_value,)
        )

    def _paginate(self, where: str, params: Sequence[Any], order_by: str,
                  page: int, page_size: int) -> Page:
        """Count and slice with one shared filter, so the total always matches."""
        page = max(1, int(page))
        page_size = max(1, min(int(page_size), 500))
        total = int(self.scalar(
            f"SELECT COUNT(*) FROM {self.TABLE}{where}", tuple(params), default=0
        ) or 0)
        rows = self.query(
            f"SELECT * FROM {self.TABLE}{where} ORDER BY {order_by} "
            f"LIMIT ? OFFSET ?",
            (*params, page_size, (page - 1) * page_size),
        )
        return Page(rows=rows, total=total, page=page, page_size=page_size)

    @staticmethod
    def _check_order_by(order_by: str, allowed: set[str]) -> str:
        """Whitelist a sort expression.

        ``ORDER BY`` cannot be parameterised, so the value is interpolated into
        SQL and must never come from user input unchecked.
        """
        if order_by not in allowed:
            raise ValueError(
                f"order_by must be one of {sorted(allowed)}, got {order_by!r}"
            )
        return order_by


# ══════════════════════════════════════════════════════════════════════════
class InventoryRepository(BaseRepository):
    """Inventory table operations.

    Targets ``batches``, the normalised stock fact. The former ``inventory``
    target does not exist in the schema, so every method on this class raised
    ``no such table`` before Milestone 2.
    """

    TABLE = "batches"

    _ORDERABLE = {
        "action_priority DESC", "action_priority ASC",
        "days_to_expiry ASC", "days_to_expiry DESC",
        "quantity_available ASC", "quantity_available DESC",
        "inventory_value DESC", "capital_at_risk DESC",
        "product_name ASC", "expiry_date ASC",
    }
    _FILTERABLE = {"product_name", "category", "zone", "seller_id",
                   "quality_grade", "storage_type", "unit", "status"}

    # ── read ──────────────────────────────────────────────────────────
    def get(self, batch_id: str) -> dict[str, Any] | None:
        """One batch by its primary key."""
        return self.query_one(
            f"SELECT * FROM {self.TABLE} WHERE batch_id = ?", (batch_id,)
        )

    def active(self, limit: int | None = None) -> pd.DataFrame:
        """Batches still on shelf, most urgent first."""
        sql = (f"SELECT * FROM {self.TABLE} WHERE status = 'active' "
               f"AND quantity_available > 0 ORDER BY action_priority DESC")
        if limit:
            sql += f" LIMIT {int(limit)}"
        return self.query(sql)

    def _where(
        self,
        *,
        product: str | None = None,
        category: str | None = None,
        zone: str | None = None,
        seller_id: str | None = None,
        grade: str | None = None,
        status: str | None = None,
        max_days_to_expiry: int | None = None,
        min_quantity: float | None = None,
        active_only: bool = True,
    ) -> tuple[str, list[Any]]:
        """Build the shared WHERE clause used by search, count and pagination.

        One builder for all three, because a total computed from a differently
        built filter silently stops describing the rows beside it.
        """
        clauses: list[str] = []
        params: list[Any] = []

        if active_only:
            clauses.append("status = 'active' AND quantity_available > 0")
        elif status:
            clauses.append("status = ?")
            params.append(status)
        if product:
            clauses.append("LOWER(product_name) LIKE ?")
            params.append(f"%{product.lower()}%")
        if category:
            clauses.append("category = ?")
            params.append(category)
        if zone:
            clauses.append("zone = ?")
            params.append(zone)
        if seller_id:
            clauses.append("seller_id = ?")
            params.append(seller_id)
        if grade:
            clauses.append("quality_grade = ?")
            params.append(grade)
        if max_days_to_expiry is not None:
            clauses.append("days_to_expiry <= ?")
            params.append(int(max_days_to_expiry))
        if min_quantity is not None:
            clauses.append("quantity_available >= ?")
            params.append(float(min_quantity))

        return (f" WHERE {' AND '.join(clauses)}" if clauses else ""), params

    def search(self, **filters: Any) -> pd.DataFrame:
        """Every batch matching the filters, unpaginated."""
        where, params = self._where(**filters)
        return self.query(
            f"SELECT * FROM {self.TABLE}{where} ORDER BY action_priority DESC",
            tuple(params),
        )

    def count_matching(self, **filters: Any) -> int:
        """How many batches match, independent of any page size."""
        where, params = self._where(**filters)
        return int(self.scalar(
            f"SELECT COUNT(*) FROM {self.TABLE}{where}", tuple(params), default=0
        ) or 0)

    def search_page(self, *, page: int = 1, page_size: int = 50,
                    order_by: str = "action_priority DESC",
                    **filters: Any) -> Page:
        """Paginated search returning rows *and* the matching total."""
        order_by = self._check_order_by(order_by, self._ORDERABLE)
        where, params = self._where(**filters)
        return self._paginate(where, params, order_by, page, page_size)

    def low_stock(self, threshold: float = 5) -> pd.DataFrame:
        """Batches at or below a quantity threshold.

        Now restricted to active stock: a sold-out batch trivially sits below
        every threshold and would swamp the result with rows needing no action.
        """
        return self.query(
            f"""
            SELECT *
            FROM {self.TABLE}
            WHERE status = 'active'
              AND quantity_available > 0
              AND quantity_available <= ?
            ORDER BY quantity_available ASC
            """,
            (float(threshold),),
        )

    def expiring_products(self, days: int = 2) -> pd.DataFrame:
        """Active batches expiring within ``days``.

        The lower bound matters: without ``>= 0`` this returned every already
        expired batch too, mixing stock that can still be sold with stock that
        must be written off.
        """
        return self.query(
            f"""
            SELECT *
            FROM {self.TABLE}
            WHERE status = 'active'
              AND quantity_available > 0
              AND days_to_expiry BETWEEN 0 AND ?
            ORDER BY days_to_expiry, action_priority DESC
            """,
            (int(days),),
        )

    def expired(self, limit: int = 100) -> pd.DataFrame:
        """Batches past expiry that still hold stock — the write-off queue."""
        return self.query(
            f"SELECT * FROM {self.TABLE} WHERE days_to_expiry < 0 "
            f"AND quantity_available > 0 ORDER BY days_to_expiry ASC LIMIT ?",
            (int(limit),),
        )

    def distinct(self, column: str) -> list[str]:
        """Distinct values of a whitelisted column, for UI filter menus."""
        if column not in self._FILTERABLE:
            raise ValueError(
                f"Column '{column}' is not permitted for distinct(); "
                f"allowed: {sorted(self._FILTERABLE)}"
            )
        frame = self.query(
            f"SELECT DISTINCT {column} AS value FROM {self.TABLE} "
            f"WHERE {column} IS NOT NULL ORDER BY {column}"
        )
        return frame["value"].astype(str).tolist() if not frame.empty else []

    # ── write ─────────────────────────────────────────────────────────
    def create(self, record: dict[str, Any]) -> str:
        """Insert one batch, generating an id when the caller supplies none."""
        record = dict(record)
        record.setdefault("batch_id", f"BATCH-{uuid.uuid4().hex[:10].upper()}")
        record.setdefault("created_at", _now())
        record["updated_at"] = _now()
        self._insert(record)
        LOG.info("Batch created: %s (%s)",
                 record["batch_id"], record.get("product_name"))
        return str(record["batch_id"])

    def update(self, batch_id: str, updates: dict[str, Any]) -> int:
        """Patch one batch. The primary key is never updatable."""
        payload = dict(updates)
        payload["updated_at"] = _now()
        return self._update("batch_id", batch_id, payload)

    def delete(self, batch_id: str, *, force: bool = False) -> int:
        """Delete one batch, refusing while orders still reference it.

        SQLite would raise a bare ``FOREIGN KEY constraint failed``, naming
        neither the batch nor the blocker. Counting first makes the message
        actionable, and ``force`` makes the cascade a decision rather than an
        accident.
        """
        orders = int(self.scalar(
            "SELECT COUNT(*) FROM orders WHERE batch_id = ?", (batch_id,), default=0
        ) or 0)

        if orders and not force:
            raise DatabaseError(
                f"Cannot delete batch '{batch_id}': {orders} order(s) still "
                f"reference it. Pass force=True to delete those orders too, "
                f"which removes them from revenue history."
            )

        with self.db.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if force:
                conn.execute("DELETE FROM orders WHERE batch_id = ?", (batch_id,))
            removed = conn.execute(
                f"DELETE FROM {self.TABLE} WHERE batch_id = ?", (batch_id,)
            ).rowcount

        if force and orders:
            LOG.warning("Batch %s force-deleted with %d order(s)", batch_id, orders)
        return int(removed)

    def adjust_quantity(self, batch_id: str, delta: float) -> float:
        """Apply a signed stock movement atomically; return the new quantity.

        Read and write share one ``BEGIN IMMEDIATE`` transaction. Split across
        two connections, two concurrent sales of the same batch can both read
        the old value and both write the same new one — a lost update. Under
        Streamlit, which re-runs the script on every interaction, that is the
        normal case rather than an edge case.

        An expired batch keeps its ``expired`` status: selling stock down does
        not make it fresh again.

        Raises:
            DatabaseError: If the batch does not exist, rather than treating a
                missing batch as one holding zero.
        """
        with self.db.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                f"SELECT quantity_available, status FROM {self.TABLE} "
                f"WHERE batch_id = ?",
                (batch_id,),
            ).fetchone()

            if row is None:
                raise DatabaseError(
                    f"Cannot adjust quantity: no batch '{batch_id}' exists."
                )

            new_quantity = max(0.0, float(row["quantity_available"]) + float(delta))
            if str(row["status"]) == "expired":
                status = "expired"
            else:
                status = "sold_out" if new_quantity <= 0 else "active"

            conn.execute(
                f"UPDATE {self.TABLE} SET quantity_available = ?, status = ?, "
                f"updated_at = ? WHERE batch_id = ?",
                (new_quantity, status, _now(), batch_id),
            )

        LOG.info("Batch %s quantity %+.1f -> %.1f (%s)",
                 batch_id, delta, new_quantity, status)
        return new_quantity

    # ── aggregation ───────────────────────────────────────────────────
    def kpi_summary(self) -> dict[str, float]:
        """Portfolio position across active stock."""
        row = self.query_one(
            f"""
            SELECT COUNT(*)                                   AS total_batches,
                   COALESCE(SUM(quantity_available), 0)       AS total_quantity,
                   COALESCE(SUM(inventory_value), 0)          AS inventory_value,
                   COALESCE(SUM(capital_at_risk), 0)          AS capital_at_risk,
                   COALESCE(AVG(environment_stress_index), 0) AS avg_stress,
                   SUM(CASE WHEN days_to_expiry BETWEEN 0 AND 2 THEN 1 ELSE 0 END)
                                                              AS at_risk_count,
                   COUNT(DISTINCT seller_id)                  AS sellers,
                   COUNT(DISTINCT product_name)               AS products
            FROM {self.TABLE}
            WHERE status = 'active' AND quantity_available > 0
            """
        )
        summary = {k: float(v or 0) for k, v in (row or {}).items()}
        # Expired batches carry status 'expired', so the active-only query above
        # cannot see them. Reporting a confident zero would hide the write-off.
        summary["expired_count"] = float(self.scalar(
            f"SELECT COUNT(*) FROM {self.TABLE} WHERE days_to_expiry < 0 "
            f"AND quantity_available > 0", default=0
        ) or 0)
        return summary

    def by_category(self) -> pd.DataFrame:
        return self.query(
            f"""
            SELECT category,
                   COUNT(*)                      AS batches,
                   SUM(quantity_available)       AS quantity,
                   SUM(inventory_value)          AS value,
                   AVG(environment_stress_index) AS avg_stress,
                   SUM(CASE WHEN days_to_expiry <= 2 THEN 1 ELSE 0 END) AS at_risk
            FROM {self.TABLE}
            WHERE status = 'active' AND quantity_available > 0
            GROUP BY category ORDER BY value DESC
            """
        )

    def by_zone(self) -> pd.DataFrame:
        return self.query(
            f"""
            SELECT zone,
                   COUNT(*)                  AS batches,
                   SUM(inventory_value)      AS value,
                   SUM(capital_at_risk)      AS capital_at_risk,
                   COUNT(DISTINCT seller_id) AS sellers
            FROM {self.TABLE}
            WHERE status = 'active' AND quantity_available > 0
            GROUP BY zone ORDER BY value DESC
            """
        )

    def with_item(self, limit: int = 100) -> pd.DataFrame:
        """Batches joined to the product catalogue.

        Exercises the ``item_id`` foreign key the loader resolves, so a broken
        link shows up as missing rows here rather than silently later.
        """
        return self.query(
            f"""
            SELECT b.batch_id, b.product_name, b.quantity_available,
                   b.days_to_expiry, i.item_id, i.shelf_life_days,
                   i.perishability_level
            FROM {self.TABLE} b
            JOIN items i ON i.item_id = b.item_id
            WHERE b.status = 'active'
            ORDER BY b.action_priority DESC LIMIT ?
            """,
            (int(limit),),
        )
# ══════════════════════════════════════════════════════════════════════════
class SalesRepository(BaseRepository):
    """Sales table operations.

    Targets ``daily_item_stats``, the normalised daily demand series. The former
    ``sales`` target does not exist in the schema.
    """

    TABLE = "daily_item_stats"

    _ORDERABLE = {"stat_date DESC", "stat_date ASC",
                  "units_sold DESC", "revenue DESC", "product_name ASC"}

    def series(self, product: str | None = None,
               days: int | None = None) -> pd.DataFrame:
        """Daily demand, optionally for one product and one trailing window.

        The window is measured from the latest date *present in the data*, not
        from today: a demo dataset loaded weeks later would otherwise return
        nothing at all.
        """
        clauses, params = [], []
        if product:
            clauses.append("product_name = ?")
            params.append(product)
        if days is not None:
            clauses.append(
                f"stat_date >= (SELECT date(MAX(stat_date), ?) FROM {self.TABLE})"
            )
            params.append(f"-{int(days)} days")
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""

        frame = self.query(
            f"SELECT * FROM {self.TABLE}{where} ORDER BY stat_date", tuple(params)
        )
        if not frame.empty:
            frame["stat_date"] = pd.to_datetime(frame["stat_date"])
        return frame.reset_index(drop=True)

    def products(self) -> list[str]:
        frame = self.query(
            f"SELECT DISTINCT product_name FROM {self.TABLE} ORDER BY product_name"
        )
        return frame["product_name"].tolist() if not frame.empty else []

    def get(self, stat_date: str, product_name: str) -> dict[str, Any] | None:
        """One product-day observation by its natural key."""
        return self.query_one(
            f"SELECT * FROM {self.TABLE} WHERE stat_date = ? AND product_name = ?",
            (stat_date, product_name),
        )

    def search_page(self, *, product: str | None = None,
                    page: int = 1, page_size: int = 50,
                    order_by: str = "stat_date DESC") -> Page:
        order_by = self._check_order_by(order_by, self._ORDERABLE)
        where, params = "", []
        if product:
            where, params = " WHERE product_name = ?", [product]
        return self._paginate(where, params, order_by, page, page_size)

    def upsert_stat(self, record: dict[str, Any]) -> int:
        """Insert or correct one product-day figure.

        The natural key is ``(stat_date, product_name)``, matching the schema's
        UNIQUE constraint, so re-observing a day corrects it rather than
        double-counting demand. A duplicated product-day is the exact defect
        Sprint 1 cleaning removes; the write path must not reintroduce one.

        Raises:
            DatabaseError: If either key component is missing.
        """
        if not record.get("stat_date") or not record.get("product_name"):
            raise DatabaseError(
                "A daily stat needs both 'stat_date' and 'product_name': they "
                "are its natural key, and without them the row would duplicate "
                "rather than correct an existing day."
            )
        return self._insert(record, conflict="REPLACE", drop=["stat_id"])

    def delete_stat(self, stat_date: str, product_name: str) -> int:
        return self.db.execute(
            f"DELETE FROM {self.TABLE} WHERE stat_date = ? AND product_name = ?",
            (stat_date, product_name),
        )

    def totals_by_day(self, days: int = 90) -> pd.DataFrame:
        return self.query(
            f"""
            SELECT stat_date,
                   SUM(units_sold)     AS units_sold,
                   SUM(revenue)        AS revenue,
                   AVG(ambient_temp_c) AS ambient_temp_c
            FROM {self.TABLE}
            GROUP BY stat_date ORDER BY stat_date DESC LIMIT ?
            """,
            (int(days),),
        ).sort_values("stat_date")

    def top_products(self, limit: int = 10) -> pd.DataFrame:
        return self.query(
            f"""
            SELECT product_name, category,
                   SUM(units_sold) AS units_sold,
                   SUM(revenue)    AS revenue,
                   AVG(units_sold) AS avg_daily
            FROM {self.TABLE}
            GROUP BY product_name, category
            ORDER BY revenue DESC LIMIT ?
            """,
            (int(limit),),
        )

    def coverage(self) -> dict[str, Any]:
        """Whether the series is the complete grid forecasting requires.

        A model fitted on a series with silent gaps mis-attributes weekly
        seasonality, so the gap must be visible before training rather than
        after a poor backtest.
        """
        row = self.query_one(
            f"""
            SELECT COUNT(*)                        AS rows,
                   COUNT(DISTINCT stat_date)       AS days,
                   COUNT(DISTINCT product_name)    AS products,
                   MIN(stat_date)                  AS first_day,
                   MAX(stat_date)                  AS last_day
            FROM {self.TABLE}
            """
        ) or {}
        expected = int(row.get("days", 0) or 0) * int(row.get("products", 0) or 0)
        row["expected_rows"] = expected
        row["is_complete_grid"] = bool(expected and row.get("rows") == expected)
        return row


# ══════════════════════════════════════════════════════════════════════════
class OrdersRepository(BaseRepository):
    """Orders table operations."""

    TABLE = "orders"

    _ORDERABLE = {"order_date DESC", "order_date ASC",
                  "order_value DESC", "quantity DESC"}

    def get(self, order_id: str) -> dict[str, Any] | None:
        return self.query_one(
            f"SELECT * FROM {self.TABLE} WHERE order_id = ?", (order_id,)
        )

    def recent(self, limit: int = 100) -> pd.DataFrame:
        return self.query(
            f"SELECT * FROM {self.TABLE} ORDER BY order_date DESC LIMIT ?",
            (int(limit),),
        )

    def by_buyer(self, buyer_id: str, limit: int = 100) -> pd.DataFrame:
        return self.query(
            f"SELECT * FROM {self.TABLE} WHERE buyer_id = ? "
            f"ORDER BY order_date DESC LIMIT ?",
            (buyer_id, int(limit)),
        )

    def by_seller(self, seller_id: str, limit: int = 100) -> pd.DataFrame:
        return self.query(
            f"SELECT * FROM {self.TABLE} WHERE seller_id = ? "
            f"ORDER BY order_date DESC LIMIT ?",
            (seller_id, int(limit)),
        )

    def search_page(self, *, buyer_id: str | None = None,
                    seller_id: str | None = None,
                    product: str | None = None,
                    page: int = 1, page_size: int = 50,
                    order_by: str = "order_date DESC") -> Page:
        order_by = self._check_order_by(order_by, self._ORDERABLE)
        clauses, params = [], []
        if buyer_id:
            clauses.append("buyer_id = ?")
            params.append(buyer_id)
        if seller_id:
            clauses.append("seller_id = ?")
            params.append(seller_id)
        if product:
            clauses.append("LOWER(product_name) LIKE ?")
            params.append(f"%{product.lower()}%")
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        return self._paginate(where, params, order_by, page, page_size)

    def create(self, record: dict[str, Any]) -> str:
        """Insert one order, generating an id when the caller supplies none."""
        record = dict(record)
        record.setdefault(
            "order_id", f"FL{datetime.now():%y%m%d}{uuid.uuid4().hex[:6].upper()}"
        )
        record.setdefault("order_date", _now())
        record.setdefault("created_at", _now())
        self._insert(record)
        LOG.info("Order created: %s", record["order_id"])
        return str(record["order_id"])

    def update(self, order_id: str, updates: dict[str, Any]) -> int:
        return self._update("order_id", order_id, updates)

    def cancel(self, order_id: str, *, reason: str = "") -> float:
        """Cancel an order and return its stock to the batch, atomically.

        Cancellation is not deletion. The order is a historical fact the
        recommender learned from and monitoring scored; removing the row would
        rewrite that history. Marking it and restoring the quantity happen in
        one transaction, so stock cannot be lost if the process dies between the
        two writes.

        Returns:
            The batch's resulting quantity, or 0.0 when the order referenced no
            batch — legitimate, for an order whose batch was never listed.

        Raises:
            DatabaseError: If the order does not exist.
        """
        with self.db.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                f"SELECT batch_id, quantity, payment_status FROM {self.TABLE} "
                f"WHERE order_id = ?",
                (order_id,),
            ).fetchone()

            if row is None:
                raise DatabaseError(f"Cannot cancel: no order '{order_id}' exists.")
            if str(row["payment_status"]) == "Cancelled":
                LOG.warning("Order %s is already cancelled; nothing to do", order_id)
                return 0.0

            conn.execute(
                f"UPDATE {self.TABLE} SET payment_status = 'Cancelled', "
                f"dispute_reason = ? WHERE order_id = ?",
                (reason or "Cancelled by operator", order_id),
            )

            restored, batch_id = 0.0, row["batch_id"]
            if batch_id:
                batch = conn.execute(
                    "SELECT quantity_available, status FROM batches "
                    "WHERE batch_id = ?",
                    (batch_id,),
                ).fetchone()
                if batch is not None:
                    restored = (float(batch["quantity_available"])
                                + float(row["quantity"]))
                    status = ("expired" if str(batch["status"]) == "expired"
                              else "active")
                    conn.execute(
                        "UPDATE batches SET quantity_available = ?, status = ?, "
                        "updated_at = ? WHERE batch_id = ?",
                        (restored, status, _now(), batch_id),
                    )

        LOG.info("Order %s cancelled; %.1f unit(s) returned to batch %s",
                 order_id, float(row["quantity"]), batch_id or "-")
        return restored

    def delete(self, order_id: str) -> int:
        """Hard-delete an order.

        Prefer :meth:`cancel`. This exists for genuine data-entry errors — a
        duplicate submission that never represented a real transaction — where
        keeping the row would corrupt the metrics rather than record history.
        """
        removed = self._delete("order_id", order_id)
        if removed:
            LOG.warning("Order %s hard-deleted; consider cancel() instead to "
                        "preserve history", order_id)
        return removed

    def interaction_matrix(self) -> pd.DataFrame:
        """Buyer x product quantity matrix, for collaborative filtering."""
        frame = self.query(
            f"SELECT buyer_id, product_name, SUM(quantity) AS quantity "
            f"FROM {self.TABLE} GROUP BY buyer_id, product_name"
        )
        if frame.empty:
            return pd.DataFrame()
        return frame.pivot_table(index="buyer_id", columns="product_name",
                                 values="quantity", fill_value=0)

    def business_kpis(self, days: int = 90) -> dict[str, float]:
        """Commercial headline figures over a trailing window.

        When the window is empty the query falls back to all time so a fresh
        install shows something — but returns ``window_applied`` so the caller
        knows which it received. All-time figures under a seven-day label
        silently overstate recent performance.
        """
        row = self.query_one(
            f"""
            SELECT COUNT(*)                               AS orders,
                   COALESCE(SUM(order_value), 0)          AS gmv,
                   COALESCE(SUM(buyer_savings), 0)        AS savings,
                   COALESCE(SUM(waste_prevented_kg), 0)   AS waste_prevented_kg,
                   COALESCE(AVG(distance_km), 0)          AS avg_distance_km,
                   COALESCE(AVG(dispute_raised), 0) * 100 AS dispute_rate_pct,
                   COUNT(DISTINCT buyer_id)               AS buyers,
                   COUNT(DISTINCT seller_id)              AS sellers
            FROM {self.TABLE}
            WHERE date(order_date) >= date('now', ?)
            """,
            (f"-{int(days)} days",),
        )
        windowed = bool(row and row.get("orders"))

        if not windowed:
            row = self.query_one(
                f"""
                SELECT COUNT(*)                               AS orders,
                       COALESCE(SUM(order_value), 0)          AS gmv,
                       COALESCE(SUM(buyer_savings), 0)        AS savings,
                       COALESCE(SUM(waste_prevented_kg), 0)   AS waste_prevented_kg,
                       COALESCE(AVG(distance_km), 0)          AS avg_distance_km,
                       COALESCE(AVG(dispute_raised), 0) * 100 AS dispute_rate_pct,
                       COUNT(DISTINCT buyer_id)               AS buyers,
                       COUNT(DISTINCT seller_id)              AS sellers
                FROM {self.TABLE}
                """
            )
            if row and row.get("orders"):
                LOG.info("No order in the last %d day(s); reporting all-time "
                         "figures instead", days)

        metrics = {k: float(v or 0) for k, v in (row or {}).items()}
        metrics["window_days"] = float(days)
        metrics["window_applied"] = float(windowed)
        return metrics

    def daily_revenue(self, limit: int = 90) -> pd.DataFrame:
        """Revenue per day, from the ``v_daily_revenue`` view."""
        return self.query(
            "SELECT * FROM v_daily_revenue ORDER BY day DESC LIMIT ?",
            (int(limit),),
        ).sort_values("day")

    def orphan_check(self) -> pd.DataFrame:
        """Orders whose foreign keys have no parent.

        An independent assertion after any load: a run that reports success but
        leaves dangling references has not actually succeeded.
        """
        rows = []
        for column, (parent, key) in {
            "buyer_id": ("buyers", "buyer_id"),
            "seller_id": ("sellers", "seller_id"),
            "batch_id": ("batches", "batch_id"),
        }.items():
            orphans = int(self.scalar(
                f"SELECT COUNT(*) FROM {self.TABLE} c WHERE c.{column} IS NOT NULL "
                f"AND NOT EXISTS (SELECT 1 FROM {parent} p WHERE p.{key} = c.{column})",
                default=0,
            ) or 0)
            rows.append({"column": column, "parent": parent, "orphans": orphans,
                         "status": "ok" if not orphans else "VIOLATION"})
        return pd.DataFrame(rows)
# ══════════════════════════════════════════════════════════════════════════
class _DimensionRepository(BaseRepository):
    """Shared CRUD for the buyer and seller dimensions.

    Both use a natural primary key supplied by the caller, so ``INSERT OR
    REPLACE`` is safe: nothing references an autoincrement id that replacing a
    row could reissue. ``items`` is the counter-example — it must never be
    upserted this way.
    """

    KEY: str = ""
    #: ``{child_table: child_column}`` blocking a delete, declared in
    #: deletion order — deepest child first, so a forced cascade never removes
    #: a row another child still points at.
    DEPENDANTS: dict[str, str] = {}

    def get(self, identifier: str) -> dict[str, Any] | None:
        return self.query_one(
            f"SELECT * FROM {self.TABLE} WHERE {self.KEY} = ?", (identifier,)
        )

    def upsert(self, record: dict[str, Any]) -> str:
        identifier = record.get(self.KEY)
        if not identifier:
            raise DatabaseError(
                f"Cannot write to '{self.TABLE}': record has no '{self.KEY}'. "
                f"Dimension rows use a natural key supplied by the caller, not "
                f"an autoincrement id."
            )
        self._insert(record, conflict="REPLACE")
        LOG.info("%s upserted: %s", self.TABLE, identifier)
        return str(identifier)

    def update(self, identifier: str, updates: dict[str, Any]) -> int:
        """Patch one row. The primary key itself is never updatable."""
        return self._update(self.KEY, identifier, updates)

    def dependants(self, identifier: str) -> dict[str, int]:
        return {
            table: int(self.scalar(
                f"SELECT COUNT(*) FROM {table} WHERE {column} = ?",
                (identifier,), default=0,
            ) or 0)
            for table, column in self.DEPENDANTS.items()
        }

    def delete(self, identifier: str, *, force: bool = False) -> int:
        """Delete one row, refusing while children still reference it."""
        blocking = {t: n for t, n in self.dependants(identifier).items() if n}

        if blocking and not force:
            detail = ", ".join(f"{n} row(s) in {t}" for t, n in blocking.items())
            raise DatabaseError(
                f"Cannot delete '{identifier}' from {self.TABLE}: {detail} "
                f"still reference it. Pass force=True to delete those rows too, "
                f"which destroys their history."
            )

        with self.db.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if force:
                # DEPENDANTS is declared in deletion order: the deepest child
                # first. For a seller that is orders, then batches — orders
                # reference batches, so removing batches first trips the very
                # foreign key the cascade exists to satisfy.
                for table, column in self.DEPENDANTS.items():
                    conn.execute(f"DELETE FROM {table} WHERE {column} = ?",
                                 (identifier,))
            removed = conn.execute(
                f"DELETE FROM {self.TABLE} WHERE {self.KEY} = ?", (identifier,)
            ).rowcount

        if force and blocking:
            LOG.warning("%s %s force-deleted with dependants: %s",
                        self.TABLE, identifier, blocking)
        return int(removed)

    def search(self, *, zone: str | None = None,
               name_like: str | None = None) -> pd.DataFrame:
        clauses, params = [], []
        if zone:
            clauses.append("zone = ?")
            params.append(zone)
        if name_like:
            column = "buyer_name" if self.TABLE == "buyers" else "seller_name"
            clauses.append(f"LOWER({column}) LIKE ?")
            params.append(f"%{name_like.lower()}%")
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        return self.query(f"SELECT * FROM {self.TABLE}{where}", tuple(params))


class BuyerRepository(_DimensionRepository):
    """Buyer table operations."""

    TABLE = "buyers"
    KEY = "buyer_id"
    DEPENDANTS = {"orders": "buyer_id"}

    _ORDERABLE = {"buyer_name ASC", "avg_order_value DESC",
                  "buyer_rating DESC", "total_orders DESC"}

    def search_page(self, *, zone: str | None = None, page: int = 1,
                    page_size: int = 50,
                    order_by: str = "buyer_name ASC") -> Page:
        order_by = self._check_order_by(order_by, self._ORDERABLE)
        where, params = ("", [])
        if zone:
            where, params = " WHERE zone = ?", [zone]
        return self._paginate(where, params, order_by, page, page_size)

    def by_type(self) -> pd.DataFrame:
        """Buyer mix by segment, for the recommendation engine's priors."""
        return self.query(
            f"""
            SELECT buyer_type,
                   COUNT(*)              AS buyers,
                   AVG(avg_order_value)  AS avg_order_value,
                   AVG(max_distance_km)  AS avg_radius_km,
                   AVG(price_sensitivity) AS avg_price_sensitivity
            FROM {self.TABLE}
            GROUP BY buyer_type ORDER BY buyers DESC
            """
        )

    def top_by_spend(self, limit: int = 10) -> pd.DataFrame:
        return self.query(
            f"""
            SELECT b.buyer_id, b.buyer_name, b.buyer_type, b.zone,
                   COUNT(o.order_id)                  AS orders,
                   COALESCE(SUM(o.order_value), 0)    AS total_spend
            FROM {self.TABLE} b
            LEFT JOIN orders o ON o.buyer_id = b.buyer_id
            GROUP BY b.buyer_id ORDER BY total_spend DESC LIMIT ?
            """,
            (int(limit),),
        )


class SellerRepository(_DimensionRepository):
    """Seller table operations."""

    TABLE = "sellers"
    KEY = "seller_id"
    #: Ordered so a forced cascade deletes children before parents.
    DEPENDANTS = {"orders": "seller_id", "batches": "seller_id"}

    _ORDERABLE = {"seller_rating DESC", "seller_name ASC",
                  "total_orders DESC", "dispute_rate_pct ASC"}

    def search_page(self, *, zone: str | None = None, page: int = 1,
                    page_size: int = 50,
                    order_by: str = "seller_rating DESC") -> Page:
        order_by = self._check_order_by(order_by, self._ORDERABLE)
        where, params = ("", [])
        if zone:
            where, params = " WHERE zone = ?", [zone]
        return self._paginate(where, params, order_by, page, page_size)

    def verified(self) -> pd.DataFrame:
        """FSSAI-verified sellers only."""
        return self.query(
            f"SELECT * FROM {self.TABLE} WHERE fssai_verified = 1 "
            f"ORDER BY seller_rating DESC"
        )

    def performance(self, limit: int = 20) -> pd.DataFrame:
        """Reputation set against actual stock and fulfilment."""
        return self.query(
            f"""
            SELECT s.seller_id, s.seller_name, s.zone, s.seller_rating,
                   s.dispute_rate_pct, s.fssai_verified,
                   COUNT(DISTINCT b.batch_id)      AS active_batches,
                   COALESCE(SUM(b.inventory_value), 0) AS inventory_value,
                   COUNT(DISTINCT o.order_id)      AS orders
            FROM {self.TABLE} s
            LEFT JOIN batches b
                   ON b.seller_id = s.seller_id AND b.status = 'active'
            LEFT JOIN orders o ON o.seller_id = s.seller_id
            GROUP BY s.seller_id
            ORDER BY s.seller_rating DESC LIMIT ?
            """,
            (int(limit),),
        )


class PartyRepository(BaseRepository):
    """Facade over both dimensions, for callers that need either.

    Additive: it delegates to :class:`BuyerRepository` and
    :class:`SellerRepository` rather than duplicating their SQL.
    """

    TABLE = "buyers"

    def __init__(self, database: Database | None = None):
        super().__init__(database)
        self.buyers_repo = BuyerRepository(self.db)
        self.sellers_repo = SellerRepository(self.db)

    def buyers(self) -> pd.DataFrame:
        return self.buyers_repo.load_dataframe()

    def sellers(self) -> pd.DataFrame:
        return self.sellers_repo.load_dataframe()

    def buyer(self, buyer_id: str) -> dict[str, Any] | None:
        return self.buyers_repo.get(buyer_id)

    def seller(self, seller_id: str) -> dict[str, Any] | None:
        return self.sellers_repo.get(seller_id)


# ══════════════════════════════════════════════════════════════════════════
class DocumentRepository(BaseRepository):
    """Knowledge-base document operations.

    ``search()`` previously filtered on ``documents.content``, which the schema
    does not define — chunk text lives in ``chunks.text``. The corrected query
    joins the two, so a keyword match reports the document *and* the passage
    that matched.
    """

    TABLE = "documents"

    def all_documents(self) -> pd.DataFrame:
        return self.query(f"SELECT * FROM {self.TABLE} ORDER BY ingested_at DESC")

    def get(self, document_id: int) -> dict[str, Any] | None:
        return self.query_one(
            f"SELECT * FROM {self.TABLE} WHERE document_id = ?", (int(document_id),)
        )

    def search(self, keyword: str) -> pd.DataFrame:
        """Find documents whose title or chunk text matches a keyword."""
        pattern = f"%{keyword.lower()}%"
        return self.query(
            f"""
            SELECT DISTINCT d.document_id, d.title, d.source, d.doc_type,
                            d.license, d.ingested_at,
                            c.chunk_id, c.section, c.text
            FROM {self.TABLE} d
            LEFT JOIN chunks c ON c.document_id = d.document_id
            WHERE LOWER(d.title) LIKE ?
               OR LOWER(COALESCE(c.text, '')) LIKE ?
            ORDER BY d.document_id, c.chunk_index
            """,
            (pattern, pattern),
        )

    def add_document(self, *, source: str, title: str, uri: str = "",
                     doc_type: str = "markdown", license_: str = "",
                     checksum: str = "") -> int:
        """Insert one document; returns its new ``document_id``."""
        return self.db.insert_returning_id(
            f"INSERT INTO {self.TABLE} (source, title, uri, doc_type, license, "
            f"checksum, ingested_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (source, title, uri, doc_type, license_, checksum, _now()),
        )

    def add_chunks(self, rows: Sequence[dict[str, Any]]) -> int:
        """Bulk-insert chunk rows for already-inserted documents."""
        if not rows:
            return 0
        payload = [
            (int(r["document_id"]), int(r.get("chunk_index", i)),
             r.get("section", ""), r["text"], int(r.get("token_count", 0)),
             r.get("vector_id"), _now())
            for i, r in enumerate(rows)
        ]
        return self.db.execute_many(
            "INSERT INTO chunks (document_id, chunk_index, section, text, "
            "token_count, vector_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            payload,
        )

    def chunks(self, document_id: int | None = None) -> pd.DataFrame:
        where, params = "", ()
        if document_id is not None:
            where, params = " WHERE c.document_id = ?", (int(document_id),)
        return self.query(
            f"""
            SELECT c.chunk_id, c.document_id, c.chunk_index, c.section, c.text,
                   c.token_count, c.vector_id, d.title, d.source
            FROM chunks c JOIN {self.TABLE} d ON d.document_id = c.document_id
            {where}
            ORDER BY c.document_id, c.chunk_index
            """,
            params,
        )

    def delete_document(self, document_id: int) -> int:
        """Delete a document. Its chunks cascade via the schema's FK."""
        return self._delete("document_id", int(document_id))

    def clear(self) -> None:
        """Empty both tables, chunks first so the foreign key holds."""
        self.db.truncate("chunks", self.TABLE)

    def stats(self) -> dict[str, int]:
        return {"documents": self.db.row_count(self.TABLE),
                "chunks": self.db.row_count("chunks")}


# ══════════════════════════════════════════════════════════════════════════
class ChatRepository(BaseRepository):
    """Stores user conversations."""

    TABLE = "chat_history"

    def history(self, session_id: str | None = None,
                limit: int = 200) -> pd.DataFrame:
        """Transcript, newest first, optionally for one session.

        ``session_id`` is optional so the original no-argument call still works.
        """
        where, params = "", []
        if session_id:
            where, params = " WHERE session_id = ?", [session_id]
        return self.query(
            f"SELECT * FROM {self.TABLE}{where} ORDER BY created_at DESC LIMIT ?",
            (*params, int(limit)),
        )

    def conversation(self, session_id: str, limit: int = 100) -> pd.DataFrame:
        """One session in reading order, oldest first."""
        return self.query(
            f"SELECT * FROM {self.TABLE} WHERE session_id = ? "
            f"ORDER BY created_at ASC, message_id ASC LIMIT ?",
            (session_id, int(limit)),
        )

    def append(self, *, session_id: str, role: str, content: str,
               sources: Any = None, latency_ms: int = 0, provider: str = "",
               refused: bool = False) -> int:
        """Add one turn; returns its ``message_id``."""
        return self.db.insert_returning_id(
            f"INSERT INTO {self.TABLE} (session_id, role, content, sources, "
            f"latency_ms, provider, refused, created_at) "
            f"VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (session_id, role, content,
             json.dumps(sources) if sources is not None else None,
             int(latency_ms), provider, int(bool(refused)), _now()),
        )

    def sessions(self) -> pd.DataFrame:
        return self.query(
            f"""
            SELECT session_id,
                   COUNT(*)              AS messages,
                   SUM(refused)          AS refusals,
                   ROUND(AVG(latency_ms), 1) AS avg_latency_ms,
                   MIN(created_at)       AS started,
                   MAX(created_at)       AS last_message
            FROM {self.TABLE}
            GROUP BY session_id ORDER BY last_message DESC
            """
        )

    def delete_session(self, session_id: str) -> int:
        return self._delete("session_id", session_id)

    def clear(self) -> None:
        self.db.truncate(self.TABLE)
# ══════════════════════════════════════════════════════════════════════════
class MetricsRepository(BaseRepository):
    """Monitoring metrics.

    Targets ``model_metrics``; the former ``metrics`` table does not exist. The
    schema also splits business KPIs into ``business_metrics``, which this class
    reaches through its own methods rather than a second class, so existing
    callers keep one entry point.

    ``latest()`` previously ordered by ``timestamp``, which is not a column
    here — the schema names it ``recorded_at``.
    """

    TABLE = "model_metrics"
    BUSINESS_TABLE = "business_metrics"

    def all_metrics(self) -> pd.DataFrame:
        return self.query(f"SELECT * FROM {self.TABLE} ORDER BY recorded_at DESC")

    def latest(self, limit: int = 20) -> pd.DataFrame:
        return self.query(
            f"SELECT * FROM {self.TABLE} ORDER BY recorded_at DESC LIMIT ?",
            (int(limit),),
        )

    def for_model(self, model_name: str, limit: int = 100) -> pd.DataFrame:
        return self.query(
            f"SELECT * FROM {self.TABLE} WHERE model_name = ? "
            f"ORDER BY recorded_at DESC LIMIT ?",
            (model_name, int(limit)),
        )

    def record(self, model_name: str, metrics: dict[str, float],
               dataset: str = "test") -> int:
        """Store a metric bundle for one model.

        Non-numeric values are skipped rather than coerced: a metric column that
        silently accepted a string would break every later aggregate over it.
        """
        rows = [(model_name, name, float(value), dataset, _now())
                for name, value in metrics.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)]
        if not rows:
            LOG.warning("record(%s): no numeric metric to store", model_name)
            return 0
        return self.db.execute_many(
            f"INSERT INTO {self.TABLE} (model_name, metric_name, metric_value, "
            f"dataset, recorded_at) VALUES (?, ?, ?, ?, ?)",
            rows,
        )

    def history(self, model_name: str, metric_name: str) -> pd.DataFrame:
        """One metric over time — how a regression becomes visible."""
        return self.query(
            f"SELECT recorded_at, metric_value FROM {self.TABLE} "
            f"WHERE model_name = ? AND metric_name = ? ORDER BY recorded_at",
            (model_name, metric_name),
        )

    def latest_per_metric(self, model_name: str) -> pd.DataFrame:
        """Most recent value of each metric for one model."""
        return self.query(
            f"""
            SELECT m.metric_name, m.metric_value, m.dataset, m.recorded_at
            FROM {self.TABLE} m
            JOIN (SELECT metric_name, MAX(recorded_at) AS latest
                  FROM {self.TABLE} WHERE model_name = ?
                  GROUP BY metric_name) t
              ON t.metric_name = m.metric_name AND t.latest = m.recorded_at
            WHERE m.model_name = ?
            ORDER BY m.metric_name
            """,
            (model_name, model_name),
        )

    def delete_for_model(self, model_name: str) -> int:
        return self._delete("model_name", model_name)

    # ── business metrics ──────────────────────────────────────────────
    def record_business_metric(self, metric_date: str, metric_name: str,
                               value: float, dimension: str = "") -> int:
        """Store one dated business KPI.

        ``INSERT OR REPLACE`` matches the table's UNIQUE
        ``(metric_date, metric_name, dimension)``, so re-recording a day
        corrects it instead of double-counting.
        """
        return self.db.execute(
            f"INSERT OR REPLACE INTO {self.BUSINESS_TABLE} "
            f"(metric_date, metric_name, metric_value, dimension) "
            f"VALUES (?, ?, ?, ?)",
            (metric_date, metric_name, float(value), dimension),
        )

    def business_metrics(self, metric_name: str | None = None) -> pd.DataFrame:
        where, params = "", ()
        if metric_name:
            where, params = " WHERE metric_name = ?", (metric_name,)
        return self.query(
            f"SELECT * FROM {self.BUSINESS_TABLE}{where} ORDER BY metric_date",
            params,
        )


# ══════════════════════════════════════════════════════════════════════════
class PredictionRepository(BaseRepository):
    """Model output, and the outcome table that makes monitoring possible.

    New in Milestone 2: ``predictions`` and ``prediction_outcomes`` exist in the
    schema but had no repository, so nothing could read or write them.
    """

    TABLE = "predictions"

    def record(self, *, entity_type: str, entity_id: str, prediction_type: str,
               value: float, label: str = "", confidence: float = 0.0,
               explanation: Any = None, model_name: str = "",
               model_version: str = "v1.0",
               horizon_date: str | None = None) -> int:
        return self.db.insert_returning_id(
            f"""
            INSERT INTO {self.TABLE}
                (entity_type, entity_id, prediction_type, value, label,
                 confidence, explanation, model_name, model_version,
                 horizon_date, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (entity_type, entity_id, prediction_type, float(value), label[:120],
             float(confidence),
             json.dumps(explanation) if explanation is not None else None,
             model_name, model_version, horizon_date, _now()),
        )

    def latest_for(self, entity_id: str,
                   prediction_type: str) -> dict[str, Any] | None:
        return self.query_one(
            f"SELECT * FROM {self.TABLE} WHERE entity_id = ? "
            f"AND prediction_type = ? ORDER BY created_at DESC LIMIT 1",
            (entity_id, prediction_type),
        )

    def recent(self, prediction_type: str | None = None,
               limit: int = 200) -> pd.DataFrame:
        where, params = "", []
        if prediction_type:
            where, params = " WHERE prediction_type = ?", [prediction_type]
        return self.query(
            f"SELECT * FROM {self.TABLE}{where} ORDER BY created_at DESC LIMIT ?",
            (*params, int(limit)),
        )

    def record_outcome(self, prediction_id: int, actual_value: float) -> int:
        """Score one prediction against what actually happened.

        The prediction is looked up rather than assumed. Without that, a missing
        one surfaced as a bare ``FOREIGN KEY constraint failed`` naming neither
        the id nor the cause — and a null predicted value silently became 0.0,
        recording an unscoreable prediction as a maximal miss and dragging
        measured accuracy down for a reason unrelated to the model.

        Raises:
            DatabaseError: If no such prediction exists.
        """
        row = self.query_one(
            f"SELECT value FROM {self.TABLE} WHERE prediction_id = ?",
            (int(prediction_id),),
        )
        if row is None:
            raise DatabaseError(
                f"Cannot record an outcome for prediction {prediction_id}: no "
                f"such prediction. Outcomes are scored against a prediction "
                f"that was already recorded — call record() first."
            )

        error = float(actual_value) - float(row["value"])
        return self.db.insert_returning_id(
            "INSERT INTO prediction_outcomes (prediction_id, actual_value, "
            "error, abs_error, observed_at) VALUES (?, ?, ?, ?, ?)",
            (int(prediction_id), float(actual_value), error, abs(error), _now()),
        )

    def outcomes(self, prediction_type: str | None = None) -> pd.DataFrame:
        where, params = "", []
        if prediction_type:
            where, params = " WHERE p.prediction_type = ?", [prediction_type]
        return self.query(
            f"""
            SELECT p.prediction_id, p.entity_id, p.prediction_type,
                   p.value AS predicted, p.label, p.confidence, p.model_name,
                   p.created_at, o.actual_value, o.error, o.abs_error,
                   o.observed_at
            FROM {self.TABLE} p
            JOIN prediction_outcomes o ON o.prediction_id = p.prediction_id
            {where}
            ORDER BY o.observed_at DESC
            """,
            tuple(params),
        )

    def unscored(self, limit: int = 500) -> pd.DataFrame:
        """Predictions with no recorded outcome — the monitoring backlog."""
        return self.query(
            f"""
            SELECT p.* FROM {self.TABLE} p
            LEFT JOIN prediction_outcomes o ON o.prediction_id = p.prediction_id
            WHERE o.outcome_id IS NULL
            ORDER BY p.created_at DESC LIMIT ?
            """,
            (int(limit),),
        )


# ══════════════════════════════════════════════════════════════════════════
class RecommendationRepository(BaseRepository):
    """Persisted recommendations plus the acceptance signal.

    New in Milestone 2: the ``recommendations`` table had no repository.
    """

    TABLE = "recommendations"

    def save_many(self, rows: Sequence[dict[str, Any]]) -> int:
        if not rows:
            return 0
        payload = [
            (r["rec_type"], r["entity_type"], str(r["entity_id"]),
             r.get("target_type"), r.get("target_id"),
             float(r.get("score", 0)), int(r.get("rank", 0)),
             r.get("rationale", ""),
             json.dumps(r.get("payload")) if r.get("payload") is not None else None,
             r.get("algorithm", ""), _now())
            for r in rows
        ]
        return self.db.execute_many(
            f"""
            INSERT INTO {self.TABLE}
                (rec_type, entity_type, entity_id, target_type, target_id,
                 score, rank, rationale, payload, algorithm, shown_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            payload,
        )

    def recent(self, rec_type: str | None = None,
               limit: int = 100) -> pd.DataFrame:
        where, params = "", []
        if rec_type:
            where, params = " WHERE rec_type = ?", [rec_type]
        return self.query(
            f"SELECT * FROM {self.TABLE}{where} "
            f"ORDER BY shown_at DESC, rank ASC LIMIT ?",
            (*params, int(limit)),
        )

    def accept(self, recommendation_id: int) -> int:
        return self.db.execute(
            f"UPDATE {self.TABLE} SET accepted_at = ? WHERE recommendation_id = ?",
            (_now(), int(recommendation_id)),
        )

    def dismiss(self, recommendation_id: int, *, reason: str = "") -> int:
        """Mark a recommendation seen and rejected.

        Distinct from deleting it. An operator who declines a discount supplies
        a negative label, and acceptance rate only means something when the
        denominator includes the rejections — deleting them would make every
        recommendation look accepted.
        """
        return self.db.execute(
            f"UPDATE {self.TABLE} SET rationale = ? WHERE recommendation_id = ?",
            (f"[dismissed] {reason}".strip(), int(recommendation_id)),
        )

    def delete(self, recommendation_id: int) -> int:
        return self._delete("recommendation_id", int(recommendation_id))

    def prune(self, *, keep_days: int = 30) -> int:
        """Delete stale, never-acted-on recommendations.

        Accepted rows are always kept: they are the outcome record monitoring
        scores against, and pruning them would inflate the acceptance rate of
        whatever remains.
        """
        removed = self.db.execute(
            f"DELETE FROM {self.TABLE} WHERE accepted_at IS NULL "
            f"AND shown_at < datetime('now', ?)",
            (f"-{int(keep_days)} days",),
        )
        if removed:
            LOG.info("Pruned %d recommendation(s) older than %d day(s)",
                     removed, keep_days)
        return removed

    def acceptance_rate(self) -> float:
        total = int(self.scalar(f"SELECT COUNT(*) FROM {self.TABLE}",
                                default=0) or 0)
        if not total:
            return 0.0
        accepted = int(self.scalar(
            f"SELECT COUNT(*) FROM {self.TABLE} WHERE accepted_at IS NOT NULL",
            default=0,
        ) or 0)
        return round(accepted / total * 100, 2)


# ══════════════════════════════════════════════════════════════════════════
class MonitoringRepository(BaseRepository):
    """System logs, the LLM call ledger and agent execution traces.

    New in Milestone 2: ``system_logs``, ``llm_calls``, ``agent_runs`` and
    ``agent_steps`` exist in the schema but had no repository.
    """

    TABLE = "system_logs"

    def log_event(self, *, component: str, action: str, status: str = "ok",
                  latency_ms: int = 0, detail: str = "",
                  metadata: Any = None) -> int:
        return self.db.insert_returning_id(
            f"INSERT INTO {self.TABLE} (component, action, status, latency_ms, "
            f"detail, metadata, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (component, action, status, int(latency_ms), str(detail)[:500],
             json.dumps(metadata) if metadata is not None else None, _now()),
        )

    def events(self, limit: int = 300) -> pd.DataFrame:
        return self.query(
            f"SELECT * FROM {self.TABLE} ORDER BY created_at DESC LIMIT ?",
            (int(limit),),
        )

    def component_stats(self) -> pd.DataFrame:
        return self.query(
            f"""
            SELECT component,
                   COUNT(*)                  AS calls,
                   ROUND(AVG(latency_ms), 1) AS avg_latency_ms,
                   MAX(latency_ms)           AS max_latency_ms,
                   ROUND(AVG(CASE WHEN status = 'ok' THEN 1.0 ELSE 0 END) * 100, 1)
                                             AS success_rate
            FROM {self.TABLE}
            GROUP BY component ORDER BY calls DESC
            """
        )

    def prune(self, *, keep_days: int = 90) -> int:
        """Delete old events so percentile queries stay fast over time."""
        removed = self.db.execute(
            f"DELETE FROM {self.TABLE} WHERE created_at < datetime('now', ?)",
            (f"-{int(keep_days)} days",),
        )
        if removed:
            LOG.info("Pruned %d system log row(s) older than %d day(s)",
                     removed, keep_days)
        return removed

    # ── LLM ledger ────────────────────────────────────────────────────
    def log_llm_call(self, *, provider: str, model: str = "",
                     prompt_chars: int = 0, completion_chars: int = 0,
                     latency_ms: int = 0, status: str = "ok",
                     error: str = "", run_id: str | None = None) -> int:
        return self.db.insert_returning_id(
            "INSERT INTO llm_calls (run_id, provider, model, prompt_chars, "
            "completion_chars, latency_ms, status, error, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, provider, model, int(prompt_chars), int(completion_chars),
             int(latency_ms), status, str(error)[:300], _now()),
        )

    def llm_stats(self) -> pd.DataFrame:
        return self.query(
            """
            SELECT provider,
                   COUNT(*)                             AS calls,
                   ROUND(AVG(latency_ms), 1)            AS avg_latency_ms,
                   SUM(prompt_chars + completion_chars) AS total_chars,
                   ROUND(AVG(CASE WHEN status = 'ok' THEN 1.0 ELSE 0 END) * 100, 1)
                                                        AS success_rate
            FROM llm_calls GROUP BY provider ORDER BY calls DESC
            """
        )

    # ── agent traces ──────────────────────────────────────────────────
    def start_run(self, run_id: str, user_query: str,
                  session_id: str = "") -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO agent_runs (run_id, session_id, user_query, "
            "status, started_at) VALUES (?, ?, ?, 'running', ?)",
            (run_id, session_id, str(user_query)[:500], _now()),
        )

    def finish_run(self, run_id: str, *, final_output: str = "",
                   status: str = "completed", latency_ms: int = 0,
                   replan_count: int = 0) -> int:
        return self.db.execute(
            "UPDATE agent_runs SET final_output = ?, status = ?, "
            "total_latency_ms = ?, replan_count = ?, completed_at = ? "
            "WHERE run_id = ?",
            (str(final_output)[:2000], status, int(latency_ms),
             int(replan_count), _now(), run_id),
        )

    def log_steps(self, run_id: str, steps: Sequence[dict[str, Any]]) -> int:
        if not steps:
            return 0
        payload = [
            (run_id, int(s.get("step_index", i)), s.get("agent_name", ""),
             s.get("action", ""), s.get("tool_called", ""),
             str(s.get("detail", ""))[:500], s.get("status", "ok"),
             int(s.get("latency_ms", 0)), _now())
            for i, s in enumerate(steps)
        ]
        return self.db.execute_many(
            "INSERT INTO agent_steps (run_id, step_index, agent_name, action, "
            "tool_called, detail, status, latency_ms, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            payload,
        )

    def runs(self, limit: int = 50) -> pd.DataFrame:
        return self.query(
            "SELECT * FROM agent_runs ORDER BY started_at DESC LIMIT ?",
            (int(limit),),
        )

    def steps(self, run_id: str) -> pd.DataFrame:
        return self.query(
            "SELECT * FROM agent_steps WHERE run_id = ? ORDER BY step_index",
            (run_id,),
        )

    def agent_stats(self) -> pd.DataFrame:
        return self.query(
            """
            SELECT agent_name,
                   COUNT(*)                  AS steps,
                   ROUND(AVG(latency_ms), 1) AS avg_latency_ms,
                   SUM(CASE WHEN status = 'replan' THEN 1 ELSE 0 END) AS replans,
                   ROUND(AVG(CASE WHEN status IN ('ok', 'pending')
                                  THEN 1.0 ELSE 0 END) * 100, 1) AS success_rate
            FROM agent_steps GROUP BY agent_name ORDER BY steps DESC
            """
        )


__all__ = [
    "Page",
    "BaseRepository",
    "InventoryRepository",
    "SalesRepository",
    "OrdersRepository",
    "BuyerRepository",
    "SellerRepository",
    "PartyRepository",
    "DocumentRepository",
    "ChatRepository",
    "MetricsRepository",
    "PredictionRepository",
    "RecommendationRepository",
    "MonitoringRepository",
]