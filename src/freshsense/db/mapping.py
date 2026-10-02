"""Canonical schema mapping between the analytical and operational models.

The Sprint 1 pipeline emits wide, denormalised, ``Title_Case`` frames — the
correct shape for analysis and modelling. The SQLite schema is normalised,
FK-linked and ``snake_case`` — the correct shape for transactions. Both are
right; what was missing is the contract that translates between them.

That absence is why persistence failed. ``Database.table_columns()`` returns an
empty list for a table that does not exist, so writing a frame to an unmapped
name silently produced a zero-column payload and pandas emitted
``CREATE TABLE inventory ()``. A missing table and a fully-mismatched frame were
indistinguishable.

This module makes the translation explicit and, through
:func:`verify_against_schema`, checkable: every mapped target column is
validated against the live schema, so drift between this file and
``schema.sql`` fails a test rather than a production write.

Direction matters. ``column_map`` is declared **pipeline -> database**, which is
the write direction. :func:`to_pipeline_columns` inverts it for the read
direction already used by the agents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from freshsense.logging_config import get_logger

LOG = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class TableSpec:
    """How one pipeline frame becomes one database table.

    Attributes:
        table: Destination table in ``schema.sql``.
        source: Key of the pipeline frame in ``PipelineResult.tables``.
        column_map: ``{pipeline_column: database_column}``. Columns absent from
            the source frame are skipped; columns absent from the destination
            table are a contract violation reported by
            :func:`verify_against_schema`.
        key: Natural key used for idempotent upsert and duplicate detection.
        depends_on: Tables that must be loaded first to satisfy foreign keys.
        foreign_keys: ``{local_db_column: (parent_table, parent_column)}``. The
            loader drops orphan rows rather than letting SQLite abort the whole
            transaction.
        date_columns: Database columns normalised to ``YYYY-MM-DD`` strings.
        surrogate_key: True when the primary key is an autoincrement integer
            other rows reference. Such tables must be inserted with
            ``INSERT OR IGNORE``, never ``INSERT OR REPLACE``, because replacing
            a row reissues its id and silently orphans every child.
        derive: Hook applied to the source frame *before* mapping, for values
            the schema needs that the pipeline does not emit directly.
        description: Why this table exists, for the architecture docs.
    """

    table: str
    source: str
    column_map: dict[str, str]
    key: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    foreign_keys: dict[str, tuple[str, str]] = field(default_factory=dict)
    date_columns: tuple[str, ...] = ()
    surrogate_key: bool = False
    derive: Callable[[pd.DataFrame], pd.DataFrame] | None = None
    description: str = ""

    @property
    def target_columns(self) -> list[str]:
        return list(self.column_map.values())


# ══════════════════════════════════════════════════════════════════════════
# Derivations — values the schema requires that the pipeline does not emit
# ══════════════════════════════════════════════════════════════════════════
def _derive_items(inventory: pd.DataFrame) -> pd.DataFrame:
    """Build the product catalogue from the batches observed in inventory.

    ``items`` is a reference dimension the raw extracts never supplied: the
    pipeline's inventory frame inlines product attributes on every batch row.
    Collapsing to one row per product recovers the dimension. Shelf life and
    perishability take the modal value per product, because a product must have
    one catalogue entry even when individual batches disagree.
    """
    if inventory.empty:
        return pd.DataFrame()

    def _mode(series: pd.Series) -> Any:
        modes = series.mode()
        return modes.iloc[0] if len(modes) else None

    return (
        inventory.groupby("Product_Name", as_index=False)
        .agg(
            Category=("Category", _mode),
            Unit=("Unit", _mode),
            Shelf_Life_Days=("Shelf_Life_Days", "median"),
            Perishability_Level=("Perishability_Level", _mode),
            Storage_Type=("Storage_Type", _mode),
        )
        .assign(Shelf_Life_Days=lambda d: d["Shelf_Life_Days"].round().astype(int))
    )


def _derive_batches(inventory: pd.DataFrame) -> pd.DataFrame:
    """Supply the one batch field the snapshot cannot observe.

    ``quantity_initial`` is the amount received at intake. The extract is a
    point-in-time snapshot with no intake record, so the only defensible value
    is the quantity currently on hand — anything else would invent a depletion
    history. Set explicitly rather than left to default to 0, which would leave
    the row internally inconsistent.
    """
    if inventory.empty:
        return inventory
    out = inventory.copy()
    out["Quantity_Initial"] = out["Quantity_Available"]
    return out


# ══════════════════════════════════════════════════════════════════════════
# The contract, in dependency order
# ══════════════════════════════════════════════════════════════════════════
SELLERS = TableSpec(
    table="sellers",
    source="sellers",
    description="Seller dimension: reputation and location, referenced by batches.",
    key=("seller_id",),
    date_columns=("onboarded_date",),
    column_map={
        "Seller_ID": "seller_id",
        "Seller_Name": "seller_name",
        "Store_Type": "store_type",
        "Zone": "zone",
        "Latitude": "latitude",
        "Longitude": "longitude",
        "Seller_Rating": "seller_rating",
        "Total_Orders_Fulfilled": "total_orders",
        "Dispute_Rate_Pct": "dispute_rate_pct",
        "FSSAI_Verified": "fssai_verified",
        "Onboarded_Date": "onboarded_date",
    },
)

BUYERS = TableSpec(
    table="buyers",
    source="buyers",
    description="Buyer dimension: matching constraints and purchasing capacity.",
    key=("buyer_id",),
    column_map={
        "Buyer_ID": "buyer_id",
        "Buyer_Name": "buyer_name",
        "Buyer_Type": "buyer_type",
        "Zone": "zone",
        "Latitude": "latitude",
        "Longitude": "longitude",
        "Avg_Order_Value": "avg_order_value",
        "Price_Sensitivity": "price_sensitivity",
        "Max_Distance_Km": "max_distance_km",
        "Min_Acceptable_Grade": "min_acceptable_grade",
        "Accepts_Split_Order": "accepts_split_order",
        "Buyer_Rating": "buyer_rating",
        "Total_Orders_Placed": "total_orders",
    },
)

ITEMS = TableSpec(
    table="items",
    source="inventory",
    description="Product catalogue, derived by collapsing inventory to one row "
                "per product. Normalises attributes the pipeline inlines.",
    key=("name",),
    surrogate_key=True,
    derive=_derive_items,
    column_map={
        "Product_Name": "name",
        "Category": "category",
        "Unit": "unit",
        "Shelf_Life_Days": "shelf_life_days",
        "Perishability_Level": "perishability_level",
        "Storage_Type": "storage_type",
    },
)

BATCHES = TableSpec(
    table="batches",
    source="inventory",
    description="Central fact: a stock batch with identity, environment, "
                "commercial position and outcome.",
    key=("batch_id",),
    depends_on=("sellers", "items"),
    foreign_keys={"seller_id": ("sellers", "seller_id")},
    date_columns=("manufacturing_date", "arrival_date", "expiry_date"),
    derive=_derive_batches,
    column_map={
        # identity
        "Inventory_ID": "batch_id",
        "Seller_ID": "seller_id",
        "Product_Name": "product_name",
        "Category": "category",
        "Brand": "brand",
        "Unit": "unit",
        "Zone": "zone",
        "Latitude": "latitude",
        "Longitude": "longitude",
        # quantity
        "Quantity_Initial": "quantity_initial",
        "Quantity_Available": "quantity_available",
        # shelf-life position
        "Manufacturing_Date": "manufacturing_date",
        "Arrival_Date": "arrival_date",
        "Expiry_Date": "expiry_date",
        "Shelf_Life_Days": "shelf_life_days",
        "Stock_Age_Days": "stock_age_days",
        "Days_To_Expiry": "days_to_expiry",
        "Age_Ratio": "age_ratio",
        # environment
        "Storage_Type": "storage_type",
        "Storage_Temperature_C": "storage_temperature_c",
        "Ambient_Temp_C": "ambient_temp_c",
        "Humidity_Pct": "humidity_pct",
        "Temp_Breach_Hours": "temp_breach_hours",
        "Perishability_Level": "perishability_level",
        "Perishability_Score": "perishability_score",
        "Storage_Risk_Score": "storage_risk_score",
        # commercial
        "Cost_Price": "cost_price",
        "Selling_Price": "mrp",
        "Discount_Pct": "discount_pct",
        "Effective_Price": "effective_price",
        "Daily_Avg_Sales": "daily_avg_sales",
        # outcome and derived state
        "Quality_Grade": "quality_grade",
        "Status": "status",
        "Is_Spoiled": "is_spoiled",
        "Waste_Quantity": "waste_quantity",
        "Environment_Stress_Index": "environment_stress_index",
        "Action_Priority": "action_priority",
        "Inventory_Value": "inventory_value",
        "Capital_At_Risk": "capital_at_risk",
        "Projected_Surplus": "projected_surplus",
        # data-quality provenance, carried through for Sprint 6 drift analysis
        "Has_Humidity_Sensor": "has_humidity_sensor",
        "Has_Temp_Logger": "has_temp_logger",
        "Is_New_Seller": "is_new_seller",
    },
)

DAILY_ITEM_STATS = TableSpec(
    table="daily_item_stats",
    source="sales",
    description="Denormalised daily demand series. Deliberately not normalised: "
                "dashboard and forecasting queries are O(days), not O(events).",
    key=("stat_date", "product_name"),
    date_columns=("stat_date",),
    column_map={
        "Date": "stat_date",
        "Product_Name": "product_name",
        "Category": "category",
        "Unit": "unit",
        "Units_Sold": "units_sold",
        "Avg_Selling_Price": "avg_selling_price",
        "Revenue": "revenue",
        "Ambient_Temp_C": "ambient_temp_c",
        "Humidity_Pct": "humidity_pct",
        "Is_Weekend": "is_weekend",
        "Is_Festival": "is_festival",
        "Day_Of_Week": "day_of_week",
    },
)

ORDERS = TableSpec(
    table="orders",
    source="orders",
    description="Transaction log: the interaction matrix for Sprint 3 and the "
                "outcome source for Sprint 6.",
    key=("order_id",),
    depends_on=("buyers", "sellers", "batches"),
    foreign_keys={
        "buyer_id": ("buyers", "buyer_id"),
        "seller_id": ("sellers", "seller_id"),
        "batch_id": ("batches", "batch_id"),
    },
    date_columns=("order_date",),
    column_map={
        "Order_ID": "order_id",
        "Order_Date": "order_date",
        "Buyer_ID": "buyer_id",
        "Seller_ID": "seller_id",
        "Inventory_ID": "batch_id",
        "Product_Name": "product_name",
        "Category": "category",
        "Quantity_Ordered": "quantity",
        "Unit": "unit",
        "Unit_Price_Paid": "unit_price",
        "MRP_Unit_Price": "mrp",
        "Order_Value_INR": "order_value",
        "Buyer_Savings_INR": "buyer_savings",
        "Distance_Km": "distance_km",
        "Quality_Grade": "quality_grade",
        "Match_Source": "match_source",
        "Fulfilment": "fulfilment",
        "Payment_Status": "payment_status",
        "Is_Disputed": "dispute_raised",
        "Dispute_Reason": "dispute_reason",
        "Buyer_Rating_Given": "buyer_rating_given",
        "Awaiting_Rating": "awaiting_rating",
        "Waste_Prevented_Kg": "waste_prevented_kg",
    },
)

#: Every spec. :func:`load_order` sorts them topologically before use.
SCHEMA_MAP: tuple[TableSpec, ...] = (
    SELLERS, BUYERS, ITEMS, BATCHES, DAILY_ITEM_STATS, ORDERS,
)

#: Lookup by destination table name.
SPECS_BY_TABLE: dict[str, TableSpec] = {s.table: s for s in SCHEMA_MAP}

#: Lookup by pipeline frame name. A source may feed several tables
#: (``inventory`` feeds both ``items`` and ``batches``), so values are tuples.
SPECS_BY_SOURCE: dict[str, tuple[TableSpec, ...]] = {
    source: tuple(s for s in SCHEMA_MAP if s.source == source)
    for source in {s.source for s in SCHEMA_MAP}
}


# ══════════════════════════════════════════════════════════════════════════
# Projection
# ══════════════════════════════════════════════════════════════════════════
def project(frame: pd.DataFrame, spec: TableSpec) -> pd.DataFrame:
    """Translate a pipeline frame into its destination table's shape.

    Applies the derivation hook, renames to database columns, drops anything
    unmapped, normalises dates to ``YYYY-MM-DD`` strings and de-duplicates on
    the natural key.

    Raises:
        KeyError: If the source frame supplies none of the mapped columns. That
            means the frame is not what the spec expects, and writing it would
            reproduce the original zero-column failure — better to fail here,
            with a legible message.
    """
    if frame.empty:
        LOG.warning("project: source frame for '%s' is empty", spec.table)
        return pd.DataFrame(columns=spec.target_columns)

    working = spec.derive(frame) if spec.derive is not None else frame
    present = {src: dst for src, dst in spec.column_map.items()
               if src in working.columns}

    if not present:
        raise KeyError(
            f"Source frame for table '{spec.table}' supplies none of the "
            f"{len(spec.column_map)} mapped column(s). Expected columns such as "
            f"{sorted(spec.column_map)[:5]}; frame has "
            f"{sorted(working.columns)[:5]}."
        )

    missing = sorted(set(spec.column_map) - set(present))
    if missing:
        LOG.warning("Table '%s': %d mapped column(s) absent from the source and "
                    "left at their schema default: %s",
                    spec.table, len(missing), missing[:8])

    out = working[list(present)].rename(columns=present)

    for column in spec.date_columns:
        if column in out.columns:
            out[column] = pd.to_datetime(
                out[column], errors="coerce"
            ).dt.strftime("%Y-%m-%d")

    # Any remaining datetime column would be written as a pandas Timestamp,
    # which SQLite stores as an opaque string later comparisons will miss.
    for column in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[column]):
            out[column] = out[column].dt.strftime("%Y-%m-%d")

    if spec.key:
        key = [c for c in spec.key if c in out.columns]
        if key:
            before = len(out)
            out = out.drop_duplicates(subset=key, keep="last")
            if len(out) < before:
                LOG.info("Table '%s': dropped %d row(s) duplicated on %s",
                         spec.table, before - len(out), key)

    return out.reset_index(drop=True)


def to_pipeline_columns(spec: TableSpec) -> dict[str, str]:
    """Invert the map for the read direction (database -> pipeline vocabulary)."""
    return {dst: src for src, dst in spec.column_map.items()}


def database_to_pipeline(frame: pd.DataFrame, table: str) -> pd.DataFrame:
    """Rename a database frame into the analytical vocabulary."""
    spec = SPECS_BY_TABLE.get(table)
    if spec is None:
        return frame
    inverse = to_pipeline_columns(spec)
    return frame.rename(
        columns={c: inverse[c] for c in frame.columns if c in inverse}
    )


# ══════════════════════════════════════════════════════════════════════════
# Contract verification
# ══════════════════════════════════════════════════════════════════════════
def verify_against_schema(db: Any) -> dict[str, list[str]]:
    """Check every mapped target column exists in the live schema.

    This is the guard that turns the original failure into a test failure. A
    table named here but absent from ``schema.sql``, or a column renamed on one
    side only, is reported instead of surfacing as ``CREATE TABLE x ()``.

    Args:
        db: A :class:`~freshsense.db.session.Database`, or anything exposing
            ``table_columns`` and ``table_exists``.

    Returns:
        ``{table: [problems]}``. Empty means the contract holds.
    """
    problems: dict[str, list[str]] = {}

    for spec in SCHEMA_MAP:
        if not db.table_exists(spec.table):
            problems[spec.table] = [
                f"table '{spec.table}' is mapped but does not exist in schema.sql"
            ]
            continue

        issues: list[str] = []
        actual = set(db.table_columns(spec.table))

        unknown = sorted(set(spec.target_columns) - actual)
        if unknown:
            issues.append(f"column(s) not in schema: {unknown}")

        for column in spec.key:
            if column not in actual:
                issues.append(f"key column '{column}' not in schema")

        for local, (parent, _) in spec.foreign_keys.items():
            if local not in actual:
                issues.append(f"foreign key column '{local}' not in schema")
            if parent not in SPECS_BY_TABLE and not db.table_exists(parent):
                issues.append(f"foreign key parent '{parent}' does not exist")

        duplicated = {c for c in spec.target_columns
                      if spec.target_columns.count(c) > 1}
        if duplicated:
            issues.append(f"target column(s) mapped more than once: "
                          f"{sorted(duplicated)}")

        if issues:
            problems[spec.table] = issues

    return problems


def load_order() -> list[TableSpec]:
    """Specs sorted so every table's dependencies precede it.

    Topological rather than hand-ordered, so adding a spec cannot silently
    produce a foreign-key failure at load time.
    """
    resolved: list[TableSpec] = []
    placed: set[str] = set()
    remaining = list(SCHEMA_MAP)

    while remaining:
        progressed = False
        for spec in list(remaining):
            if all(dep in placed for dep in spec.depends_on):
                resolved.append(spec)
                placed.add(spec.table)
                remaining.remove(spec)
                progressed = True
        if not progressed:
            unresolved = ", ".join(s.table for s in remaining)
            raise ValueError(f"Circular table dependency among: {unresolved}")

    return resolved


def describe() -> pd.DataFrame:
    """Human-readable contract summary, for the architecture documentation."""
    return pd.DataFrame([
        {
            "table": spec.table,
            "source_frame": spec.source,
            "columns_mapped": len(spec.column_map),
            "natural_key": ", ".join(spec.key) or "—",
            "depends_on": ", ".join(spec.depends_on) or "—",
            "derived": spec.derive is not None,
            "description": spec.description,
        }
        for spec in load_order()
    ])


__all__ = [
    "TableSpec", "SCHEMA_MAP", "SPECS_BY_TABLE", "SPECS_BY_SOURCE",
    "SELLERS", "BUYERS", "ITEMS", "BATCHES", "DAILY_ITEM_STATS", "ORDERS",
    "project", "to_pipeline_columns", "database_to_pipeline",
    "verify_against_schema", "load_order", "describe",
]