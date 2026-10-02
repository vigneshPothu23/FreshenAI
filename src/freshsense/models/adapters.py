"""Column-vocabulary translation between the database and the models.

The repository layer returns the normalised schema's ``snake_case`` columns.
The models were fitted on the Sprint 1 pipeline's ``Title_Case`` analytical
frame. Neither is wrong, but nothing translated between them, so the models
could not be fed from the database at all:

* ``DemandForecaster.fit`` raised ``KeyError: 'Date'``.
* ``build_feature_frame`` raised ``AttributeError`` on the first absent column —
  and had it not raised, it would have filled every feature with 0.0 and
  returned confident nonsense, which is the worse failure.

This module is the one place that translation happens. It reuses the Milestone 1
mapping when that module is importable, so a column renamed in ``schema.sql``
propagates from a single definition; the explicit maps below are the fallback
and the documentation of what the models require.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from freshsense.logging_config import get_logger

LOG = get_logger(__name__)

#: ``batches`` -> the inventory column names the spoilage model was fitted on.
BATCH_TO_MODEL: dict[str, str] = {
    "batch_id": "Inventory_ID",
    "product_name": "Product_Name",
    "category": "Category",
    "seller_id": "Seller_ID",
    "zone": "Zone",
    "unit": "Unit",
    "quantity_available": "Quantity_Available",
    "expiry_date": "Expiry_Date",
    "shelf_life_days": "Shelf_Life_Days",
    "stock_age_days": "Stock_Age_Days",
    "days_to_expiry": "Days_To_Expiry",
    "age_ratio": "Age_Ratio",
    "storage_type": "Storage_Type",
    "storage_temperature_c": "Storage_Temperature_C",
    "humidity_pct": "Humidity_Pct",
    "temp_breach_hours": "Temp_Breach_Hours",
    "perishability_level": "Perishability_Level",
    "perishability_score": "Perishability_Score",
    "storage_risk_score": "Storage_Risk_Score",
    "daily_avg_sales": "Daily_Avg_Sales",
    "cost_price": "Cost_Price",
    "mrp": "Selling_Price",
    "effective_price": "Effective_Price",
    "quality_grade": "Quality_Grade",
    "status": "Status",
    "is_spoiled": "Is_Spoiled",
    "environment_stress_index": "Environment_Stress_Index",
}

#: ``daily_item_stats`` -> the series column names the forecaster was fitted on.
STATS_TO_MODEL: dict[str, str] = {
    "stat_date": "Date",
    "product_name": "Product_Name",
    "category": "Category",
    "unit": "Unit",
    "units_sold": "Units_Sold",
    "avg_selling_price": "Avg_Selling_Price",
    "revenue": "Revenue",
    "ambient_temp_c": "Ambient_Temp_C",
    "humidity_pct": "Humidity_Pct",
    "is_weekend": "Is_Weekend",
    "is_festival": "Is_Festival",
    "day_of_week": "Day_Of_Week",
}

#: Columns the spoilage model reads. Guaranteeing they exist is what stops
#: ``build_feature_frame`` from silently substituting zeros.
REQUIRED_BATCH_COLUMNS: tuple[str, ...] = (
    "Age_Ratio", "Stock_Age_Days", "Shelf_Life_Days", "Days_To_Expiry",
    "Temp_Breach_Hours", "Storage_Temperature_C", "Humidity_Pct",
    "Quantity_Available", "Daily_Avg_Sales", "Perishability_Score",
    "Storage_Risk_Score",
)

#: Columns the forecaster reads. ``Ambient_Temp_C`` and ``Is_Festival`` are
#: optional to the model, but ``Date`` and ``Units_Sold`` are not.
REQUIRED_SERIES_COLUMNS: tuple[str, ...] = ("Date", "Units_Sold", "Product_Name")


def _mapping_from_milestone_one(table: str) -> dict[str, str] | None:
    """Reuse the Milestone 1 schema contract when it is importable.

    Preferred over the literals above so a column renamed in ``schema.sql``
    propagates from one definition rather than two that can drift apart.
    """
    try:
        from freshsense.db.mapping import SPECS_BY_TABLE, to_pipeline_columns
    except Exception:                                    # pragma: no cover
        return None
    spec = SPECS_BY_TABLE.get(table)
    return to_pipeline_columns(spec) if spec is not None else None


def to_model_frame(frame: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Rename the columns present, leaving anything unmapped untouched."""
    if frame.empty:
        return frame.copy()
    present = {db: model for db, model in mapping.items() if db in frame.columns}
    return frame.rename(columns=present)


def batches_to_inventory_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Translate a ``batches`` frame into the spoilage model's vocabulary.

    Every required feature column is guaranteed present and numeric. A missing
    one is filled with a neutral value *and logged*, because the alternative —
    the model's own scalar default — produces a frame of zeros with no signal
    that anything went wrong.
    """
    mapping = _mapping_from_milestone_one("batches") or BATCH_TO_MODEL
    out = to_model_frame(frame, mapping)
    if out.empty:
        return out

    missing = [c for c in REQUIRED_BATCH_COLUMNS if c not in out.columns]
    if missing:
        LOG.warning(
            "batches frame is missing %d model feature(s): %s. Filling with 0.0 — "
            "predictions on those features carry no information.",
            len(missing), missing,
        )
    for column in REQUIRED_BATCH_COLUMNS:
        if column not in out.columns:
            out[column] = 0.0
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)

    if "Is_Spoiled" in out.columns:
        out["Is_Spoiled"] = (
            pd.to_numeric(out["Is_Spoiled"], errors="coerce").fillna(0).astype(int)
        )
    return out


def stats_to_series_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Translate a ``daily_item_stats`` frame into the forecaster's vocabulary.

    ``Date`` is coerced to datetime and rows without one are dropped: the model
    fits a trend on elapsed days, and an unparseable date would silently become
    ``NaT`` and poison the whole design matrix.
    """
    mapping = _mapping_from_milestone_one("daily_item_stats") or STATS_TO_MODEL
    out = to_model_frame(frame, mapping)
    if out.empty:
        return out

    missing = [c for c in REQUIRED_SERIES_COLUMNS if c not in out.columns]
    if missing:
        raise KeyError(
            f"Series frame is missing required column(s) {missing}. The "
            f"forecaster fits on Date and Units_Sold; a frame without them "
            f"cannot be trained on. Columns present: {sorted(out.columns)[:8]}."
        )

    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    dropped = int(out["Date"].isna().sum())
    if dropped:
        LOG.warning("Dropped %d row(s) with an unparseable date", dropped)
        out = out[out["Date"].notna()]

    out["Units_Sold"] = pd.to_numeric(out["Units_Sold"], errors="coerce").fillna(0.0)
    for column, default in (("Ambient_Temp_C", 30.5), ("Is_Festival", 0)):
        if column not in out.columns:
            out[column] = default
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(default)

    return out.sort_values(["Product_Name", "Date"]).reset_index(drop=True)


def describe_mapping() -> dict[str, Any]:
    """Which mapping source is in use, for start-up diagnostics."""
    return {
        "batches": "milestone-1 mapping"
        if _mapping_from_milestone_one("batches") else "adapters.BATCH_TO_MODEL",
        "daily_item_stats": "milestone-1 mapping"
        if _mapping_from_milestone_one("daily_item_stats") else "adapters.STATS_TO_MODEL",
        "required_batch_columns": len(REQUIRED_BATCH_COLUMNS),
        "required_series_columns": len(REQUIRED_SERIES_COLUMNS),
    }


__all__ = [
    "BATCH_TO_MODEL", "STATS_TO_MODEL", "REQUIRED_BATCH_COLUMNS",
    "REQUIRED_SERIES_COLUMNS", "to_model_frame", "batches_to_inventory_frame",
    "stats_to_series_frame", "describe_mapping",
]