"""Sprint 5 — the inventory agent.

Answers questions about live stock. A thin adapter: every figure it reports
comes from :class:`InventoryRepository` or the trained spoilage model, and it
computes nothing itself beyond formatting.

One genuine adaptation happens here. The database stores snake_case columns
(``days_to_expiry``), while the spoilage model was trained on the Sprint 1
pipeline's title-case frame (``Days_To_Expiry``). Feeding DB rows straight into
``build_feature_frame`` would silently produce a frame of zeros — every feature
missing, every prediction meaningless, and no error raised. :func:`to_model_frame`
bridges that gap explicitly, which is exactly the kind of boundary translation
an adapter exists to own.

Every action degrades to :meth:`empty` rather than raising when it finds
nothing, so the coordinator can replan — a zero-row search is a routine
outcome, not a fault.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from freshsense.agents.base import AgentResult, BaseAgent
from freshsense.agents.state import AgentState, AgentTask
from freshsense.config import SETTINGS
from freshsense.db.repository import InventoryRepository, PredictionRepository
from freshsense.logging_config import get_logger

LOG = get_logger(__name__)

#: Database column -> Sprint 1 pipeline column. Only the fields the spoilage
#: model consumes need translating.
DB_TO_MODEL_COLUMNS: dict[str, str] = {
    "age_ratio": "Age_Ratio",
    "stock_age_days": "Stock_Age_Days",
    "shelf_life_days": "Shelf_Life_Days",
    "days_to_expiry": "Days_To_Expiry",
    "temp_breach_hours": "Temp_Breach_Hours",
    "storage_temperature_c": "Storage_Temperature_C",
    "humidity_pct": "Humidity_Pct",
    "quantity_available": "Quantity_Available",
    "daily_avg_sales": "Daily_Avg_Sales",
    "perishability_score": "Perishability_Score",
    "storage_risk_score": "Storage_Risk_Score",
    "batch_id": "Inventory_ID",
}


def to_model_frame(batches: pd.DataFrame) -> pd.DataFrame:
    """Rename database columns to the names the trained model expects.

    Without this the model receives a frame of zeros and returns confident
    nonsense, because ``build_feature_frame`` fills absent columns with 0.0
    rather than failing.
    """
    present = {db: model for db, model in DB_TO_MODEL_COLUMNS.items()
               if db in batches.columns}
    return batches.rename(columns=present)


class InventoryAgent(BaseAgent):
    """Reads live stock and reports what needs attention."""

    name = "inventory"
    description = (
        "Queries live batch inventory, scores spoilage risk and surfaces stock "
        "that is expiring, at risk or running low."
    )
    supported_actions = (
        "find_at_risk", "search_inventory", "inventory_summary",
        "expiring_soon", "low_stock",
    )

    def __init__(
        self,
        repository: InventoryRepository | None = None,
        predictions: PredictionRepository | None = None,
    ) -> None:
        super().__init__()
        self._repository = repository
        self._predictions = predictions
        self._spoilage: Any = None
        self._spoilage_loaded = False
        self.currency = SETTINGS.currency

    # ── lazily-resolved dependencies ──────────────────────────────────
    @property
    def repository(self) -> InventoryRepository:
        if self._repository is None:
            self._repository = InventoryRepository()
        return self._repository

    @property
    def predictions(self) -> PredictionRepository:
        if self._predictions is None:
            self._predictions = PredictionRepository()
        return self._predictions

    @property
    def spoilage(self) -> Any:
        """The trained spoilage model, or ``None`` when no artefact exists.

        Loaded on first use rather than at construction: the coordinator builds
        every agent to populate its roster, and an unused agent should not pull
        a model file off disk.
        """
        if not self._spoilage_loaded:
            self._spoilage_loaded = True
            try:
                from freshsense.models.spoilage import get_spoilage_model

                self._spoilage = get_spoilage_model()
            except Exception as exc:                     # pragma: no cover
                LOG.warning("Spoilage model unavailable: %s", exc)
                self._spoilage = None
        return self._spoilage

    # ── risk scoring ──────────────────────────────────────────────────
    def _score_risk(self, batches: pd.DataFrame) -> pd.DataFrame:
        """Attach modelled spoilage risk, falling back to the stored index.

        ``environment_stress_index`` is the transparent heuristic computed in
        Sprint 1. It is a genuine fallback rather than a placeholder: the
        learned model was benchmarked against it, so an answer built on it is
        still defensible when the artefact is missing.
        """
        out = batches.copy()
        model = self.spoilage

        if model is not None and not out.empty:
            try:
                out["risk_score"] = model.predict_proba(to_model_frame(out)).round(4)
                out["risk_source"] = "model"
                return out
            except Exception as exc:
                LOG.warning("Risk scoring failed (%s); using the stress index", exc)

        out["risk_score"] = pd.to_numeric(
            out.get("environment_stress_index", 0.3), errors="coerce"
        ).fillna(0.3).round(4)
        out["risk_source"] = "stress_index"
        return out

    def _log_predictions(self, batches: pd.DataFrame) -> None:
        """Persist risk scores so Sprint 6 can compare them against outcomes."""
        if batches.empty or "risk_score" not in batches.columns:
            return
        if str(batches["risk_source"].iloc[0]) != "model":
            return
        try:
            from freshsense.models.spoilage import MODEL_NAME
            from freshsense.data.features import risk_label

            self.predictions.record_many([
                {
                    "entity_type": "batch",
                    "entity_id": str(row["batch_id"]),
                    "prediction_type": "spoilage_risk",
                    "value": float(row["risk_score"]),
                    "label": risk_label(float(row["risk_score"])),
                    "confidence": 0.9,
                    "model_name": MODEL_NAME,
                }
                for _, row in batches.head(50).iterrows()
            ])
        except Exception as exc:                         # pragma: no cover
            LOG.debug("Could not persist spoilage predictions: %s", exc)

    # ── formatting helpers ────────────────────────────────────────────
    def _money(self, value: float) -> str:
        return f"{self.currency}{float(value):,.0f}"

    @staticmethod
    def _name_list(batches: pd.DataFrame, limit: int = 3) -> str:
        """Top products by capital at risk, as readable prose."""
        if batches.empty or "product_name" not in batches.columns:
            return ""
        ranked = (
            batches.groupby("product_name")["capital_at_risk"].sum()
            .sort_values(ascending=False).head(limit)
        )
        names = list(ranked.index)
        if len(names) == 1:
            return names[0]
        return ", ".join(names[:-1]) + f" and {names[-1]}"

    @staticmethod
    def _ids(batches: pd.DataFrame, limit: int = 25) -> list[str]:
        if batches.empty or "batch_id" not in batches.columns:
            return []
        return batches["batch_id"].astype(str).head(limit).tolist()

    # ── actions ───────────────────────────────────────────────────────
    def action_find_at_risk(self, state: AgentState, task: AgentTask) -> AgentResult:
        """Batches expiring inside the window, ranked by action priority."""
        days = int(task.param("days", SETTINGS.features.get("at_risk_days", 2)))
        zone = task.param("zone")
        product = task.param("product")
        limit = int(task.param("limit", 200))

        batches = self.repository.at_risk(days=days, limit=limit)
        if not batches.empty and zone:
            batches = batches[batches["zone"] == zone]
        if not batches.empty and product:
            batches = batches[
                batches["product_name"].astype(str).str.lower() == str(product).lower()
            ]

        scope = (f" in {zone}" if zone else "") + (f" for {product}" if product else "")
        if batches.empty:
            return self.empty(
                f"No stock is expiring within {days} day(s){scope}.",
                tool_called="InventoryRepository.at_risk",
            )

        batches = self._score_risk(batches)
        self._log_predictions(batches)

        # A full page means there may be more beyond it. Reporting the page
        # size as if it were the total is the kind of quiet understatement that
        # makes an operator under-react.
        truncated = len(batches) >= limit
        count_phrase = f"At least {len(batches)}" if truncated else f"{len(batches)}"

        capital = float(batches["capital_at_risk"].sum())
        quantity = float(batches["quantity_available"].sum())
        critical = int((batches["risk_score"] >= 0.7).sum())
        headline = self._name_list(batches)

        narrative = (
            f"{count_phrase} batch(es){scope} expire within {days} day(s), "
            f"holding {quantity:,.0f} units and {self._money(capital)} of capital. "
            f"{critical} carry critical spoilage risk. "
            f"Largest exposure: {headline}."
        )
        if truncated:
            narrative += (
                f" This is the top {limit} by action priority; ask for a larger "
                f"page to see the rest."
            )

        return self.ok(
            narrative,
            facts={
                "at_risk_batch_ids": self._ids(batches),
                "at_risk_count": len(batches),
                "at_risk_truncated": truncated,
                "capital_at_risk": round(capital, 2),
                "critical_count": critical,
                "at_risk_products": sorted(batches["product_name"].unique().tolist())[:10],
                "risk_source": str(batches["risk_source"].iloc[0]),
            },
            artifacts={"at_risk_batches": batches},
            tool_called="InventoryRepository.at_risk",
        )

    def action_search_inventory(self, state: AgentState, task: AgentTask) -> AgentResult:
        """Filtered batch search by product, zone, category or grade."""
        product = task.param("product")
        zone = task.param("zone")
        category = task.param("category")
        grade = task.param("grade")
        min_quantity = task.param("min_quantity")

        batches = self.repository.search(
            product=product, zone=zone, category=category, grade=grade,
            min_quantity=float(min_quantity) if min_quantity else None,
            active_only=True,
        )

        criteria = ", ".join(
            f"{label} {value}" for label, value in (
                ("product", product), ("zone", zone),
                ("category", category), ("grade", grade),
            ) if value
        ) or "active stock"

        if batches.empty:
            return self.empty(
                f"No active stock matches {criteria}.",
                tool_called="InventoryRepository.search",
            )

        batches = self._score_risk(batches)
        quantity = float(batches["quantity_available"].sum())
        value = float(batches["inventory_value"].sum())
        sellers = int(batches["seller_id"].nunique())
        soonest = int(batches["days_to_expiry"].min())
        unit = str(batches["unit"].iloc[0]) if "unit" in batches.columns else "units"

        narrative = (
            f"Found {len(batches)} batch(es) matching {criteria}: "
            f"{quantity:,.0f} {unit} across {sellers} seller(s), "
            f"worth {self._money(value)}. "
            f"The earliest expires in {soonest} day(s)."
        )

        return self.ok(
            narrative,
            facts={
                "matched_batch_ids": self._ids(batches),
                "matched_count": len(batches),
                "matched_quantity": round(quantity, 1),
                "matched_value": round(value, 2),
                "soonest_expiry_days": soonest,
                "product": product,
                "zone": zone,
            },
            artifacts={"matched_batches": batches},
            tool_called="InventoryRepository.search",
        )

    def action_inventory_summary(self, state: AgentState, task: AgentTask) -> AgentResult:
        """Portfolio-level position: value, exposure and concentration."""
        zone = task.param("zone")
        kpis = self.repository.kpi_summary()

        if not kpis or not kpis.get("total_batches"):
            return self.empty(
                "The inventory is empty — no active batches are recorded.",
                tool_called="InventoryRepository.kpi_summary",
            )

        by_category = self.repository.by_category()
        by_zone = self.repository.by_zone()

        # kpi_summary filters to status='active', so expired stock is invisible
        # to it. Query that separately rather than reporting a confident zero.
        expired_batches = self.repository.expired(limit=500)
        expired = len(expired_batches)
        expired_capital = (
            float(expired_batches["capital_at_risk"].sum()) if expired else 0.0
        )

        if zone is not None and not by_zone.empty:
            row = by_zone[by_zone["zone"] == zone]
            if not row.empty:
                kpis["zone_value"] = float(row["value"].iloc[0])
                kpis["zone_batches"] = int(row["batches"].iloc[0])

        top_category = (
            str(by_category["category"].iloc[0]) if not by_category.empty else "—"
        )
        at_risk = int(kpis.get("at_risk_count", 0))

        narrative = (
            f"{int(kpis['total_batches'])} active batch(es) across "
            f"{int(kpis.get('products', 0))} product(s) and "
            f"{int(kpis.get('sellers', 0))} seller(s), valued at "
            f"{self._money(kpis.get('inventory_value', 0))}. "
            f"{at_risk} batch(es) expire within 2 days, putting "
            f"{self._money(kpis.get('capital_at_risk', 0))} of capital at stake. "
            f"A further {expired} batch(es) are already past expiry, holding "
            f"{self._money(expired_capital)} to write off or donate. "
            f"Largest category by value: {top_category}."
        )
        if zone and "zone_value" in kpis:
            narrative += (
                f" In {zone}: {int(kpis['zone_batches'])} batch(es) worth "
                f"{self._money(kpis['zone_value'])}."
            )

        return self.ok(
            narrative,
            facts={
                "total_batches": int(kpis["total_batches"]),
                "inventory_value": round(float(kpis.get("inventory_value", 0)), 2),
                "capital_at_risk": round(float(kpis.get("capital_at_risk", 0)), 2),
                "at_risk_count": at_risk,
                "expired_count": expired,
                "expired_capital": round(expired_capital, 2),
                "top_category": top_category,
            },
            artifacts={
                "kpi_summary": kpis,
                "by_category": by_category,
                "by_zone": by_zone,
                "expired_batches": expired_batches,
            },
            tool_called="InventoryRepository.kpi_summary",
        )

    def action_expiring_soon(self, state: AgentState, task: AgentTask) -> AgentResult:
        """Expiry timeline, bucketed by day, plus stock already past expiry."""
        days = int(task.param("days", 3))
        batches = self.repository.search(max_days_to_expiry=days, active_only=True)
        expired = self.repository.expired(limit=100)

        if batches.empty and expired.empty:
            return self.empty(
                f"Nothing expires within {days} day(s) and no stock is past expiry.",
                tool_called="InventoryRepository.search",
            )

        timeline = pd.DataFrame()
        if not batches.empty:
            timeline = (
                batches.groupby("days_to_expiry")
                .agg(batches=("batch_id", "count"),
                     quantity=("quantity_available", "sum"),
                     capital=("capital_at_risk", "sum"))
                .reset_index()
                .sort_values("days_to_expiry")
            )

        parts: list[str] = []
        if not timeline.empty:
            buckets = "; ".join(
                f"{int(r['days_to_expiry'])}d: {int(r['batches'])} batch(es), "
                f"{self._money(r['capital'])}"
                for _, r in timeline.iterrows()
            )
            parts.append(f"Expiry timeline over the next {days} day(s) — {buckets}.")
        if not expired.empty:
            write_off = float(expired["capital_at_risk"].sum())
            parts.append(
                f"{len(expired)} batch(es) are already past expiry, representing "
                f"{self._money(write_off)} to write off or donate."
            )

        return self.ok(
            " ".join(parts),
            facts={
                "expiring_count": int(len(batches)),
                "expired_count": int(len(expired)),
                "expired_capital": round(float(expired["capital_at_risk"].sum()), 2)
                if not expired.empty else 0.0,
            },
            artifacts={"expiry_timeline": timeline, "expired_batches": expired},
            tool_called="InventoryRepository.search",
        )

    def action_low_stock(self, state: AgentState, task: AgentTask) -> AgentResult:
        """Batches at or below the low-stock threshold."""
        threshold = float(task.param(
            "threshold", SETTINGS.features.get("low_stock_threshold", 5.0)
        ))
        batches = self.repository.low_stock(threshold=threshold, limit=50)

        if batches.empty:
            return self.empty(
                f"No batch is below the {threshold:.0f}-unit low-stock threshold.",
                tool_called="InventoryRepository.low_stock",
            )

        products = sorted(batches["product_name"].unique().tolist())
        narrative = (
            f"{len(batches)} batch(es) are at or below {threshold:.0f} units, "
            f"covering {len(products)} product(s): "
            f"{', '.join(products[:5])}"
            f"{' and others' if len(products) > 5 else ''}. "
            f"These will sell out shortly — reorder or mark them sold out to "
            f"avoid failed pickups."
        )

        return self.ok(
            narrative,
            facts={
                "low_stock_count": len(batches),
                "low_stock_products": products[:10],
                "low_stock_batch_ids": self._ids(batches),
            },
            artifacts={"low_stock_batches": batches},
            tool_called="InventoryRepository.low_stock",
        )


__all__ = ["InventoryAgent", "to_model_frame", "DB_TO_MODEL_COLUMNS"]