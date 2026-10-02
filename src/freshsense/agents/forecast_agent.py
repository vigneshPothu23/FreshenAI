"""Sprint 5 — the forecast agent.

Serves demand projections from the Sprint 2 :class:`ForecastService`. It fits
nothing and predicts nothing itself; it assembles the series, calls the trained
models and reports the result together with the error the model actually
achieved in backtesting.

That last part is the point. A forecast quoted without its error band is not
decision-grade — an operator ordering against "we expect 120 kg" needs to know
whether that model scores 8% MAPE or 40%, and whether it even beats repeating
last week. Every narrative here carries the accuracy and the baseline
comparison, and flags explicitly when a product's model is too weak to order
against.

When no trained artefact exists the agent fails cleanly rather than
improvising. That failure is what triggers the planner's recovery path onto
historical averages, so silently substituting a worse number here would hide a
degradation the user is entitled to see.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pandas as pd

from freshsense.agents.base import AgentResult, BaseAgent
from freshsense.agents.state import AgentState, AgentTask
from freshsense.config import SETTINGS
from freshsense.db.repository import (InventoryRepository, PredictionRepository,
                                      SalesRepository)
from freshsense.logging_config import get_logger

LOG = get_logger(__name__)

#: Daily-stats table -> Sprint 1 pipeline column vocabulary.
DB_TO_PIPELINE_SALES: dict[str, str] = {
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

#: Above this backtest MAPE the model is reported as unreliable rather than
#: quoted as if it were dependable. Ordering stock against a forecast that is
#: routinely 35% out costs more than ordering on judgement.
UNRELIABLE_MAPE = 35.0


class ForecastAgent(BaseAgent):
    """Projects demand and compares it against stock on hand."""

    name = "forecast"
    description = (
        "Projects product demand from the trained forecasting models, reports "
        "accuracy against a seasonal-naive baseline and compares forward demand "
        "with current stock cover."
    )
    supported_actions = ("forecast_demand", "forecast_summary")

    def __init__(
        self,
        *,
        sales: SalesRepository | None = None,
        inventory: InventoryRepository | None = None,
        predictions: PredictionRepository | None = None,
    ) -> None:
        super().__init__()
        self._sales = sales
        self._inventory = inventory
        self._predictions = predictions
        self._service: Any = None
        self._service_loaded = False
        self.currency = SETTINGS.currency

    # ── lazily-resolved dependencies ──────────────────────────────────
    @property
    def sales(self) -> SalesRepository:
        if self._sales is None:
            self._sales = SalesRepository()
        return self._sales

    @property
    def inventory(self) -> InventoryRepository:
        if self._inventory is None:
            self._inventory = InventoryRepository()
        return self._inventory

    @property
    def predictions(self) -> PredictionRepository:
        if self._predictions is None:
            self._predictions = PredictionRepository()
        return self._predictions

    @property
    def service(self) -> Any:
        """The trained forecast bundle, or ``None`` when no artefact exists.

        Deferred: the bundle holds one fitted model per product and is read off
        disk, which an agent should not do merely to be listed on a roster.
        """
        if not self._service_loaded:
            self._service_loaded = True
            try:
                from freshsense.models.forecasting import get_forecast_service

                service = get_forecast_service()
                self._service = service if service.is_ready else None
                if self._service is None:
                    LOG.warning("Forecast service loaded but holds no trained models")
            except Exception as exc:                     # pragma: no cover
                LOG.warning("Forecast service unavailable: %s", exc)
                self._service = None
        return self._service

    @property
    def is_ready(self) -> bool:
        return self.service is not None

    # ── input assembly ────────────────────────────────────────────────
    def _series(self, product: str | None = None) -> pd.DataFrame:
        """Daily demand history in the vocabulary the models were trained on."""
        frame = self.sales.series(product=product)
        if frame.empty:
            return frame
        present = {db: pipe for db, pipe in DB_TO_PIPELINE_SALES.items()
                   if db in frame.columns}
        renamed = frame.rename(columns=present)
        renamed["Date"] = pd.to_datetime(renamed["Date"])
        return renamed

    @staticmethod
    def _no_models_message() -> str:
        return (
            "No trained forecasting model is available. Run "
            "`python scripts/train_models.py` to fit them, or proceed on "
            "historical daily averages."
        )

    def _accuracy_phrase(self, metrics: dict[str, Any]) -> tuple[str, bool]:
        """Render backtest accuracy, and say whether it is trustworthy.

        Returns ``(phrase, reliable)``. A model with no recorded metrics is
        treated as unproven rather than assumed good.
        """
        mape = metrics.get("model_mape")
        baseline = metrics.get("baseline_mape")

        if mape is None:
            return ("This product has no recorded backtest, so its accuracy is "
                    "unproven."), False

        reliable = float(mape) <= UNRELIABLE_MAPE
        phrase = f"Backtest error {float(mape):.1f}% MAPE"
        if baseline is not None:
            beats = float(mape) < float(baseline)
            phrase += (
                f" against a seasonal-naive baseline of {float(baseline):.1f}% — "
                f"{'better than' if beats else 'no better than'} repeating last week"
            )
        phrase += "."
        if not reliable:
            phrase += (
                f" That is above the {UNRELIABLE_MAPE:.0f}% reliability threshold, "
                f"so treat this as indicative only and do not order against it."
            )
        return phrase, reliable

    def _log_forecast(
        self, product: str, future: pd.DataFrame, mape: float | None
    ) -> None:
        """Persist projections so Sprint 6 can score them against actuals."""
        if future.empty:
            return
        try:
            from freshsense.models.forecasting import MODEL_NAME

            # Confidence falls as backtest error rises; an unproven model is
            # recorded at low confidence rather than none, so it still appears
            # in monitoring.
            confidence = (
                max(0.1, 1.0 - float(mape) / 100.0) if mape is not None else 0.4
            )
            self.predictions.record_many([
                {
                    "entity_type": "product",
                    "entity_id": product,
                    "prediction_type": "demand",
                    "value": float(row["predicted"]),
                    "label": f"{row['day_of_week']}",
                    "confidence": round(confidence, 3),
                    "model_name": MODEL_NAME,
                    "horizon_date": pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
                }
                for _, row in future.iterrows()
            ])
        except Exception as exc:                         # pragma: no cover
            LOG.debug("Could not persist demand predictions: %s", exc)

    # ── actions ───────────────────────────────────────────────────────
    def action_forecast_demand(
        self, state: AgentState, task: AgentTask
    ) -> AgentResult:
        """Project demand for one product, with interval and accuracy."""
        if not self.is_ready:
            return self.fail(self._no_models_message(),
                             tool_called="ForecastService.load")

        product = task.param("product") or state.fact("product")
        horizon = int(task.param("horizon", SETTINGS.forecasting.get("horizon_days", 14)))

        series = self._series()
        if series.empty:
            return self.empty(
                "No sales history is loaded, so demand cannot be projected.",
                tool_called="SalesRepository.series",
            )

        # No product named: report the movers that matter most by revenue
        # rather than refusing a reasonable question.
        if not product:
            return self._forecast_top_movers(series, horizon)

        available = set(self.service.products)
        if product not in available:
            return self.empty(
                f"No trained model exists for '{product}'. Models are available "
                f"for {len(available)} product(s).",
                tool_called="ForecastService.forecast",
            )

        try:
            result = self.service.forecast(product, series, horizon=horizon)
        except ValueError as exc:
            return self.empty(f"Cannot forecast '{product}': {exc}",
                              tool_called="ForecastService.forecast")

        future = result.future
        total = float(future["predicted"].sum())
        daily = float(future["predicted"].mean())
        accuracy, reliable = self._accuracy_phrase(result.metrics)
        self._log_forecast(product, future, result.metrics.get("model_mape"))

        peak = future.loc[future["predicted"].idxmax()]
        unit = str(series[series["Product_Name"] == product]["Unit"].iloc[0]) \
            if "Unit" in series.columns and not series[series["Product_Name"] == product].empty \
            else "units"

        narrative = (
            f"{product}: {total:,.0f} {unit} expected over the next {horizon} "
            f"day(s), averaging {daily:.0f} {unit}/day. "
            f"Peak on {peak['day_of_week']} "
            f"({pd.Timestamp(peak['date']).strftime('%d %b')}) at "
            f"{float(peak['predicted']):.0f} {unit}. "
            f"Next two days: {result.next_2_days:.0f} {unit} "
            f"(range {float(future['lower'].head(2).sum()):.0f}–"
            f"{float(future['upper'].head(2).sum()):.0f}). {accuracy}"
        )

        return self.ok(
            narrative,
            facts={
                "forecast_product": product,
                "forecast_total": round(total, 1),
                "forecast_daily_mean": round(daily, 1),
                "forecast_next_2_days": round(result.next_2_days, 1),
                "forecast_horizon": horizon,
                "forecast_mape": result.metrics.get("model_mape"),
                "forecast_reliable": reliable,
                "beats_baseline": result.beats_baseline,
            },
            artifacts={
                "forecast_future": future,
                "forecast_history": result.history,
                "forecast_components": result.components,
                "forecast_metrics": result.metrics,
            },
            tool_called="ForecastService.forecast",
        )

    def _forecast_top_movers(
        self, series: pd.DataFrame, horizon: int
    ) -> AgentResult:
        """Fallback when no product was named: project the biggest movers."""
        top = self.sales.top_products(limit=5)
        if top.empty:
            return self.empty("No product has enough sales history to forecast.",
                              tool_called="SalesRepository.top_products")

        rows: list[dict[str, Any]] = []
        for product in top["product_name"]:
            if product not in set(self.service.products):
                continue
            try:
                result = self.service.forecast(product, series, horizon=horizon)
            except ValueError:
                continue
            rows.append({
                "product_name": product,
                "forecast_total": round(float(result.future["predicted"].sum()), 1),
                "daily_mean": round(float(result.future["predicted"].mean()), 1),
                "mape": result.metrics.get("model_mape"),
                "beats_baseline": result.beats_baseline,
            })

        if not rows:
            return self.empty("None of the top products has a trained model.",
                              tool_called="ForecastService.forecast")

        frame = pd.DataFrame(rows).sort_values("forecast_total", ascending=False)
        leader = frame.iloc[0]
        narrative = (
            f"No product was specified, so here are the {len(frame)} biggest "
            f"movers by revenue over the next {horizon} day(s). "
            f"Largest expected demand: {leader['product_name']} at "
            f"{leader['forecast_total']:,.0f} units "
            f"({leader['daily_mean']:.0f}/day). "
            f"{int(frame['beats_baseline'].sum())} of {len(frame)} models beat "
            f"the seasonal-naive baseline."
        )

        return self.ok(
            narrative,
            facts={
                "forecast_products": frame["product_name"].tolist(),
                "forecast_horizon": horizon,
                "forecast_scope": "top_movers",
            },
            artifacts={"forecast_top_movers": frame},
            tool_called="ForecastService.forecast",
        )

    def action_forecast_summary(
        self, state: AgentState, task: AgentTask
    ) -> AgentResult:
        """Forward demand across the catalogue, set against stock on hand.

        This is the action that makes forecasting operational rather than
        decorative: demand alone is a number, but demand compared with cover
        tells the operator what will run out and what will be left over.
        """
        if not self.is_ready:
            return self.fail(self._no_models_message(),
                             tool_called="ForecastService.load")

        horizon = int(task.param("horizon", 2))
        series = self._series()
        if series.empty:
            return self.empty(
                "No sales history is loaded, so demand cannot be projected.",
                tool_called="SalesRepository.series",
            )

        forecasts = self.service.forecast_all(series, horizon=horizon)
        if forecasts.empty:
            return self.empty("No product could be forecast.",
                              tool_called="ForecastService.forecast_all")

        batches = self.inventory.all_batches(active_only=True)
        merged = forecasts.copy()

        if not batches.empty:
            on_hand = (
                batches.groupby("product_name")
                .agg(quantity_on_hand=("quantity_available", "sum"),
                     batches=("batch_id", "count"))
                .reset_index()
            )
            merged = merged.merge(on_hand, on="product_name", how="left")
            merged["quantity_on_hand"] = merged["quantity_on_hand"].fillna(0.0)
            merged["surplus"] = (
                merged["quantity_on_hand"] - merged["forecast_units"]
            ).round(1)
            merged["days_of_cover"] = (
                merged["quantity_on_hand"]
                / merged["daily_mean"].clip(lower=0.1)
            ).round(1)
        else:
            merged["quantity_on_hand"] = 0.0
            merged["surplus"] = -merged["forecast_units"]
            merged["days_of_cover"] = 0.0

        total_demand = float(merged["forecast_units"].sum())
        shortfalls = merged[merged["surplus"] < 0].sort_values("surplus")
        surpluses = merged[merged["surplus"] > 0].sort_values(
            "surplus", ascending=False
        )
        beating = int(merged["beats_baseline"].sum())
        mean_mape = float(merged["mape"].mean(skipna=True))

        parts = [
            f"Across {len(merged)} product(s), {total_demand:,.0f} units are "
            f"expected over the next {horizon} day(s)."
        ]
        if not shortfalls.empty:
            names = ", ".join(shortfalls["product_name"].head(3))
            parts.append(
                f"{len(shortfalls)} product(s) will run short — worst: {names}."
            )
        if not surpluses.empty:
            leader = surpluses.iloc[0]
            parts.append(
                f"{len(surpluses)} product(s) are oversupplied — largest surplus: "
                f"{leader['product_name']} with {float(leader['surplus']):,.0f} "
                f"units beyond expected demand."
            )
        parts.append(
            f"Model quality: {beating} of {len(merged)} beat the seasonal-naive "
            f"baseline, mean backtest error {mean_mape:.1f}% MAPE."
        )

        return self.ok(
            " ".join(parts),
            facts={
                "forecast_total_demand": round(total_demand, 1),
                "forecast_horizon": horizon,
                "shortfall_products": shortfalls["product_name"].head(10).tolist(),
                "surplus_products": surpluses["product_name"].head(10).tolist(),
                "models_beating_baseline": beating,
                "mean_forecast_mape": round(mean_mape, 2),
            },
            artifacts={"forecast_vs_stock": merged},
            tool_called="ForecastService.forecast_all",
        )


__all__ = ["ForecastAgent", "DB_TO_PIPELINE_SALES", "UNRELIABLE_MAPE"]