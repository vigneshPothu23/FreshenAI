"""Model inference: load artefact -> predict -> persist -> return to the app.

The read half of the ML layer. It never fits anything: a model is loaded once
from the registry and reused, so a user requesting a forecast does not trigger
a retrain. When no artefact exists the service reports that plainly rather than
training on demand, because a silent retrain inside a page load is both slow and
unrepeatable.

Predictions are written through the existing ``PredictionRepository`` into
``predictions``, which already carries ``entity_type``, ``prediction_type``,
``value``, ``confidence`` and ``horizon_date``. Nothing new is added to the
schema — and because ``prediction_outcomes`` already exists, every prediction
written here is scoreable later without further work.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from freshsense.config import SETTINGS
from freshsense.db.database import Database, get_database
from freshsense.db.repository import (InventoryRepository, PredictionRepository,
                                      SalesRepository)
from freshsense.logging_config import get_logger
from freshsense.models.adapters import (batches_to_inventory_frame,
                                        stats_to_series_frame)
from freshsense.models.demand_forecasting import (ForecastResult,
                                                  ForecastService)
from freshsense.models.demand_forecasting import MODEL_NAME as FORECAST_MODEL
from freshsense.models.spoilage_prediction import MODEL_NAME as SPOILAGE_MODEL
from freshsense.models.spoilage_prediction import RiskAssessment, SpoilageModel

LOG = get_logger(__name__)

#: Prediction-type vocabulary written into ``predictions.prediction_type``.
DEMAND, SPOILAGE_RISK = "demand", "spoilage_risk"


@dataclass
class PredictionBatch:
    """A set of predictions, with what was persisted and what was not."""

    prediction_type: str
    model_name: str
    rows: int = 0
    persisted: int = 0
    latency_ms: int = 0
    available: bool = True
    reason: str = ""
    payload: Any = None
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.available and self.rows > 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "prediction_type": self.prediction_type, "model": self.model_name,
            "rows": self.rows, "persisted": self.persisted,
            "latency_ms": self.latency_ms, "available": self.available,
            "reason": self.reason,
        }


class InferenceService:
    """Serves predictions from persisted artefacts and records them."""

    def __init__(self, db: Database | None = None) -> None:
        self.db = db or get_database()
        self.inventory = InventoryRepository(self.db)
        self.sales = SalesRepository(self.db)
        self.predictions = PredictionRepository(self.db)
        self._forecast: ForecastService | None = None
        self._forecast_loaded = False
        self._spoilage: SpoilageModel | None = None
        self._spoilage_loaded = False

    # ── artefact loading, once per process ────────────────────────────
    @property
    def forecast_service(self) -> ForecastService | None:
        """The trained forecast bundle, or ``None`` when none exists."""
        if not self._forecast_loaded:
            self._forecast_loaded = True
            service = ForecastService()
            self._forecast = service if service.load() and service.is_ready else None
            if self._forecast is None:
                LOG.warning("No trained forecast artefact; forecasting unavailable")
        return self._forecast

    @property
    def spoilage_model(self) -> SpoilageModel | None:
        """The trained spoilage classifier, or ``None`` when none exists."""
        if not self._spoilage_loaded:
            self._spoilage_loaded = True
            self._spoilage = SpoilageModel.load()
            if self._spoilage is None:
                LOG.warning("No trained spoilage artefact; risk scoring unavailable")
        return self._spoilage

    def reload(self) -> None:
        """Drop cached artefacts so the next call picks up a fresh retrain."""
        self._forecast, self._forecast_loaded = None, False
        self._spoilage, self._spoilage_loaded = None, False
        try:
            from freshsense.models.model_registry import get_registry

            get_registry().clear_cache()
        except Exception:                                # pragma: no cover
            pass
        LOG.info("Inference caches cleared")

    def status(self) -> dict[str, Any]:
        """Which models are servable — rendered on the Settings page."""
        service = self.forecast_service
        return {
            "forecast_available": service is not None,
            "forecast_products": len(service.products) if service else 0,
            "spoilage_available": self.spoilage_model is not None,
            "spoilage_algorithm": getattr(self.spoilage_model, "algorithm", ""),
        }

    # ── demand ────────────────────────────────────────────────────────
    def forecast_product(self, product: str, *, horizon: int | None = None,
                         persist: bool = True) -> PredictionBatch:
        """Project demand for one product from the persisted artefact."""
        started = time.perf_counter()
        batch = PredictionBatch(prediction_type=DEMAND, model_name=FORECAST_MODEL)

        service = self.forecast_service
        if service is None:
            batch.available = False
            batch.reason = ("No trained forecasting model. Run "
                            "`python scripts/train_models.py` first.")
            return batch

        series = stats_to_series_frame(self.sales.series())
        if series.empty:
            batch.available = False
            batch.reason = "daily_item_stats is empty; nothing to forecast from"
            return batch

        if product not in set(service.products):
            batch.available = False
            batch.reason = (f"No trained model for '{product}'. Trained products: "
                            f"{len(service.products)}.")
            return batch

        horizon = int(horizon or SETTINGS.forecasting.get("horizon_days", 14))
        try:
            result: ForecastResult = service.forecast(product, series, horizon=horizon)
        except ValueError as exc:
            batch.available = False
            batch.reason = f"Cannot forecast '{product}': {exc}"
            return batch

        batch.payload = result
        batch.rows = len(result.future)
        batch.metrics = dict(result.metrics)
        if persist:
            batch.persisted = self._persist_forecast(product, result)
        batch.latency_ms = int((time.perf_counter() - started) * 1000)
        return batch

    def forecast_all(self, *, horizon: int = 2,
                     persist: bool = False) -> PredictionBatch:
        """Short-horizon demand across every trained product."""
        started = time.perf_counter()
        batch = PredictionBatch(prediction_type=DEMAND, model_name=FORECAST_MODEL)

        service = self.forecast_service
        if service is None:
            batch.available = False
            batch.reason = "No trained forecasting model."
            return batch

        series = stats_to_series_frame(self.sales.series())
        if series.empty:
            batch.available = False
            batch.reason = "daily_item_stats is empty"
            return batch

        frame = service.forecast_all(series, horizon=horizon)
        batch.payload = frame
        batch.rows = len(frame)
        if persist and not frame.empty:
            batch.persisted = self._persist_forecast_summary(frame, horizon)
        batch.latency_ms = int((time.perf_counter() - started) * 1000)
        return batch

    def _persist_forecast(self, product: str, result: ForecastResult) -> int:
        """Write one row per forecast day, keyed to its horizon date.

        ``horizon_date`` is what makes the prediction scoreable: monitoring can
        later join it to the actual units sold on that day. Confidence falls as
        backtest error rises, so an unproven model is recorded at low confidence
        rather than being indistinguishable from a good one.
        """
        mape = result.metrics.get("model_mape")
        confidence = max(0.1, 1.0 - float(mape) / 100.0) if mape is not None else 0.4

        written = 0
        for _, row in result.future.iterrows():
            try:
                self.predictions.record(
                    entity_type="product", entity_id=product,
                    prediction_type=DEMAND, value=float(row["predicted"]),
                    label=str(row.get("day_of_week", "")),
                    confidence=round(confidence, 3), model_name=FORECAST_MODEL,
                    horizon_date=pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
                )
                written += 1
            except Exception as exc:                     # pragma: no cover
                LOG.warning("Could not persist a forecast row: %s", exc)
                break
        return written

    def _persist_forecast_summary(self, frame: pd.DataFrame, horizon: int) -> int:
        written = 0
        for _, row in frame.iterrows():
            mape = row.get("mape")
            confidence = (max(0.1, 1.0 - float(mape) / 100.0)
                          if pd.notna(mape) else 0.4)
            try:
                self.predictions.record(
                    entity_type="product", entity_id=str(row["product_name"]),
                    prediction_type=DEMAND, value=float(row["forecast_units"]),
                    label=f"{horizon}d total", confidence=round(confidence, 3),
                    model_name=FORECAST_MODEL,
                )
                written += 1
            except Exception as exc:                     # pragma: no cover
                LOG.warning("Could not persist a forecast summary row: %s", exc)
                break
        return written

    # ── spoilage ──────────────────────────────────────────────────────
    def assess_inventory(self, *, active_only: bool = True, limit: int = 500,
                         explain: bool = True,
                         persist: bool = True) -> PredictionBatch:
        """Score current stock for spoilage risk, with per-batch explanations."""
        started = time.perf_counter()
        batch = PredictionBatch(prediction_type=SPOILAGE_RISK,
                                model_name=SPOILAGE_MODEL)

        model = self.spoilage_model
        if model is None:
            batch.available = False
            batch.reason = ("No trained spoilage model. Run "
                            "`python scripts/train_models.py` first.")
            return batch

        raw = (self.inventory.active(limit=limit) if active_only
               else self.inventory.load_dataframe().head(limit))
        if raw.empty:
            batch.available = False
            batch.reason = "No batches to assess"
            return batch

        inventory = batches_to_inventory_frame(raw)
        assessments: list[RiskAssessment] = model.assess(inventory, explain=explain)

        batch.payload = assessments
        batch.rows = len(assessments)
        batch.metrics = {
            "critical": sum(1 for a in assessments if a.risk_band == "Critical"),
            "high": sum(1 for a in assessments if a.risk_band == "High"),
            "mean_risk": round(
                sum(a.risk_score for a in assessments) / max(len(assessments), 1), 4
            ),
        }
        if persist:
            batch.persisted = self._persist_assessments(assessments)
        batch.latency_ms = int((time.perf_counter() - started) * 1000)
        return batch

    def assess_batch(self, batch_id: str) -> RiskAssessment | None:
        """Score one batch. Returns ``None`` when the model or batch is absent."""
        model = self.spoilage_model
        if model is None:
            return None
        record = self.inventory.get(batch_id)
        if record is None:
            LOG.warning("assess_batch: no batch '%s'", batch_id)
            return None
        inventory = batches_to_inventory_frame(pd.DataFrame([record]))
        assessments = model.assess(inventory, explain=True)
        return assessments[0] if assessments else None

    def _persist_assessments(self, assessments: list[RiskAssessment]) -> int:
        """Write risk scores with their explanation, in one batched insert.

        The drivers are stored alongside the score because a risk figure without
        a reason is not actionable — and re-deriving it later would need the
        exact model version that produced it.
        """
        rows = [
            {
                "entity_type": "batch", "entity_id": a.batch_id,
                "prediction_type": SPOILAGE_RISK, "value": a.risk_score,
                "label": a.risk_band, "confidence": a.confidence,
                "explanation": a.drivers, "model_name": SPOILAGE_MODEL,
            }
            for a in assessments if a.batch_id
        ]
        if not rows:
            return 0
        try:
            return int(self.predictions.record_many(rows))
        except AttributeError:
            # PredictionRepository in this project exposes record() only.
            written = 0
            for row in rows:
                self.predictions.record(**row)
                written += 1
            return written
        except Exception as exc:                         # pragma: no cover
            LOG.warning("Could not persist spoilage assessments: %s", exc)
            return 0

    # ── application-facing views ──────────────────────────────────────
    def risk_frame(self, *, active_only: bool = True,
                   limit: int = 200) -> pd.DataFrame:
        """Risk scores as a table the UI can render directly."""
        batch = self.assess_inventory(active_only=active_only, limit=limit,
                                      explain=False, persist=False)
        if not batch.ok:
            return pd.DataFrame(columns=["batch_id", "risk_score", "risk_band",
                                         "confidence"])
        return pd.DataFrame([
            {"batch_id": a.batch_id, "risk_score": a.risk_score,
             "risk_band": a.risk_band, "confidence": a.confidence}
            for a in batch.payload
        ]).sort_values("risk_score", ascending=False).reset_index(drop=True)

    def scoreable_backlog(self) -> pd.DataFrame:
        """Predictions still awaiting an observed outcome."""
        return self.predictions.unscored()


_SERVICE: InferenceService | None = None


def get_inference_service(db: Database | None = None) -> InferenceService:
    """Process-wide inference service, so artefacts load once."""
    global _SERVICE
    if _SERVICE is None or db is not None:
        _SERVICE = InferenceService(db)
    return _SERVICE


__all__ = [
    "InferenceService", "PredictionBatch", "get_inference_service",
    "DEMAND", "SPOILAGE_RISK",
]