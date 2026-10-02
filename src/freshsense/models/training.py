"""Model training: repository -> features -> fit -> evaluate -> persist.

Training and inference are deliberately separate modules. Training is a
scheduled, expensive, side-effecting operation that writes artefacts and
metrics; inference is a cheap read that loads an artefact and returns a result.
Collapsing them is how an application ends up retraining on a page load.

Nothing here reimplements a model. It retrieves data through the Milestone 2
repositories, hands it to the existing ``ForecastService`` and ``SpoilageModel``
unchanged, and records what came back through ``MetricsRepository`` so a
regression is visible in the same place every other metric lives.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from freshsense.config import SETTINGS
from freshsense.db.database import Database, get_database
from freshsense.db.repository import (InventoryRepository, MetricsRepository,
                                      SalesRepository)
from freshsense.logging_config import get_logger
from freshsense.models.adapters import (batches_to_inventory_frame,
                                        stats_to_series_frame)
from freshsense.models.demand_forecasting import ForecastService
from freshsense.models.demand_forecasting import MODEL_NAME as FORECAST_MODEL
from freshsense.models.spoilage_prediction import MODEL_NAME as SPOILAGE_MODEL
from freshsense.models.spoilage_prediction import SpoilageModel

LOG = get_logger(__name__)


@dataclass
class TrainingResult:
    """What one training run produced, and whether it is usable."""

    model_name: str
    trained: bool = False
    rows: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    duration_ms: int = 0
    skipped_reason: str = ""
    artifact_path: str = ""

    @property
    def ok(self) -> bool:
        return self.trained and not self.skipped_reason

    def summary(self) -> str:
        if not self.ok:
            return f"{self.model_name}: skipped — {self.skipped_reason}"
        headline = ", ".join(
            f"{k}={v}" for k, v in list(self.metrics.items())[:4]
            if isinstance(v, (int, float))
        )
        return (f"{self.model_name}: trained on {self.rows:,} row(s) in "
                f"{self.duration_ms} ms ({headline})")

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model_name, "trained": self.trained, "rows": self.rows,
            "duration_ms": self.duration_ms, "skipped_reason": self.skipped_reason,
            **{k: v for k, v in self.metrics.items() if isinstance(v, (int, float))},
        }


class ModelTrainer:
    """Trains both models from the database and records their metrics."""

    def __init__(self, db: Database | None = None) -> None:
        self.db = db or get_database()
        self.inventory = InventoryRepository(self.db)
        self.sales = SalesRepository(self.db)
        self.metrics = MetricsRepository(self.db)

    # ── data retrieval ────────────────────────────────────────────────
    def load_series(self) -> pd.DataFrame:
        """Daily demand from ``daily_item_stats``, in the model's vocabulary."""
        return stats_to_series_frame(self.sales.series())

    def load_inventory(self, *, active_only: bool = False) -> pd.DataFrame:
        """Batches in the model's vocabulary.

        Training deliberately reads *all* batches, not just active ones. The
        spoilage target is only observable once a batch has resolved, so
        restricting to active stock would train on the rows whose outcome is
        precisely the thing not yet known.
        """
        frame = (self.inventory.active() if active_only
                 else self.inventory.load_dataframe())
        return batches_to_inventory_frame(frame)

    # ── forecasting ───────────────────────────────────────────────────
    def train_forecaster(self) -> TrainingResult:
        """Fit one demand model per product and persist the bundle."""
        result = TrainingResult(model_name=FORECAST_MODEL)
        started = time.perf_counter()

        series = self.load_series()
        result.rows = len(series)
        if series.empty:
            result.skipped_reason = (
                "daily_item_stats is empty — run the Milestone 1 pipeline first"
            )
            LOG.warning("%s", result.summary())
            return result

        minimum = int(SETTINGS.forecasting.get("min_history_days", 30))
        longest = int(series.groupby("Product_Name").size().max())
        if longest < minimum:
            result.skipped_reason = (
                f"longest product series is {longest} day(s); the configured "
                f"minimum is {minimum}. A model fitted on less cannot separate "
                f"trend from weekly seasonality."
            )
            LOG.warning("%s", result.summary())
            return result

        service = ForecastService()
        artifact = service.train(series)

        result.trained = True
        result.metrics = dict(artifact.metrics)
        result.artifact_path = str(artifact.name)
        result.duration_ms = int((time.perf_counter() - started) * 1000)

        self._record_metrics(FORECAST_MODEL, result.metrics, dataset="backtest")
        LOG.info("%s", result.summary())
        return result

    # ── spoilage ──────────────────────────────────────────────────────
    def train_spoilage(self) -> TrainingResult:
        """Fit the spoilage classifier and persist it with its evaluation."""
        result = TrainingResult(model_name=SPOILAGE_MODEL)
        started = time.perf_counter()

        inventory = self.load_inventory()
        result.rows = len(inventory)
        if inventory.empty:
            result.skipped_reason = (
                "batches is empty — run the Milestone 1 pipeline first"
            )
            LOG.warning("%s", result.summary())
            return result

        if "Is_Spoiled" not in inventory.columns:
            result.skipped_reason = (
                "no Is_Spoiled column after adaptation; the batches table must "
                "carry is_spoiled for supervised training"
            )
            LOG.warning("%s", result.summary())
            return result

        positives = int(inventory["Is_Spoiled"].sum())
        if positives < 2 or positives == len(inventory):
            result.skipped_reason = (
                f"the target has {positives} positive(s) out of {len(inventory)}; "
                f"a classifier needs both classes present to learn anything"
            )
            LOG.warning("%s", result.summary())
            return result

        model = SpoilageModel().fit(inventory)
        artifact = model.save()

        result.trained = True
        result.metrics = {k: v for k, v in model.metrics.items()
                          if isinstance(v, (int, float))}
        result.artifact_path = str(artifact.name)
        result.duration_ms = int((time.perf_counter() - started) * 1000)

        self._record_metrics(SPOILAGE_MODEL, result.metrics, dataset="test")
        LOG.info("%s", result.summary())
        return result

    # ── orchestration ─────────────────────────────────────────────────
    def train_all(self) -> dict[str, TrainingResult]:
        """Train both models. One failing does not prevent the other."""
        return {
            FORECAST_MODEL: self.train_forecaster(),
            SPOILAGE_MODEL: self.train_spoilage(),
        }

    def _record_metrics(self, model_name: str, metrics: dict[str, Any],
                        *, dataset: str) -> int:
        """Store evaluation results through the existing MetricsRepository.

        Recorded rather than only logged so ``MetricsRepository.history()`` can
        show a metric moving across retrains — which is the only way a slow
        degradation becomes visible before it reaches a user.
        """
        try:
            written = self.metrics.record(model_name, metrics, dataset=dataset)
            LOG.info("Recorded %d metric(s) for '%s'", written, model_name)
            return written
        except Exception as exc:                         # pragma: no cover
            LOG.warning("Could not record metrics for '%s': %s", model_name, exc)
            return 0

    @staticmethod
    def report(results: dict[str, TrainingResult]) -> pd.DataFrame:
        """Training outcomes as a table, for scripts and the Settings page."""
        if not results:
            return pd.DataFrame(columns=["model", "trained", "rows", "duration_ms"])
        return pd.DataFrame([r.as_dict() for r in results.values()])


def train_all(db: Database | None = None) -> dict[str, TrainingResult]:
    """Convenience entry point for ``scripts/train_models.py``."""
    return ModelTrainer(db).train_all()


__all__ = ["ModelTrainer", "TrainingResult", "train_all"]