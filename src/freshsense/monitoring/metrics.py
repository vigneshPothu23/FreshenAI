"""Sprint 6 — metric computation.

Monitoring is not logging. A log records that a prediction was made; monitoring
records whether it was *right*. The difference lives in ``prediction_outcomes``,
and this module is what fills and reads it.

Four families of metric, each answering a different question:

============  ===============================================================
Family        Question
============  ===============================================================
Model         Are the predictions still accurate against observed reality?
Business      Is the system producing value in rupees and kilograms?
System        Is it responsive and reliable?
Drift         Has the live data moved away from what the models were fitted on?
============  ===============================================================

Two principles run through it:

**Absence of evidence is reported as such.** With no observed outcomes yet, the
model metrics say so rather than returning zeros — a dashboard showing 0%
accuracy when nothing has been scored is worse than one showing "not yet
measurable", because the first invites action on a number that means nothing.

**Every metric carries a status band.** A raw figure is not monitoring; a figure
plus the threshold it is judged against is. Bands come from ``config.yaml``, so
what counts as an alert is configuration rather than folklore buried in code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Sequence

import numpy as np
import pandas as pd

from freshsense.config import SETTINGS
from freshsense.db.repository import (InventoryRepository, MonitoringRepository,
                                      OrderRepository, PredictionRepository,
                                      RecommendationRepository, SalesRepository)
from freshsense.db.session import Database, get_database
from freshsense.logging_config import get_logger

LOG = get_logger(__name__)

OK, WARN, ALERT, UNKNOWN = "ok", "warn", "alert", "unknown"


# ══════════════════════════════════════════════════════════════════════════
@dataclass
class MetricValue:
    """One measured quantity, its status band and the reason for that band."""

    name: str
    value: float | None
    unit: str = ""
    status: str = OK
    detail: str = ""
    higher_is_better: bool = True

    @property
    def is_measurable(self) -> bool:
        return self.value is not None

    def display(self) -> str:
        """Formatted for a dashboard tile."""
        if self.value is None:
            return "—"
        if self.unit == "%":
            return f"{self.value:,.1f}%"
        if self.unit == SETTINGS.currency:
            return f"{self.unit}{self.value:,.0f}"
        if self.unit == "ms":
            return f"{self.value:,.0f} ms"
        return f"{self.value:,.2f}{(' ' + self.unit) if self.unit else ''}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "value": self.value, "unit": self.unit,
            "display": self.display(), "status": self.status,
            "detail": self.detail,
        }


def band(
    value: float | None,
    *,
    warn: float,
    alert: float,
    higher_is_better: bool = False,
) -> str:
    """Classify a value against warn and alert thresholds."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return UNKNOWN
    if higher_is_better:
        if value < alert:
            return ALERT
        return WARN if value < warn else OK
    if value > alert:
        return ALERT
    return WARN if value > warn else OK


# ══════════════════════════════════════════════════════════════════════════
# Drift
# ══════════════════════════════════════════════════════════════════════════
def population_stability_index(
    reference: Sequence[float],
    current: Sequence[float],
    *,
    bins: int = 10,
    epsilon: float = 1e-6,
    min_samples: int = 30,
) -> float:
    """Population Stability Index between a reference and a current sample.

    PSI sums ``(current% - reference%) * ln(current% / reference%)`` over bins
    cut from the reference distribution. Cutting on the reference rather than on
    the pooled data is the point: it measures movement *away from what the model
    was fitted on*, which is the thing that degrades predictions.

    Conventional reading: below 0.1 stable, 0.1–0.25 moderate shift, above 0.25
    significant shift. Thresholds are configurable in ``config.yaml``.

    Below ``min_samples`` the result is ``nan`` rather than a number. PSI over
    a handful of observations is dominated by sampling noise and routinely
    exceeds the alert threshold on a perfectly healthy model — returning it
    would send someone retraining for no reason.

    Returns:
        The PSI, or ``nan`` when either sample is too small to bin meaningfully.
    """
    ref = pd.Series(reference, dtype="float64").dropna()
    cur = pd.Series(current, dtype="float64").dropna()
    if len(ref) < max(bins, min_samples) or len(cur) < min_samples:
        return float("nan")

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(ref, quantiles))
    if len(edges) < 3:
        return float("nan")                              # near-constant reference
    edges[0], edges[-1] = -np.inf, np.inf

    ref_share = np.histogram(ref, bins=edges)[0] / len(ref)
    cur_share = np.histogram(cur, bins=edges)[0] / len(cur)
    ref_share = np.clip(ref_share, epsilon, None)
    cur_share = np.clip(cur_share, epsilon, None)

    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def categorical_drift(
    reference: Sequence[Any],
    current: Sequence[Any],
    *,
    epsilon: float = 1e-6,
    min_samples: int = 30,
) -> float:
    """PSI over categories rather than numeric bins.

    Categories present in one sample and absent from the other are retained at
    the epsilon floor, so a category disappearing registers as drift instead of
    being quietly dropped.
    """
    ref = pd.Series(reference).dropna().astype(str)
    cur = pd.Series(current).dropna().astype(str)
    if len(ref) < min_samples or len(cur) < min_samples:
        return float("nan")

    categories = sorted(set(ref) | set(cur))
    ref_share = np.clip(
        ref.value_counts().reindex(categories).fillna(0).to_numpy() / len(ref),
        epsilon, None,
    )
    cur_share = np.clip(
        cur.value_counts().reindex(categories).fillna(0).to_numpy() / len(cur),
        epsilon, None,
    )
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


@dataclass
class DriftReport:
    """Per-feature drift, with the worst offender surfaced."""

    features: pd.DataFrame = field(default_factory=pd.DataFrame)
    reference_rows: int = 0
    current_rows: int = 0
    note: str = ""

    @property
    def measurable(self) -> bool:
        return not self.features.empty

    @property
    def worst(self) -> dict[str, Any] | None:
        if not self.measurable:
            return None
        row = self.features.iloc[0]
        return {"feature": row["feature"], "psi": float(row["psi"]),
                "status": row["status"]}

    @property
    def status(self) -> str:
        if not self.measurable:
            return UNKNOWN
        statuses = set(self.features["status"])
        return ALERT if ALERT in statuses else (WARN if WARN in statuses else OK)

    def narrative(self) -> str:
        if not self.measurable:
            return self.note or "Not enough data to assess drift."
        worst = self.worst or {}
        drifting = int((self.features["status"] != OK).sum())
        if drifting == 0:
            return (
                f"No meaningful drift across {len(self.features)} feature(s); "
                f"the largest movement is {worst.get('feature')} at PSI "
                f"{worst.get('psi', 0):.3f}."
            )
        return (
            f"{drifting} of {len(self.features)} feature(s) have shifted away "
            f"from the training distribution. Largest: {worst.get('feature')} "
            f"at PSI {worst.get('psi', 0):.3f} ({worst.get('status')}). "
            f"Predictions on those features are less reliable than the "
            f"backtest suggests."
        )


# ══════════════════════════════════════════════════════════════════════════
class ModelMonitor:
    """Accuracy of live predictions against observed outcomes."""

    def __init__(self, db: Database | None = None) -> None:
        self.db = db or get_database()
        self.predictions = PredictionRepository(self.db)
        self.inventory = InventoryRepository(self.db)
        self.sales = SalesRepository(self.db)
        self.monitoring = MonitoringRepository(self.db)

    # ── closing the loop ──────────────────────────────────────────────
    def record_observed_outcomes(self, *, limit: int = 2000) -> dict[str, int]:
        """Match past predictions to what actually happened.

        This is the step that turns a prediction log into monitoring. Spoilage
        predictions are matched against the batch's recorded ``is_spoiled``;
        demand predictions against the units actually sold on the horizon date.
        Only predictions whose horizon has passed are scored, and each is scored
        once.

        Returns:
            Counts of newly recorded outcomes by prediction type.
        """
        recorded = {"spoilage_risk": 0, "demand": 0}

        pending = self.db.query(
            """
            SELECT p.prediction_id, p.entity_id, p.prediction_type, p.value,
                   p.horizon_date
            FROM predictions p
            LEFT JOIN prediction_outcomes o ON o.prediction_id = p.prediction_id
            WHERE o.outcome_id IS NULL
            ORDER BY p.created_at DESC LIMIT ?
            """,
            (int(limit),),
        )
        if pending.empty:
            return recorded

        # ── spoilage: the batch either spoiled or it did not ──────────
        spoilage = pending[pending["prediction_type"] == "spoilage_risk"]
        if not spoilage.empty:
            truth = self.db.query(
                "SELECT batch_id, is_spoiled, status FROM batches"
            ).set_index("batch_id")
            for _, row in spoilage.iterrows():
                batch = truth.loc[row["entity_id"]] if row["entity_id"] in truth.index else None
                if batch is None:
                    continue
                # A batch still on shelf has no outcome yet — scoring it as
                # "did not spoil" would flatter the model on stock that simply
                # has not had time to fail.
                if str(batch["status"]) == "active":
                    continue
                self.predictions.record_outcome(
                    int(row["prediction_id"]), float(batch["is_spoiled"])
                )
                recorded["spoilage_risk"] += 1

        # ── demand: units actually sold on the horizon date ───────────
        demand = pending[
            (pending["prediction_type"] == "demand")
            & pending["horizon_date"].notna()
        ]
        if not demand.empty:
            actuals = self.db.query(
                "SELECT stat_date, product_name, units_sold FROM daily_item_stats"
            )
            if not actuals.empty:
                lookup = actuals.set_index(["product_name", "stat_date"])["units_sold"]
                today = datetime.now().strftime("%Y-%m-%d")
                for _, row in demand.iterrows():
                    horizon = str(row["horizon_date"])[:10]
                    if horizon >= today:
                        continue                         # the day has not happened
                    key = (row["entity_id"], horizon)
                    if key not in lookup.index:
                        continue
                    self.predictions.record_outcome(
                        int(row["prediction_id"]), float(lookup.loc[key])
                    )
                    recorded["demand"] += 1

        if any(recorded.values()):
            LOG.info("Recorded observed outcomes: %s", recorded)
        return recorded

    # ── accuracy ──────────────────────────────────────────────────────
    def spoilage_metrics(self) -> tuple[list[MetricValue], pd.DataFrame]:
        """Classification quality of live spoilage predictions."""
        outcomes = self.predictions.outcomes("spoilage_risk")
        if outcomes.empty:
            return [MetricValue(
                "Spoilage accuracy", None, "%", UNKNOWN,
                "No batch with a spoilage prediction has resolved yet, so live "
                "accuracy is not yet measurable. Backtest figures are on the "
                "model card.",
            )], outcomes

        threshold = float(SETTINGS.spoilage.get("decision_threshold", 0.5))
        predicted = (outcomes["predicted"] >= threshold).astype(int)
        actual = outcomes["actual_value"].astype(int)

        true_positive = int(((predicted == 1) & (actual == 1)).sum())
        false_positive = int(((predicted == 1) & (actual == 0)).sum())
        false_negative = int(((predicted == 0) & (actual == 1)).sum())

        accuracy = float((predicted == actual).mean() * 100)
        precision = (
            true_positive / (true_positive + false_positive) * 100
            if (true_positive + false_positive) else float("nan")
        )
        recall = (
            true_positive / (true_positive + false_negative) * 100
            if (true_positive + false_negative) else float("nan")
        )
        brier = float(((outcomes["predicted"] - actual) ** 2).mean())

        return [
            MetricValue("Spoilage accuracy", round(accuracy, 1), "%",
                        band(accuracy, warn=80, alert=70, higher_is_better=True),
                        f"{len(outcomes)} resolved prediction(s)"),
            MetricValue("Spoilage recall", round(recall, 1), "%",
                        band(recall, warn=75, alert=60, higher_is_better=True),
                        "Share of batches that spoiled which were flagged in "
                        "advance — the metric that matters most, because a miss "
                        "is unrecovered waste"),
            MetricValue("Spoilage precision", round(precision, 1), "%",
                        band(precision, warn=65, alert=50, higher_is_better=True),
                        "Share of flagged batches that actually spoiled — low "
                        "precision means needless discounting"),
            MetricValue("Calibration (Brier)", round(brier, 4), "",
                        band(brier, warn=0.15, alert=0.25),
                        "Lower is better; measures whether a 70% score spoils "
                        "roughly seven times in ten"),
        ], outcomes

    def demand_metrics(self) -> tuple[list[MetricValue], pd.DataFrame]:
        """Live forecast error against realised sales."""
        outcomes = self.predictions.outcomes("demand")
        if outcomes.empty:
            return [MetricValue(
                "Live forecast MAPE", None, "%", UNKNOWN,
                "No forecast horizon has elapsed with a matching actual yet.",
            )], outcomes

        actual = outcomes["actual_value"].astype(float)
        predicted = outcomes["predicted"].astype(float)
        denominator = actual.abs().clip(lower=1e-6)

        mape = float((predicted - actual).abs().div(denominator).mean() * 100)
        mae = float((predicted - actual).abs().mean())
        bias = float((predicted - actual).mean())

        return [
            MetricValue("Live forecast MAPE", round(mape, 1), "%",
                        band(mape, warn=20, alert=35),
                        f"{len(outcomes)} scored prediction(s)"),
            MetricValue("Live forecast MAE", round(mae, 1), "units",
                        band(mae, warn=25, alert=50), "Mean absolute error"),
            MetricValue("Forecast bias", round(bias, 1), "units",
                        band(abs(bias), warn=10, alert=25),
                        "Positive means systematic over-prediction, which "
                        "causes over-ordering and then waste"),
        ], outcomes

    def drift(self, *, reference_days: int = 90, current_days: int = 14) -> DriftReport:
        """Compare recent demand and price distributions against an earlier window.

        The reference window is the older period the models were effectively
        fitted on; the current window is what they are now being asked to
        predict. Chennai's climate makes seasonal movement expected, so this is
        a prompt to check rather than an automatic failure.
        """
        config = SETTINGS.monitoring
        warn = float(config.get("drift_psi_warn", 0.10))
        alert = float(config.get("drift_psi_alert", 0.25))

        series = self.sales.series()
        if series.empty:
            return DriftReport(note="No sales history is loaded.")

        series["stat_date"] = pd.to_datetime(series["stat_date"])
        latest = series["stat_date"].max()
        current_start = latest - timedelta(days=current_days)
        reference_start = current_start - timedelta(days=reference_days)

        current = series[series["stat_date"] > current_start]
        reference = series[
            (series["stat_date"] > reference_start)
            & (series["stat_date"] <= current_start)
        ]
        if reference.empty or current.empty:
            return DriftReport(
                note=f"Need more than {current_days} days of history to compare "
                     f"windows; only {series['stat_date'].nunique()} day(s) present.",
            )

        numeric = [
            ("units_sold", "Daily units sold"),
            ("avg_selling_price", "Average selling price"),
            ("ambient_temp_c", "Ambient temperature"),
            ("humidity_pct", "Humidity"),
        ]
        rows: list[dict[str, Any]] = []
        for column, label in numeric:
            if column not in series.columns:
                continue
            value = population_stability_index(reference[column], current[column])
            if np.isnan(value):
                continue
            rows.append({
                "feature": label, "psi": round(value, 4),
                "status": band(value, warn=warn, alert=alert),
                "kind": "numeric",
            })

        if "category" in series.columns:
            value = categorical_drift(reference["category"], current["category"])
            if not np.isnan(value):
                rows.append({
                    "feature": "Category mix", "psi": round(value, 4),
                    "status": band(value, warn=warn, alert=alert),
                    "kind": "categorical",
                })

        if not rows:
            return DriftReport(note="No comparable feature had enough variation.")

        frame = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
        return DriftReport(
            features=frame,
            reference_rows=len(reference),
            current_rows=len(current),
        )


# ══════════════════════════════════════════════════════════════════════════
class BusinessMonitor:
    """Value delivered, in the units an executive actually cares about."""

    def __init__(self, db: Database | None = None) -> None:
        self.db = db or get_database()
        self.orders = OrderRepository(self.db)
        self.inventory = InventoryRepository(self.db)
        self.recommendations = RecommendationRepository(self.db)
        self.monitoring = MonitoringRepository(self.db)

    def kpis(self, *, days: int = 90) -> list[MetricValue]:
        """Headline commercial and sustainability metrics."""
        config = SETTINGS.monitoring.get("business", {})
        commission_pct = float(config.get("commission_pct", 4.0))
        co2_per_kg = float(config.get("co2_kg_per_kg_food", 2.5))
        currency = SETTINGS.currency

        orders = self.orders.business_kpis(days=days)
        stock = self.inventory.kpi_summary()

        if not orders.get("orders"):
            return [MetricValue("Orders", 0, "", UNKNOWN,
                                "No orders recorded in the window.")]

        gmv = float(orders.get("gmv", 0))
        savings = float(orders.get("savings", 0))
        waste_kg = float(orders.get("waste_prevented_kg", 0))
        dispute_rate = float(orders.get("dispute_rate_pct", 0))
        # A 0% acceptance rate means nothing until enough recommendations have
        # been shown to judge. Alerting on the first few would train the
        # operator to ignore the metric.
        shown = int(self.db.scalar("SELECT COUNT(*) FROM recommendations"))
        acceptance = self.recommendations.acceptance_rate() if shown else None
        acceptance_status = (
            UNKNOWN if shown < 10
            else band(acceptance, warn=25, alert=10, higher_is_better=True)
        )
        acceptance_detail = (
            f"Only {shown} recommendation(s) shown so far — too few to judge"
            if shown < 10 else
            "Share of shown recommendations acted on — the honest test of "
            "whether they are useful"
        )

        # Capital at risk is the exposure the platform exists to reduce, so it
        # belongs beside the value it has already recovered.
        at_risk = float(stock.get("capital_at_risk", 0))

        return [
            MetricValue("Gross merchandise value", round(gmv, 2), currency, OK,
                        f"{int(orders['orders'])} order(s) over {days} day(s)"),
            MetricValue("Platform revenue", round(gmv * commission_pct / 100, 2),
                        currency, OK, f"At {commission_pct:.1f}% commission"),
            MetricValue("Buyer savings", round(savings, 2), currency, OK,
                        "Discount value passed to buyers against MRP"),
            MetricValue("Waste prevented", round(waste_kg, 1), "kg", OK,
                        "Stock sold that would otherwise have expired"),
            MetricValue("CO₂ avoided", round(waste_kg * co2_per_kg, 1), "kg", OK,
                        f"At {co2_per_kg} kg CO₂ per kg of food waste avoided"),
            MetricValue("Capital at risk", round(at_risk, 2), currency,
                        band(at_risk, warn=gmv * 0.5, alert=gmv * 1.5),
                        "Cost value of stock expiring within two days — the "
                        "exposure still to be recovered"),
            MetricValue("Dispute rate", round(dispute_rate, 2), "%",
                        band(dispute_rate, warn=5, alert=10),
                        "Orders raising a dispute; grading exists to keep this low"),
            MetricValue("Recommendation acceptance",
                        round(acceptance, 1) if acceptance is not None else None,
                        "%", acceptance_status, acceptance_detail),
            MetricValue("Active buyers", float(orders.get("buyers", 0)), "", OK,
                        "Distinct buyers transacting in the window"),
        ]

    def daily_series(self, *, limit: int = 60) -> pd.DataFrame:
        """Revenue, savings and waste prevented per day, for trend charts."""
        frame = self.orders.daily_revenue(limit=limit)
        if frame.empty:
            return frame
        co2 = float(
            SETTINGS.monitoring.get("business", {}).get("co2_kg_per_kg_food", 2.5)
        )
        frame["co2_avoided_kg"] = (frame["waste_prevented_kg"] * co2).round(1)
        return frame

    def persist_daily_snapshot(self) -> int:
        """Write today's headline metrics to ``business_metrics``.

        Stored rather than recomputed so the executive dashboard can show a
        trend without re-aggregating the whole order log on every page load.
        """
        today = datetime.now().strftime("%Y-%m-%d")
        written = 0
        for metric in self.kpis():
            if metric.value is None:
                continue
            written += self.monitoring.record_business_metric(
                today, metric.name, float(metric.value)
            )
        return written


# ══════════════════════════════════════════════════════════════════════════
class SystemMonitor:
    """Responsiveness and reliability of the application itself."""

    def __init__(self, db: Database | None = None) -> None:
        self.db = db or get_database()
        self.monitoring = MonitoringRepository(self.db)

    def kpis(self) -> list[MetricValue]:
        events = self.monitoring.events(limit=2000)
        warn_ms = float(SETTINGS.monitoring.get("latency_warn_ms", 1500))

        if events.empty:
            return [MetricValue("Logged operations", 0, "", UNKNOWN,
                                "No instrumented operation has run yet.")]

        latency = events["latency_ms"].astype(float)
        p50, p95 = float(latency.quantile(0.5)), float(latency.quantile(0.95))
        success = float((events["status"] == "ok").mean() * 100)

        metrics = [
            MetricValue("Operations logged", float(len(events)), "", OK,
                        f"Across {events['component'].nunique()} component(s)"),
            MetricValue("Median latency", round(p50, 0), "ms",
                        band(p50, warn=warn_ms / 2, alert=warn_ms)),
            MetricValue("P95 latency", round(p95, 0), "ms",
                        band(p95, warn=warn_ms, alert=warn_ms * 2),
                        "The tail users actually notice"),
            MetricValue("Success rate", round(success, 1), "%",
                        band(success, warn=98, alert=95, higher_is_better=True)),
        ]

        llm = self.monitoring.llm_stats()
        if not llm.empty:
            calls = int(llm["calls"].sum())
            stub_share = 0.0
            if "stub" in set(llm["provider"]):
                stub_calls = int(llm.loc[llm["provider"] == "stub", "calls"].sum())
                stub_share = stub_calls / max(calls, 1) * 100
            metrics.append(MetricValue(
                "LLM calls", float(calls), "", OK,
                f"{stub_share:.0f}% served by the offline stub provider"
            ))

        agents = self.monitoring.agent_stats()
        if not agents.empty:
            replans = int(agents["replans"].sum())
            steps = int(agents["steps"].sum())
            metrics.append(MetricValue(
                "Agent replans", float(replans), "",
                band(replans / max(steps, 1) * 100, warn=20, alert=40),
                f"{replans} recovery step(s) across {steps} agent step(s) — some "
                f"replanning is healthy, persistent replanning is not"
            ))
        return metrics

    def component_breakdown(self) -> pd.DataFrame:
        return self.monitoring.component_stats()

    def agent_breakdown(self) -> pd.DataFrame:
        return self.monitoring.agent_stats()

    def recent_failures(self, limit: int = 25) -> pd.DataFrame:
        return self.db.query(
            "SELECT created_at, component, action, detail FROM system_logs "
            "WHERE status != 'ok' ORDER BY created_at DESC LIMIT ?",
            (int(limit),),
        )


# ══════════════════════════════════════════════════════════════════════════
class MonitoringService:
    """Facade the Monitoring page and the evaluation harness both use."""

    def __init__(self, db: Database | None = None) -> None:
        self.db = db or get_database()
        self.model = ModelMonitor(self.db)
        self.business = BusinessMonitor(self.db)
        self.system = SystemMonitor(self.db)

    def health_summary(self, *, refresh_outcomes: bool = True) -> dict[str, Any]:
        """Everything the Monitoring page renders, in one call."""
        if refresh_outcomes:
            self.model.record_observed_outcomes()

        spoilage, spoilage_outcomes = self.model.spoilage_metrics()
        demand, demand_outcomes = self.model.demand_metrics()
        drift = self.model.drift()
        business = self.business.kpis()
        system = self.system.kpis()

        everything = spoilage + demand + business + system
        statuses = {m.status for m in everything} | {drift.status}
        overall = (
            ALERT if ALERT in statuses
            else WARN if WARN in statuses
            else OK if OK in statuses else UNKNOWN
        )

        return {
            "overall_status": overall,
            "model_metrics": spoilage + demand,
            "business_metrics": business,
            "system_metrics": system,
            "drift": drift,
            "spoilage_outcomes": spoilage_outcomes,
            "demand_outcomes": demand_outcomes,
            "component_stats": self.system.component_breakdown(),
            "agent_stats": self.system.agent_breakdown(),
            "recent_failures": self.system.recent_failures(),
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    @staticmethod
    def to_frame(metrics: Sequence[MetricValue]) -> pd.DataFrame:
        """Metric list as a table for display."""
        if not metrics:
            return pd.DataFrame(columns=["name", "display", "status", "detail"])
        return pd.DataFrame([m.as_dict() for m in metrics])[
            ["name", "display", "status", "detail"]
        ]


__all__ = [
    "MetricValue", "band", "OK", "WARN", "ALERT", "UNKNOWN",
    "population_stability_index", "categorical_drift", "DriftReport",
    "ModelMonitor", "BusinessMonitor", "SystemMonitor", "MonitoringService",
]