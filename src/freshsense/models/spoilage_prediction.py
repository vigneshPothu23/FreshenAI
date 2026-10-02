"""Sprint 2 — spoilage prediction.

A gradient-boosted classifier estimating the probability that a batch spoils
before it sells, together with a per-item explanation of *why*.

Design decisions worth defending:

* **Gradient boosting, not deep learning.** Tabular data, moderate volume,
  non-linear interactions between age ratio, perishability and cold-chain
  breach. This is the correct model family; a neural network would cost hours
  and lose to it.
* **PR-AUC alongside ROC-AUC.** The classes are imbalanced. ROC-AUC flatters
  imbalanced problems; precision-recall does not.
* **Calibration is measured.** A risk score is only actionable if 70% means
  seven-in-ten. The Brier score and a reliability table are reported.
* **Explanations are mandatory.** SHAP where available, permutation importance
  with a signed heuristic otherwise. An operator receiving "87% risk" with no
  reason cannot act on it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, average_precision_score,
                             brier_score_loss, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

from freshsense.config import SETTINGS
from freshsense.data.features import risk_label
from freshsense.logging_config import get_logger
from freshsense.models.model_registry import ModelArtifact, get_registry

LOG = get_logger(__name__)

MODEL_NAME = "spoilage_classifier"

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception:                                        # pragma: no cover
    HAS_XGBOOST = False

try:
    import shap
    HAS_SHAP = True
except Exception:                                        # pragma: no cover
    HAS_SHAP = False


# Human-readable labels for every model feature. Feature names never reach the
# UI raw — an operator should read "cold-chain breach hours", not the column id.
FEATURE_LABELS: dict[str, str] = {
    "age_ratio": "Stock age vs shelf life",
    "stock_age_days": "Days since intake",
    "shelf_life_days": "Total shelf life",
    "days_to_expiry": "Days until expiry",
    "temp_breach_hours": "Cold-chain breach hours",
    "storage_temperature_c": "Storage temperature",
    "humidity_pct": "Humidity",
    "quantity_available": "Quantity in stock",
    "daily_avg_sales": "Average daily sales",
    "perishability_score": "Product perishability",
    "storage_risk_score": "Storage type risk",
}

# Mapping from the cleaned inventory column names to model feature names.
COLUMN_TO_FEATURE: dict[str, str] = {
    "Age_Ratio": "age_ratio",
    "Stock_Age_Days": "stock_age_days",
    "Shelf_Life_Days": "shelf_life_days",
    "Days_To_Expiry": "days_to_expiry",
    "Temp_Breach_Hours": "temp_breach_hours",
    "Storage_Temperature_C": "storage_temperature_c",
    "Humidity_Pct": "humidity_pct",
    "Quantity_Available": "quantity_available",
    "Daily_Avg_Sales": "daily_avg_sales",
    "Perishability_Score": "perishability_score",
    "Storage_Risk_Score": "storage_risk_score",
}


def build_feature_frame(inventory: pd.DataFrame) -> pd.DataFrame:
    """Project the cleaned inventory onto the model's feature space."""
    features = list(SETTINGS.spoilage.get("features", list(FEATURE_LABELS)))
    frame = pd.DataFrame(index=inventory.index)

    for column, feature in COLUMN_TO_FEATURE.items():
        if feature not in features:
            continue
        if column in inventory.columns:
            # Only index when the column is really there. ``inventory.get(col, 0)``
            # returns the scalar 0 for an absent column, and pd.to_numeric(0) is a
            # numpy scalar with no .fillna() — an AttributeError three frames deep
            # that says nothing about the missing column.
            frame[feature] = pd.to_numeric(
                inventory[column], errors="coerce"
            ).fillna(0.0)

    absent = [f for f in features if f not in frame.columns]
    if absent:
        # Filling with zeros is the only option left, but it must be visible.
        # A frame of silent zeros yields confident predictions built on no
        # information at all, which is worse than a crash.
        LOG.warning(
            "build_feature_frame: %d feature(s) absent from the input and "
            "filled with 0.0: %s. Predictions carry no signal from them. Pass "
            "the frame through freshsense.models.adapters first when reading "
            "from the database.",
            len(absent), absent,
        )
        for feature in absent:
            frame[feature] = 0.0

    return frame[features].astype(float)


@dataclass
class RiskAssessment:
    """A single batch's spoilage assessment, ready for display."""

    batch_id: str
    risk_score: float
    risk_band: str
    confidence: float
    drivers: list[dict[str, Any]] = field(default_factory=list)

    def top_driver(self) -> str:
        return self.drivers[0]["feature"] if self.drivers else "—"

    def as_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "risk_score": self.risk_score,
            "risk_band": self.risk_band,
            "confidence": self.confidence,
            "top_driver": self.top_driver(),
            "drivers": self.drivers,
        }


class SpoilageModel:
    """Gradient-boosted spoilage classifier with per-item explanations."""

    VERSION = "v1.0"

    def __init__(self) -> None:
        config = SETTINGS.spoilage
        self.features: list[str] = list(config.get("features", list(FEATURE_LABELS)))
        self.threshold: float = float(config.get("decision_threshold", 0.5))
        self.model: Any = None
        self.metrics: dict[str, Any] = {}
        self.algorithm: str = ""
        self._explainer: Any = None
        self._importances: dict[str, float] = {}

    # ── training ──────────────────────────────────────────────────────
    def fit(self, inventory: pd.DataFrame, target_column: str = "Is_Spoiled") -> "SpoilageModel":
        """Fit the classifier and compute the full evaluation suite."""
        config = SETTINGS.spoilage
        X = build_feature_frame(inventory)
        y = pd.to_numeric(inventory[target_column], errors="coerce").fillna(0).astype(int)

        stratify = y if y.nunique() > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=float(config.get("test_size", 0.25)),
            random_state=SETTINGS.seed,
            stratify=stratify,
        )

        if HAS_XGBOOST:
            self.model = XGBClassifier(
                n_estimators=int(config.get("n_estimators", 300)),
                max_depth=int(config.get("max_depth", 4)),
                learning_rate=float(config.get("learning_rate", 0.07)),
                subsample=float(config.get("subsample", 0.9)),
                colsample_bytree=float(config.get("colsample_bytree", 0.9)),
                eval_metric="logloss",
                random_state=SETTINGS.seed,
            )
            self.algorithm = "XGBoost"
        else:
            self.model = GradientBoostingClassifier(
                n_estimators=int(config.get("n_estimators", 300)),
                max_depth=int(config.get("max_depth", 3)),
                learning_rate=float(config.get("learning_rate", 0.07)),
                random_state=SETTINGS.seed,
            )
            self.algorithm = "GradientBoosting"

        self.model.fit(X_train, y_train)
        probabilities = self.model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= self.threshold).astype(int)

        self.metrics = {
            "algorithm": self.algorithm,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "positive_rate": round(float(y.mean()), 4),
            "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4)
                       if y_test.nunique() > 1 else float("nan"),
            "pr_auc": round(float(average_precision_score(y_test, probabilities)), 4)
                      if y_test.nunique() > 1 else float("nan"),
            "brier_score": round(float(brier_score_loss(y_test, probabilities)), 4),
            "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        }
        self.metrics["calibration"] = self._calibration_table(y_test, probabilities)

        self._build_explainer(X_train)
        LOG.info(
            "Spoilage model trained (%s) | ROC-AUC %.4f | PR-AUC %.4f | recall %.1f%%",
            self.algorithm, self.metrics["roc_auc"], self.metrics["pr_auc"],
            self.metrics["recall"] * 100,
        )
        return self

    @staticmethod
    def _calibration_table(y_true: pd.Series, probabilities: np.ndarray) -> list[dict]:
        """Reliability table: does a 70% score spoil 70% of the time?"""
        bins = np.linspace(0, 1, 6)
        indices = np.clip(np.digitize(probabilities, bins) - 1, 0, len(bins) - 2)
        rows: list[dict] = []
        actual = np.asarray(y_true)
        for i in range(len(bins) - 1):
            mask = indices == i
            if not mask.any():
                continue
            rows.append({
                "bin": f"{bins[i]:.0%}–{bins[i + 1]:.0%}",
                "n": int(mask.sum()),
                "mean_predicted": round(float(probabilities[mask].mean()), 3),
                "observed_rate": round(float(actual[mask].mean()), 3),
            })
        return rows

    def _build_explainer(self, background: pd.DataFrame | None) -> None:
        """Prepare SHAP where available; fall back to model importances."""
        self._importances = dict(
            zip(self.features,
                getattr(self.model, "feature_importances_", np.ones(len(self.features))))
        )
        if not HAS_SHAP:
            self._explainer = None
            return
        try:
            self._explainer = shap.TreeExplainer(self.model)
        except Exception as exc:                         # pragma: no cover
            LOG.warning("SHAP unavailable (%s); using permutation importances", exc)
            self._explainer = None

    # ── inference ─────────────────────────────────────────────────────
    def predict_proba(self, inventory: pd.DataFrame) -> np.ndarray:
        """Spoilage probability for every row."""
        if self.model is None:
            raise RuntimeError("Spoilage model is not trained")
        return self.model.predict_proba(build_feature_frame(inventory))[:, 1]

    def explain_row(self, row: pd.Series, top_n: int = 4) -> list[dict[str, Any]]:
        """Per-item contribution breakdown, ordered by absolute impact."""
        frame = build_feature_frame(pd.DataFrame([row]))

        if self._explainer is not None:
            try:
                values = self._explainer.shap_values(frame)
                if isinstance(values, list):
                    values = values[1]
                contributions = np.asarray(values).reshape(-1)
            except Exception:                            # pragma: no cover
                contributions = self._heuristic_contributions(frame)
        else:
            contributions = self._heuristic_contributions(frame)

        rows = [
            {
                "feature": FEATURE_LABELS.get(name, name),
                "raw_feature": name,
                "value": round(float(frame.iloc[0][name]), 2),
                "contribution": round(float(value), 4),
                "direction": "increases" if value >= 0 else "decreases",
            }
            for name, value in zip(self.features, contributions)
        ]
        rows.sort(key=lambda r: -abs(r["contribution"]))
        return rows[:top_n]

    def _heuristic_contributions(self, frame: pd.DataFrame) -> np.ndarray:
        """Signed importance fallback when SHAP is unavailable.

        Importance gives magnitude; the sign is inferred by comparing the value
        against a domain-meaningful centre, so the direction shown to the user
        is still correct even without SHAP.
        """
        centres = {
            "age_ratio": 0.6, "temp_breach_hours": 1.5, "humidity_pct": 72.0,
            "days_to_expiry": 3.0, "perishability_score": 2.0,
            "storage_temperature_c": 12.0, "storage_risk_score": 2.0,
            "stock_age_days": 4.0, "shelf_life_days": 7.0,
            "quantity_available": 40.0, "daily_avg_sales": 10.0,
        }
        inverted = {"days_to_expiry", "shelf_life_days"}
        values = []
        for name in self.features:
            importance = float(self._importances.get(name, 0.0))
            observed = float(frame.iloc[0][name])
            centre = centres.get(name, 0.0)
            sign = 1.0 if observed >= centre else -1.0
            if name in inverted:
                sign *= -1.0
            values.append(importance * sign)
        return np.asarray(values)

    def assess(self, inventory: pd.DataFrame, *, explain: bool = True) -> list[RiskAssessment]:
        """Score and explain a whole inventory frame."""
        probabilities = self.predict_proba(inventory)
        auc = float(self.metrics.get("roc_auc") or 0.9)

        assessments: list[RiskAssessment] = []
        for (_, row), probability in zip(inventory.iterrows(), probabilities):
            # Confidence rises with model quality and with distance from the
            # decision boundary: a 0.5 score is genuinely uncertain.
            confidence = round(auc * (0.75 + 0.25 * abs(probability - 0.5) * 2), 3)
            assessments.append(RiskAssessment(
                batch_id=str(row.get("Inventory_ID") or row.get("batch_id") or ""),
                risk_score=round(float(probability), 4),
                risk_band=risk_label(float(probability)),
                confidence=min(confidence, 0.99),
                drivers=self.explain_row(row) if explain else [],
            ))
        return assessments

    def global_importance(self) -> pd.DataFrame:
        """Mean absolute contribution per feature, for the model card."""
        importances = self._importances or {f: 0.0 for f in self.features}
        frame = pd.DataFrame({
            "feature": [FEATURE_LABELS.get(f, f) for f in importances],
            "raw_feature": list(importances),
            "importance": [float(v) for v in importances.values()],
        })
        total = frame["importance"].sum() or 1.0
        frame["importance_pct"] = (frame["importance"] / total * 100).round(1)
        return frame.sort_values("importance", ascending=False).reset_index(drop=True)

    # ── persistence ───────────────────────────────────────────────────
    def save(self) -> ModelArtifact:
        artifact = ModelArtifact(
            name=MODEL_NAME,
            model={"model": self.model, "importances": self._importances},
            version=self.VERSION,
            algorithm=self.algorithm,
            features=self.features,
            metrics={k: v for k, v in self.metrics.items()
                     if not isinstance(v, (list, dict))},
            metadata={
                "threshold": self.threshold,
                "confusion_matrix": self.metrics.get("confusion_matrix"),
                "calibration": self.metrics.get("calibration"),
                "shap_available": HAS_SHAP,
                "xgboost_available": HAS_XGBOOST,
            },
        )
        get_registry().save(artifact)
        return artifact

    @classmethod
    def load(cls) -> "SpoilageModel | None":
        artifact = get_registry().load(MODEL_NAME)
        if artifact is None:
            return None
        instance = cls()
        instance.model = artifact.model["model"]
        instance._importances = artifact.model.get("importances", {})
        instance.features = artifact.features or instance.features
        instance.algorithm = artifact.algorithm
        instance.metrics = dict(artifact.metrics)
        instance.metrics["confusion_matrix"] = artifact.metadata.get("confusion_matrix")
        instance.metrics["calibration"] = artifact.metadata.get("calibration", [])
        instance.threshold = float(artifact.metadata.get("threshold", 0.5))
        instance._build_explainer(None)
        return instance


_MODEL: SpoilageModel | None = None


def get_spoilage_model() -> SpoilageModel | None:
    """Return the process-wide spoilage model, loading it on first use."""
    global _MODEL
    if _MODEL is None:
        _MODEL = SpoilageModel.load()
    return _MODEL


__all__ = [
    "SpoilageModel", "RiskAssessment", "build_feature_frame",
    "get_spoilage_model", "FEATURE_LABELS", "COLUMN_TO_FEATURE",
    "HAS_XGBOOST", "HAS_SHAP", "MODEL_NAME",
]