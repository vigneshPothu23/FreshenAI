"""Sprint 2 — demand forecasting.

An additive decomposition fitted with ridge regression:

    units_sold ≈ trend + weekly seasonality + temperature effect + festival effect

This is structurally the same decomposition Prophet performs, implemented in
scikit-learn. Three reasons for the choice:

1. **Installability.** No compiler toolchain, no Stan backend. It resolves
   cleanly on Python 3.13 where Prophet frequently does not.
2. **Fit for the data shape.** Forty-five short, partly intermittent series with
   exogenous regressors. Prophet targets one long, smooth series; deep sequence
   models need far more history than 180 points per product.
3. **Interpretability.** Each component can be plotted separately, so the
   dashboard can show *why* the forecast moves rather than only *where* it goes.

Every model is reported against a seasonal-naive baseline. A forecast that
cannot beat "same weekday last week" is not a result, and MASE makes that
comparison explicit and scale-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from freshsense.config import SETTINGS
from freshsense.logging_config import get_logger
from freshsense.models.model_registry import ModelArtifact, get_registry

LOG = get_logger(__name__)

MODEL_NAME = "demand_forecaster"


# ══════════════════════════════════════════════════════════════════════════
def seasonal_naive(history: np.ndarray, horizon: int, period: int = 7) -> np.ndarray:
    """Repeat the last full seasonal period forward. The baseline to beat."""
    if len(history) < period:
        return np.repeat(history[-1] if len(history) else 0.0, horizon)
    window = history[-period:]
    return np.tile(window, int(np.ceil(horizon / period)))[:horizon]


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean absolute percentage error, guarded against division by zero."""
    actual = np.asarray(actual, dtype=float)
    denominator = np.clip(np.abs(actual), 1e-6, None)
    return float(np.mean(np.abs((actual - predicted) / denominator)) * 100)


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((np.asarray(actual, dtype=float) - predicted) ** 2)))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(np.asarray(actual, dtype=float) - predicted)))


def mase(actual: np.ndarray, predicted: np.ndarray, history: np.ndarray,
         period: int = 7) -> float:
    """Mean absolute scaled error.

    Scaled by the in-sample seasonal-naive error, so a value below 1.0 means the
    model beats the naive benchmark. Scale-free, which makes it comparable across
    products with very different volumes.
    """
    history = np.asarray(history, dtype=float)
    if len(history) <= period:
        return float("nan")
    scale = np.mean(np.abs(history[period:] - history[:-period]))
    if scale <= 1e-9:
        return float("nan")
    return float(mae(actual, predicted) / scale)


def chennai_temperature(days: pd.DatetimeIndex) -> np.ndarray:
    """Climatological daily mean temperature for Chennai.

    Used to supply the temperature regressor for future dates, where no
    observation exists. Parameters come from the ``forecasting.chennai_climate``
    configuration section.
    """
    climate = SETTINGS.forecasting.get("chennai_climate", {})
    mean = float(climate.get("annual_mean_temp_c", 30.5))
    amplitude = float(climate.get("amplitude_c", 5.0))
    peak = float(climate.get("peak_day_of_year", 105))
    doy = days.dayofyear.to_numpy(dtype=float)
    # Cosine, not sine. sin(2*pi*(doy - peak)/365) peaks a quarter of a year
    # *after* ``peak``, so a parameter named peak_day_of_year placed the hottest
    # day 91 days late — mid-July instead of the configured mid-April. Chennai
    # is hottest from April to June, and this regressor is synthesised for future
    # dates, so the phase error fed the model a summer temperature during the
    # monsoon. Cosine peaks exactly at ``peak``, which is what the name promises.
    return mean + amplitude * np.cos(2 * np.pi * (doy - peak) / 365.0)


# ══════════════════════════════════════════════════════════════════════════
@dataclass
class ForecastResult:
    """Everything the UI needs to render a forecast and defend it."""

    product: str
    history: pd.DataFrame
    future: pd.DataFrame
    components: pd.DataFrame
    metrics: dict[str, float]
    backtest: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def beats_baseline(self) -> bool:
        """Whether backtest error came in below the seasonal-naive benchmark."""
        return bool(self.metrics.get("model_mape", 1e9)
                    < self.metrics.get("baseline_mape", 0))

    @property
    def next_2_days(self) -> float:
        """Total projected demand over the next two days."""
        return float(self.future["predicted"].head(2).sum()) if len(self.future) else 0.0


class DemandForecaster:
    """Additive decomposition demand model with an explicit design matrix."""

    VERSION = "v1.0"

    def __init__(self, n_fourier: int | None = None, alpha: float | None = None) -> None:
        config = SETTINGS.forecasting
        self.n_fourier = int(n_fourier or config.get("fourier_terms", 3))
        self.alpha = float(alpha if alpha is not None else config.get("ridge_alpha", 1.0))
        self.model = Ridge(alpha=self.alpha)
        self.origin: pd.Timestamp | None = None
        self.residual_sigma: float = 1.0
        self.feature_names: list[str] = []

    # ── design matrix ─────────────────────────────────────────────────
    def _design(self, dates: Any, temperature: Any, festival: Any) -> np.ndarray:
        """Build the additive design matrix.

        Columns: linear trend, quadratic trend, Fourier weekly harmonics,
        temperature, festival indicator.
        """
        index = pd.to_datetime(pd.Series(np.asarray(dates))).reset_index(drop=True)
        elapsed = (index - self.origin).dt.days.to_numpy(dtype=float)

        columns = [elapsed, elapsed**2 / 1000.0]
        names = ["trend", "trend_squared"]

        weekday = index.dt.dayofweek.to_numpy(dtype=float)
        for k in range(1, self.n_fourier + 1):
            columns.append(np.sin(2 * np.pi * k * weekday / 7.0))
            columns.append(np.cos(2 * np.pi * k * weekday / 7.0))
            names.extend([f"weekly_sin_{k}", f"weekly_cos_{k}"])

        columns.append(np.asarray(temperature, dtype=float))
        columns.append(np.asarray(festival, dtype=float))
        names.extend(["temperature", "festival"])

        self.feature_names = names
        return np.column_stack(columns)

    # ── fitting ───────────────────────────────────────────────────────
    def fit(self, frame: pd.DataFrame) -> "DemandForecaster":
        """Fit on a single product's daily series.

        Args:
            frame: Must contain ``Date`` and ``Units_Sold``, and optionally
                ``Ambient_Temp_C`` and ``Is_Festival``.

        Returns:
            This instance, fitted.
        """
        data = frame.sort_values("Date").copy()
        data["Date"] = pd.to_datetime(data["Date"])
        self.origin = data["Date"].min()

        temperature = data.get(
            "Ambient_Temp_C", pd.Series(30.5, index=data.index)
        ).fillna(30.5)
        festival = data.get("Is_Festival", pd.Series(0, index=data.index)).fillna(0)

        design = self._design(data["Date"], temperature, festival)
        target = data["Units_Sold"].to_numpy(dtype=float)

        self.model.fit(design, target)
        self.residual_sigma = float(np.std(target - self.model.predict(design))) or 1.0
        return self

    def predict(self, dates: Any, temperature: Any = None,
                festival: Any = None) -> np.ndarray:
        """Predict for arbitrary dates, synthesising regressors when absent.

        Predictions are clipped at zero: negative demand is not physically
        meaningful, and a negative value would corrupt every downstream surplus
        calculation that subtracts it from stock on hand.
        """
        index = pd.to_datetime(pd.DatetimeIndex(np.asarray(dates)))
        if temperature is None:
            temperature = chennai_temperature(index)
        if festival is None:
            festival = np.zeros(len(index))
        return np.clip(self.model.predict(self._design(index, temperature, festival)), 0, None)

    def decompose(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return each additive component separately, for plotting.

        This is what lets the dashboard show *why* a forecast moves — a rising
        trend reads very differently from a weekend spike, and a single fitted
        line cannot distinguish them.
        """
        data = frame.sort_values("Date").copy()
        data["Date"] = pd.to_datetime(data["Date"])
        temperature = data.get(
            "Ambient_Temp_C", pd.Series(30.5, index=data.index)
        ).fillna(30.5)
        festival = data.get("Is_Festival", pd.Series(0, index=data.index)).fillna(0)

        design = self._design(data["Date"], temperature, festival)
        coefficients = self.model.coef_

        components = pd.DataFrame({"Date": data["Date"].to_numpy()})
        components["trend"] = (
            design[:, 0] * coefficients[0]
            + design[:, 1] * coefficients[1]
            + self.model.intercept_
        )
        weekly = np.zeros(len(data))
        for i in range(2, 2 + 2 * self.n_fourier):
            weekly += design[:, i] * coefficients[i]
        components["weekly"] = weekly
        components["temperature"] = design[:, -2] * coefficients[-2]
        components["festival"] = design[:, -1] * coefficients[-1]
        return components


# ══════════════════════════════════════════════════════════════════════════
class ForecastService:
    """Trains, persists and serves per-product demand forecasts."""

    def __init__(self) -> None:
        self.config = SETTINGS.forecasting
        self.horizon = int(self.config.get("horizon_days", 14))
        self.holdout = int(self.config.get("holdout_days", 21))
        self.min_history = int(self.config.get("min_history_days", 30))
        self.confidence_z = float(self.config.get("confidence_z", 1.28))
        self._models: dict[str, DemandForecaster] = {}
        self._metrics: dict[str, dict[str, float]] = {}

    # ── training ──────────────────────────────────────────────────────
    def train(self, sales: pd.DataFrame) -> ModelArtifact:
        """Fit one model per product and persist the whole bundle.

        Backtesting uses a strict time-based holdout: the final ``holdout`` days
        are withheld, never a random split. A random split on temporal data
        leaks future information and produces a meaningless error metric.

        Args:
            sales: Daily series carrying ``Date``, ``Product_Name`` and
                ``Units_Sold``, optionally ``Ambient_Temp_C`` and
                ``Is_Festival``.

        Returns:
            The persisted :class:`ModelArtifact` holding every fitted model.
        """
        sales = sales.copy()
        sales["Date"] = pd.to_datetime(sales["Date"])
        products = sorted(sales["Product_Name"].dropna().unique())

        self._models, self._metrics = {}, {}
        skipped: list[str] = []

        for product in products:
            series = sales[sales["Product_Name"] == product].sort_values("Date")
            if len(series) < self.min_history:
                skipped.append(product)
                continue

            train_set = series.iloc[: -self.holdout]
            test_set = series.iloc[-self.holdout:]

            backtest_model = DemandForecaster().fit(train_set)
            predicted = backtest_model.predict(
                test_set["Date"],
                test_set.get("Ambient_Temp_C"),
                test_set.get("Is_Festival"),
            )
            actual = test_set["Units_Sold"].to_numpy(dtype=float)
            baseline = seasonal_naive(
                train_set["Units_Sold"].to_numpy(dtype=float), len(test_set)
            )
            history = train_set["Units_Sold"].to_numpy(dtype=float)

            self._metrics[product] = {
                "model_mape": round(mape(actual, predicted), 2),
                "baseline_mape": round(mape(actual, baseline), 2),
                "model_rmse": round(rmse(actual, predicted), 2),
                "baseline_rmse": round(rmse(actual, baseline), 2),
                "model_mae": round(mae(actual, predicted), 2),
                "model_mase": round(mase(actual, predicted, history), 3),
                "baseline_mase": round(mase(actual, baseline, history), 3),
                "n_observations": int(len(series)),
            }
            # Refit on the full series for production use.
            self._models[product] = DemandForecaster().fit(series)

        aggregate = self._aggregate_metrics()

        if not self._models:
            # Every product fell below min_history_days. Reporting that plainly
            # is right; crashing on a missing aggregate key is not, and it hides
            # the actual cause behind a KeyError from the log statement itself.
            LOG.warning(
                "No product had enough history to train: %d skipped, minimum "
                "is %d day(s). No forecasting artefact was produced.",
                len(skipped), self.min_history,
            )
            artifact = ModelArtifact(
                name=MODEL_NAME,
                model={"models": {}, "metrics": {}},
                version=DemandForecaster.VERSION,
                algorithm="Ridge additive decomposition (untrained)",
                features=[],
                metrics={"n_products": 0, "products_beating_baseline": 0},
                metadata={"products": [], "skipped_products": skipped,
                          "horizon_days": self.horizon,
                          "holdout_days": self.holdout,
                          "baseline": "seasonal_naive(period=7)"},
            )
            get_registry().save(artifact)
            return artifact

        LOG.info(
            "Forecaster trained on %d product(s) (%d skipped) | "
            "mean MAPE %.2f%% vs baseline %.2f%% | beats baseline on %d/%d",
            len(self._models), len(skipped), aggregate["mean_model_mape"],
            aggregate["mean_baseline_mape"], aggregate["products_beating_baseline"],
            len(self._models),
        )

        artifact = ModelArtifact(
            name=MODEL_NAME,
            model={"models": self._models, "metrics": self._metrics},
            version=DemandForecaster.VERSION,
            algorithm="Ridge additive decomposition (trend + Fourier weekly + "
                      "temperature + festival)",
            features=["trend", "trend_squared", "weekly_fourier",
                      "temperature", "festival"],
            metrics=aggregate,
            metadata={
                "products": list(self._models),
                "skipped_products": skipped,
                "horizon_days": self.horizon,
                "holdout_days": self.holdout,
                "baseline": "seasonal_naive(period=7)",
            },
        )
        get_registry().save(artifact)
        return artifact

    def _aggregate_metrics(self) -> dict[str, float]:
        """Summarise per-product backtest metrics into a single bundle."""
        if not self._metrics:
            return {}
        frame = pd.DataFrame(self._metrics).T
        beating = int((frame["model_mape"] < frame["baseline_mape"]).sum())
        return {
            "n_products": int(len(frame)),
            "mean_model_mape": round(float(frame["model_mape"].mean()), 2),
            "mean_baseline_mape": round(float(frame["baseline_mape"].mean()), 2),
            "median_model_mape": round(float(frame["model_mape"].median()), 2),
            "mean_model_rmse": round(float(frame["model_rmse"].mean()), 2),
            "mean_model_mase": round(float(frame["model_mase"].mean(skipna=True)), 3),
            "products_beating_baseline": beating,
            "pct_beating_baseline": round(beating / len(frame) * 100, 1),
        }

    # ── loading ───────────────────────────────────────────────────────
    def load(self) -> bool:
        """Load persisted models.

        Returns:
            True when an artefact was found and loaded, False otherwise. The
            caller decides what to do about it: this must not raise, because a
            fresh install with no trained model is an expected state, not an
            error.
        """
        artifact = get_registry().load(MODEL_NAME)
        if artifact is None:
            return False
        self._models = artifact.model.get("models", {})
        self._metrics = artifact.model.get("metrics", {})
        return True

    @property
    def is_ready(self) -> bool:
        """Whether at least one product model is available to serve."""
        return bool(self._models)

    @property
    def products(self) -> list[str]:
        """Products with a trained model, sorted."""
        return sorted(self._models)

    def metrics_frame(self) -> pd.DataFrame:
        """Per-product backtest metrics — rendered on the Forecast page.

        Carries ``beats_baseline`` and ``improvement_pct`` explicitly, because a
        MAPE quoted without its baseline says nothing about whether the model is
        worth using.
        """
        if not self._metrics:
            return pd.DataFrame()
        frame = pd.DataFrame(self._metrics).T.reset_index()
        frame = frame.rename(columns={"index": "product"})
        frame["beats_baseline"] = frame["model_mape"] < frame["baseline_mape"]
        frame["improvement_pct"] = (
            (frame["baseline_mape"] - frame["model_mape"])
            / frame["baseline_mape"].clip(lower=0.01) * 100
        ).round(1)
        return frame.sort_values("model_mape")

    # ── inference ─────────────────────────────────────────────────────
    def forecast(
        self, product: str, sales: pd.DataFrame, horizon: int | None = None
    ) -> ForecastResult:
        """Produce a forecast with prediction intervals and components.

        Args:
            product: Product to forecast.
            sales: Daily series in the model's column vocabulary.
            horizon: Days ahead. Defaults to the configured horizon.

        Returns:
            A :class:`ForecastResult` carrying history, future, components and
            the backtest metrics for this product.

        Raises:
            ValueError: If the product has too little history to fit at all.
        """
        horizon = int(horizon or self.horizon)
        series = sales[sales["Product_Name"] == product].sort_values("Date").copy()
        series["Date"] = pd.to_datetime(series["Date"])

        model = self._models.get(product)
        if model is None:
            if len(series) < 7:
                raise ValueError(
                    f"Insufficient history for '{product}' ({len(series)} rows)"
                )
            model = DemandForecaster().fit(series)
            self._models[product] = model

        last_date = series["Date"].max()
        future_dates = pd.date_range(
            last_date + timedelta(days=1), periods=horizon, freq="D"
        )
        predicted = model.predict(future_dates)
        margin = self.confidence_z * model.residual_sigma

        future = pd.DataFrame({
            "date": future_dates,
            "predicted": np.round(predicted, 1),
            "lower": np.round(np.clip(predicted - margin, 0, None), 1),
            "upper": np.round(predicted + margin, 1),
            "day_of_week": future_dates.day_name(),
        })

        metrics = dict(self._metrics.get(product, {}))
        return ForecastResult(
            product=product,
            history=series[["Date", "Units_Sold"]].rename(
                columns={"Date": "date", "Units_Sold": "units_sold"}
            ),
            future=future,
            components=model.decompose(series),
            metrics=metrics,
        )

    def forecast_all(
        self, sales: pd.DataFrame, horizon: int = 2
    ) -> pd.DataFrame:
        """Short-horizon demand for every product — feeds surplus calculation.

        A product whose forecast fails is logged and skipped rather than
        aborting the whole sweep: one unfittable series must not deny the
        operator a view of the other forty-four.
        """
        rows: list[dict[str, Any]] = []
        for product in self.products:
            try:
                result = self.forecast(product, sales, horizon=horizon)
                rows.append({
                    "product_name": product,
                    "forecast_units": round(float(result.future["predicted"].sum()), 1),
                    "daily_mean": round(float(result.future["predicted"].mean()), 1),
                    "mape": result.metrics.get("model_mape", np.nan),
                    "beats_baseline": result.beats_baseline,
                })
            except Exception as exc:                     # pragma: no cover
                LOG.warning("Forecast failed for %s: %s", product, exc)
        return pd.DataFrame(rows)


_SERVICE: ForecastService | None = None


def get_forecast_service() -> ForecastService:
    """Return the process-wide forecast service, loading artefacts on first use.

    Cached so a Streamlit rerun does not reload the artefact bundle on every
    widget interaction, and so no code path can accidentally retrain during a
    user request.
    """
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = ForecastService()
        _SERVICE.load()
    return _SERVICE


__all__ = [
    "DemandForecaster", "ForecastService", "ForecastResult", "get_forecast_service",
    "seasonal_naive", "mape", "rmse", "mae", "mase", "chennai_temperature",
    "MODEL_NAME",
]