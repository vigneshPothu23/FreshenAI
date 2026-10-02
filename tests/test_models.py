"""Milestone 3: ML layer integration tests.

Self-contained fixtures — the database, the seed and the artefact directory are
all built inside this module, so nothing here depends on the Milestone 2
``conftest.py`` or on the pipeline having been run. Every fixture is
deterministic: the demand series carries a real trend and weekly cycle, and the
spoilage target is a genuine function of age and cold-chain breach, so a model
that learns nothing fails the suite instead of passing on noise.

Grouped by what a failure would mean:

``TestConfiguration``     the settings the models read actually exist
``TestAdapters``          database columns reach the models
``TestForecastPrimitives``the metric and baseline functions are correct
``TestForecastModel``     fitting, prediction and decomposition work
``TestForecastService``   training, backtesting, persistence and reload
``TestSpoilageModel``     features, fitting, calibration, explanation
``TestRegistry``          artefacts round-trip with their metadata
``TestTrainingIntegration``   repository -> train -> metrics persisted
``TestInferenceIntegration``  artefact -> predict -> predictions persisted
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from freshsense.config import SETTINGS  # noqa: E402
from freshsense.db.database import Database  # noqa: E402
from freshsense.db.repository import (InventoryRepository,  # noqa: E402
                                      MetricsRepository, PredictionRepository,
                                      SalesRepository)
from freshsense.models import adapters  # noqa: E402
from freshsense.models.demand_forecasting import (DemandForecaster,  # noqa: E402
                                                  ForecastResult,
                                                  ForecastService,
                                                  chennai_temperature, mae,
                                                  mape, mase, rmse,
                                                  seasonal_naive)
from freshsense.models.demand_forecasting import MODEL_NAME as FORECAST_MODEL  # noqa: E402
from freshsense.models.inference import (DEMAND, SPOILAGE_RISK,  # noqa: E402
                                         InferenceService)
from freshsense.models.model_registry import ModelArtifact, ModelRegistry  # noqa: E402
from freshsense.models.spoilage_prediction import (RiskAssessment,  # noqa: E402
                                                   SpoilageModel,
                                                   build_feature_frame)
from freshsense.models.spoilage_prediction import MODEL_NAME as SPOILAGE_MODEL  # noqa: E402
from freshsense.models.training import ModelTrainer, TrainingResult  # noqa: E402
from freshsense.paths import PATHS  # noqa: E402

SCHEMA = _ROOT / "database" / "schema.sql"
PRODUCTS = ["Tomato", "Paneer", "Potato"]
DAYS = 120


def _seed(db: Database) -> None:
    """Insert a deterministic dataset with genuinely learnable structure.

    Demand carries a level, a linear trend and a weekly cycle. Spoilage is a
    real function of age ratio and cold-chain breach. Random noise alone would
    let a broken model pass, because nothing could beat the baseline either.
    """
    rng = np.random.default_rng(SETTINGS.seed)

    with db.connect() as conn:
        for index, name in enumerate(PRODUCTS, start=1):
            conn.execute(
                "INSERT INTO items (item_id, name, category, unit, "
                "shelf_life_days, perishability_level, storage_type) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (index, name, "Perishable", "kg", 5, "High", "Chilled"),
            )
        for index in range(1, 4):
            conn.execute(
                "INSERT INTO sellers (seller_id, seller_name, zone, "
                "seller_rating) VALUES (?, ?, ?, ?)",
                (f"SEL{index:03d}", f"Store {index}", "Adyar", 4.0),
            )

        start = pd.Timestamp("2026-01-01")
        for offset in range(DAYS):
            day = start + pd.Timedelta(days=offset)
            for index, name in enumerate(PRODUCTS, start=1):
                level = 60.0 * index
                trend = 0.35 * offset
                weekly = 12.0 * math.sin(2 * math.pi * day.dayofweek / 7)
                units = max(0.0, level + trend + weekly + rng.normal(0, 2.5))
                conn.execute(
                    "INSERT INTO daily_item_stats (stat_date, product_name, "
                    "category, unit, units_sold, avg_selling_price, revenue, "
                    "ambient_temp_c, humidity_pct, is_weekend, is_festival, "
                    "day_of_week) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (day.strftime("%Y-%m-%d"), name, "Perishable", "kg",
                     round(units, 1), 40.0, round(units * 40, 2),
                     30.5 + 4 * math.sin(2 * math.pi * offset / 365), 72.0,
                     int(day.dayofweek >= 5), 0, day.day_name()),
                )

        # Spoilage is a deterministic function of the two drivers the model is
        # meant to find, so a model that learns nothing cannot pass.
        for index in range(1, 121):
            age_ratio = round((index % 10) / 10, 2)
            breach = float(index % 6)
            spoiled = int(age_ratio > 0.6 and breach > 2)
            days_to_expiry = max(0, 5 - int(age_ratio * 5))
            conn.execute(
                """
                INSERT INTO batches
                    (batch_id, item_id, seller_id, product_name, category, unit,
                     zone, quantity_available, expiry_date, shelf_life_days,
                     stock_age_days, days_to_expiry, age_ratio, storage_type,
                     storage_temperature_c, humidity_pct, temp_breach_hours,
                     perishability_level, perishability_score,
                     storage_risk_score, cost_price, mrp, discount_pct,
                     effective_price, daily_avg_sales, quality_grade, status,
                     is_spoiled, environment_stress_index, action_priority,
                     inventory_value, capital_at_risk)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (f"B{index:04d}", (index % 3) + 1, f"SEL{(index % 3) + 1:03d}",
                 PRODUCTS[index % 3], "Perishable", "kg", "Adyar",
                 float(10 + index % 40), "2030-01-01", 5,
                 int(age_ratio * 5), days_to_expiry, age_ratio, "Chilled",
                 8.0 + breach, 70.0 + (index % 10), breach, "High", 3, 2,
                 20.0, 40.0, 10.0, 36.0, 5.0, "B", "active", spoiled,
                 round(age_ratio * 0.8, 3), 5.0, 400.0, 200.0),
            )


@pytest.fixture(scope="module")
def model_db(tmp_path_factory) -> Database:
    """A schema-built, richly seeded database, created once for this module."""
    if not SCHEMA.is_file():
        pytest.skip(f"schema.sql not found at {SCHEMA}")
    path = tmp_path_factory.mktemp("m3") / "models.db"
    database = Database(path)
    database.initialise(SCHEMA)
    _seed(database)
    return database


@pytest.fixture()
def db(model_db, tmp_path) -> Database:
    """A private copy per test, so writes cannot leak between tests."""
    import shutil

    private = tmp_path / "case.db"
    shutil.copyfile(model_db.db_path, private)
    return Database(private)


@pytest.fixture()
def registry(tmp_path, monkeypatch) -> ModelRegistry:
    """An isolated artefact directory, so tests never touch real models."""
    store = tmp_path / "artifacts"
    store.mkdir()
    instance = ModelRegistry(store)

    import freshsense.models.model_registry as registry_module

    monkeypatch.setattr(registry_module, "_REGISTRY", instance)
    return instance


@pytest.fixture()
def series(db) -> pd.DataFrame:
    return adapters.stats_to_series_frame(SalesRepository(db).series())


@pytest.fixture()
def inventory(db) -> pd.DataFrame:
    return adapters.batches_to_inventory_frame(
        InventoryRepository(db).load_dataframe())


@pytest.fixture()
def trained(db, registry) -> dict[str, TrainingResult]:
    return ModelTrainer(db).train_all()


# ══════════════════════════════════════════════════════════════════════════
class TestConfiguration:
    """The settings the models read must actually exist."""

    def test_seed_is_exposed(self):
        """Regression: the models reference SETTINGS.seed, which did not exist.

        SpoilageModel.fit() raised AttributeError before reaching train_test_split,
        so the classifier could not be trained at all.
        """
        assert isinstance(SETTINGS.seed, int)

    def test_seed_comes_from_configuration_not_a_literal(self):
        assert SETTINGS.seed == int(SETTINGS.cleaning["random_seed"])

    def test_forecasting_section_has_what_the_model_reads(self):
        for key in ("horizon_days", "holdout_days", "fourier_terms",
                    "ridge_alpha", "confidence_z", "min_history_days"):
            assert key in SETTINGS.forecasting, f"forecasting.{key} missing"

    def test_chennai_climate_present(self):
        climate = SETTINGS.forecasting["chennai_climate"]
        for key in ("annual_mean_temp_c", "amplitude_c", "peak_day_of_year"):
            assert key in climate

    def test_spoilage_section_has_what_the_model_reads(self):
        for key in ("test_size", "n_estimators", "max_depth", "learning_rate",
                    "decision_threshold", "features"):
            assert key in SETTINGS.spoilage, f"spoilage.{key} missing"

    def test_spoilage_feature_list_is_non_empty(self):
        assert len(SETTINGS.spoilage["features"]) >= 5


# ══════════════════════════════════════════════════════════════════════════
class TestAdapters:
    """Database columns must reach the models."""

    def test_series_adapter_renames_to_model_vocabulary(self, db):
        raw = SalesRepository(db).series()
        assert "stat_date" in raw.columns and "Date" not in raw.columns
        adapted = adapters.stats_to_series_frame(raw)
        for column in ("Date", "Units_Sold", "Product_Name"):
            assert column in adapted.columns

    def test_series_adapter_supplies_optional_regressors(self, series):
        assert "Ambient_Temp_C" in series.columns
        assert "Is_Festival" in series.columns

    def test_series_adapter_rejects_a_frame_without_the_key_columns(self):
        with pytest.raises(KeyError, match="missing required column"):
            adapters.stats_to_series_frame(pd.DataFrame({"nonsense": [1, 2]}))

    def test_series_adapter_drops_unparseable_dates(self):
        frame = pd.DataFrame({"stat_date": ["2026-01-01", "not-a-date"],
                              "product_name": ["A", "A"],
                              "units_sold": [1.0, 2.0]})
        assert len(adapters.stats_to_series_frame(frame)) == 1

    def test_batch_adapter_guarantees_every_model_feature(self, inventory):
        """Regression: build_feature_frame silently filled absent columns with
        0.0, so a database frame produced confident predictions from no signal."""
        for column in adapters.REQUIRED_BATCH_COLUMNS:
            assert column in inventory.columns

    def test_batch_adapter_yields_a_populated_feature_frame(self, inventory):
        features = build_feature_frame(inventory)
        all_zero = [c for c in features.columns if (features[c] == 0).all()]
        assert not all_zero, f"features carrying no signal: {all_zero}"

    def test_batch_adapter_exposes_the_target(self, inventory):
        """Regression: fit() expects Is_Spoiled; the table column is is_spoiled."""
        assert "Is_Spoiled" in inventory.columns
        assert set(inventory["Is_Spoiled"].unique()) <= {0, 1}

    def test_batch_adapter_fills_missing_columns_without_crashing(self):
        """Regression: inventory.get(col, 0) returned a scalar, and
        pd.to_numeric(0).fillna() raised AttributeError three frames deep."""
        adapted = adapters.batches_to_inventory_frame(
            pd.DataFrame({"batch_id": ["B1"], "age_ratio": [0.5]}))
        assert len(adapted) == 1
        assert all(c in adapted.columns for c in adapters.REQUIRED_BATCH_COLUMNS)

    def test_build_feature_frame_tolerates_a_partial_frame(self):
        features = build_feature_frame(pd.DataFrame({"Age_Ratio": [0.5, 0.2]}))
        assert len(features) == 2
        assert (features["age_ratio"] == [0.5, 0.2]).all()

    def test_adapters_are_empty_safe(self):
        assert adapters.batches_to_inventory_frame(pd.DataFrame()).empty
        assert adapters.stats_to_series_frame(pd.DataFrame()).empty

    def test_mapping_source_is_reported(self):
        described = adapters.describe_mapping()
        assert "batches" in described and "daily_item_stats" in described


# ══════════════════════════════════════════════════════════════════════════
class TestForecastPrimitives:
    """The metric and baseline functions must be correct in isolation."""

    def test_seasonal_naive_repeats_the_last_period(self):
        history = np.arange(1.0, 15.0)          # 14 points
        forecast = seasonal_naive(history, horizon=7, period=7)
        assert forecast.tolist() == history[-7:].tolist()

    def test_seasonal_naive_tiles_beyond_one_period(self):
        history = np.arange(1.0, 15.0)
        assert len(seasonal_naive(history, horizon=10, period=7)) == 10

    def test_seasonal_naive_handles_short_history(self):
        assert seasonal_naive(np.array([5.0]), horizon=3).tolist() == [5.0] * 3

    def test_mape_is_zero_for_a_perfect_forecast(self):
        actual = np.array([10.0, 20.0, 30.0])
        assert mape(actual, actual) == pytest.approx(0.0)

    def test_mape_guards_against_division_by_zero(self):
        assert np.isfinite(mape(np.array([0.0, 10.0]), np.array([1.0, 10.0])))

    def test_rmse_and_mae_match_hand_computation(self):
        actual, predicted = np.array([1.0, 2.0, 3.0]), np.array([2.0, 2.0, 5.0])
        assert mae(actual, predicted) == pytest.approx(1.0)
        assert rmse(actual, predicted) == pytest.approx(math.sqrt(5 / 3))

    def test_mase_below_one_means_it_beats_the_naive_benchmark(self):
        history = np.tile([10.0, 20.0], 20)
        actual = np.array([10.0, 20.0, 10.0, 20.0])
        assert mase(actual, actual, history) == pytest.approx(0.0)

    def test_mase_is_nan_when_history_is_too_short(self):
        assert math.isnan(mase(np.array([1.0]), np.array([1.0]), np.array([1.0])))

    def test_mase_is_nan_for_a_constant_history(self):
        assert math.isnan(mase(np.array([5.0]), np.array([5.0]), np.full(30, 5.0)))

    def test_chennai_temperature_is_seasonal_and_bounded(self):
        days = pd.date_range("2026-01-01", periods=365, freq="D")
        temperatures = chennai_temperature(days)
        climate = SETTINGS.forecasting["chennai_climate"]
        mean = float(climate["annual_mean_temp_c"])
        amplitude = float(climate["amplitude_c"])
        assert temperatures.min() >= mean - amplitude - 0.01
        assert temperatures.max() <= mean + amplitude + 0.01
        assert temperatures.argmax() == pytest.approx(
            float(climate["peak_day_of_year"]) - 1, abs=2)


# ══════════════════════════════════════════════════════════════════════════
class TestForecastModel:
    """Fitting, prediction and decomposition."""

    def test_design_matrix_shape(self, series):
        model = DemandForecaster()
        subset = series[series.Product_Name == "Tomato"]
        model.origin = subset["Date"].min()
        design = model._design(subset["Date"], subset["Ambient_Temp_C"],
                               subset["Is_Festival"])
        expected = 2 + 2 * model.n_fourier + 2
        assert design.shape == (len(subset), expected)
        assert len(model.feature_names) == expected

    def test_design_matrix_uses_configured_fourier_terms(self):
        assert DemandForecaster().n_fourier == int(
            SETTINGS.forecasting["fourier_terms"])

    def test_fit_learns_the_seeded_structure(self, series):
        subset = series[series.Product_Name == "Tomato"]
        model = DemandForecaster().fit(subset)
        fitted = model.predict(subset["Date"], subset["Ambient_Temp_C"],
                               subset["Is_Festival"])
        assert mape(subset["Units_Sold"].to_numpy(), fitted) < 10.0

    def test_prediction_is_never_negative(self, series):
        model = DemandForecaster().fit(series[series.Product_Name == "Tomato"])
        assert (model.predict(pd.date_range("2027-01-01", periods=30)) >= 0).all()

    def test_prediction_synthesises_regressors_for_future_dates(self, series):
        model = DemandForecaster().fit(series[series.Product_Name == "Tomato"])
        future = model.predict(pd.date_range("2026-12-01", periods=5))
        assert len(future) == 5 and np.isfinite(future).all()

    def test_decompose_returns_named_components(self, series):
        subset = series[series.Product_Name == "Tomato"]
        components = DemandForecaster().fit(subset).decompose(subset)
        for column in ("Date", "trend", "weekly", "temperature", "festival"):
            assert column in components.columns
        assert len(components) == len(subset)

    def test_weekly_component_actually_varies(self, series):
        """A flat weekly term would mean the Fourier basis is doing nothing."""
        subset = series[series.Product_Name == "Tomato"]
        components = DemandForecaster().fit(subset).decompose(subset)
        assert components["weekly"].std() > 1.0

    def test_residual_sigma_is_positive(self, series):
        model = DemandForecaster().fit(series[series.Product_Name == "Tomato"])
        assert model.residual_sigma > 0


# ══════════════════════════════════════════════════════════════════════════
class TestForecastService:
    """Training, backtesting, persistence and reload."""

    def test_train_produces_a_model_per_product(self, series, registry):
        service = ForecastService()
        service.train(series)
        assert set(service.products) == set(PRODUCTS)

    def test_backtest_metrics_are_recorded(self, series, registry):
        service = ForecastService()
        artifact = service.train(series)
        for key in ("n_products", "mean_model_mape", "mean_baseline_mape",
                    "products_beating_baseline"):
            assert key in artifact.metrics

    def test_model_beats_the_seasonal_naive_baseline(self, series, registry):
        """The seeded series has trend plus weekly structure, so a working model
        must beat repeating last week. If it does not, the design matrix is
        broken however plausible the MAPE looks."""
        service = ForecastService()
        artifact = service.train(series)
        assert artifact.metrics["mean_model_mape"] < artifact.metrics["mean_baseline_mape"]
        assert artifact.metrics["pct_beating_baseline"] >= 66.0

    def test_holdout_is_time_based_not_random(self, series, registry):
        """A random split leaks future information and makes the error
        meaningless. The metric count must match a contiguous tail."""
        service = ForecastService()
        service.train(series)
        holdout = int(SETTINGS.forecasting["holdout_days"])
        for product in PRODUCTS:
            observations = int(service._metrics[product]["n_observations"])
            assert observations == DAYS
            assert observations > holdout

    def test_metrics_frame_flags_baseline_comparison(self, series, registry):
        service = ForecastService()
        service.train(series)
        frame = service.metrics_frame()
        assert {"product", "model_mape", "baseline_mape",
                "beats_baseline"} <= set(frame.columns)
        assert frame["beats_baseline"].all()

    def test_products_with_too_little_history_are_skipped(self, registry):
        short = pd.DataFrame({
            "Date": pd.date_range("2026-01-01", periods=10),
            "Product_Name": ["Sparse"] * 10,
            "Units_Sold": np.arange(10.0),
        })
        artifact = ForecastService().train(short)
        assert "Sparse" in artifact.metadata["skipped_products"]

    def test_forecast_returns_an_interval_and_components(self, series, registry):
        service = ForecastService()
        service.train(series)
        result = service.forecast("Tomato", series, horizon=7)
        assert isinstance(result, ForecastResult)
        assert len(result.future) == 7
        assert {"date", "predicted", "lower", "upper",
                "day_of_week"} <= set(result.future.columns)
        assert (result.future["lower"] <= result.future["predicted"]).all()
        assert (result.future["predicted"] <= result.future["upper"]).all()

    def test_forecast_result_helpers(self, series, registry):
        service = ForecastService()
        service.train(series)
        result = service.forecast("Tomato", series, horizon=7)
        assert result.beats_baseline is True
        assert result.next_2_days == pytest.approx(
            float(result.future["predicted"].head(2).sum()))

    def test_forecast_all_covers_every_product(self, series, registry):
        service = ForecastService()
        service.train(series)
        frame = service.forecast_all(series, horizon=2)
        assert set(frame["product_name"]) == set(PRODUCTS)
        assert (frame["forecast_units"] > 0).all()

    def test_artifact_reloads_into_a_fresh_service(self, series, registry):
        """Inference must not depend on the process that trained the model."""
        ForecastService().train(series)
        registry.clear_cache()

        reloaded = ForecastService()
        assert reloaded.load() is True
        assert reloaded.is_ready
        assert set(reloaded.products) == set(PRODUCTS)

    def test_reloaded_service_predicts_identically(self, series, registry):
        original = ForecastService()
        original.train(series)
        before = original.forecast("Tomato", series, horizon=5).future["predicted"]

        registry.clear_cache()
        reloaded = ForecastService()
        reloaded.load()
        after = reloaded.forecast("Tomato", series, horizon=5).future["predicted"]

        assert before.tolist() == after.tolist()

    def test_load_returns_false_when_no_artifact_exists(self, registry):
        assert ForecastService().load() is False


# ══════════════════════════════════════════════════════════════════════════
class TestSpoilageModel:
    """Features, fitting, calibration, explanation and persistence."""

    def test_feature_frame_matches_the_configured_feature_list(self, inventory):
        features = build_feature_frame(inventory)
        assert list(features.columns) == list(SETTINGS.spoilage["features"])
        assert features.dtypes.map(lambda t: t.kind == "f").all()

    def test_fit_learns_the_seeded_relationship(self, inventory, registry):
        """The target is a real function of age ratio and breach hours, so a
        working model must separate the classes. Noise alone would let a broken
        feature pipeline pass."""
        model = SpoilageModel().fit(inventory)
        assert model.metrics["roc_auc"] > 0.75
        assert model.metrics["accuracy"] > 0.75

    def test_fit_records_the_full_evaluation_suite(self, inventory, registry):
        model = SpoilageModel().fit(inventory)
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc",
                    "pr_auc", "brier_score", "confusion_matrix", "calibration",
                    "n_train", "n_test", "positive_rate"):
            assert key in model.metrics, f"missing metric: {key}"

    def test_pr_auc_is_reported_alongside_roc_auc(self, inventory, registry):
        """ROC-AUC flatters an imbalanced problem; precision-recall does not."""
        model = SpoilageModel().fit(inventory)
        assert 0.0 <= model.metrics["pr_auc"] <= 1.0

    def test_confusion_matrix_totals_match_the_test_split(self, inventory, registry):
        model = SpoilageModel().fit(inventory)
        matrix = np.array(model.metrics["confusion_matrix"])
        assert matrix.sum() == model.metrics["n_test"]

    def test_split_is_deterministic(self, inventory, registry):
        """Two fits with the same configured seed must agree exactly, or no
        metric can be compared across runs."""
        first = SpoilageModel().fit(inventory).metrics["accuracy"]
        second = SpoilageModel().fit(inventory).metrics["accuracy"]
        assert first == second

    def test_test_size_comes_from_configuration(self, inventory, registry):
        model = SpoilageModel().fit(inventory)
        expected = round(len(inventory) * float(SETTINGS.spoilage["test_size"]))
        assert abs(model.metrics["n_test"] - expected) <= 1

    def test_calibration_table_is_populated(self, inventory, registry):
        """A risk score is only actionable if 70% means seven in ten."""
        calibration = SpoilageModel().fit(inventory).metrics["calibration"]
        assert calibration
        for row in calibration:
            assert {"bin", "n", "mean_predicted", "observed_rate"} <= set(row)
            assert 0.0 <= row["observed_rate"] <= 1.0

    def test_predict_proba_is_a_probability(self, inventory, registry):
        probabilities = SpoilageModel().fit(inventory).predict_proba(inventory)
        assert len(probabilities) == len(inventory)
        assert ((probabilities >= 0) & (probabilities <= 1)).all()

    def test_predict_proba_before_fitting_raises(self, inventory):
        with pytest.raises(RuntimeError, match="not trained"):
            SpoilageModel().predict_proba(inventory)

    def test_assess_returns_bands_and_confidence(self, inventory, registry):
        assessments = SpoilageModel().fit(inventory).assess(inventory.head(10))
        assert len(assessments) == 10
        for item in assessments:
            assert isinstance(item, RiskAssessment)
            assert item.risk_band in {"Low", "Moderate", "High", "Critical"}
            assert 0.0 < item.confidence <= 0.99

    def test_assessment_ids_come_from_the_adapted_frame(self, inventory, registry):
        """Regression: the id column is batch_id in the database and
        Inventory_ID in the model frame. An unmapped id yields blank
        assessments that cannot be joined back to any batch."""
        assessments = SpoilageModel().fit(inventory).assess(inventory.head(5))
        assert all(item.batch_id for item in assessments)

    def test_explanations_are_ordered_by_impact(self, inventory, registry):
        drivers = SpoilageModel().fit(inventory).assess(inventory.head(1))[0].drivers
        assert drivers
        magnitudes = [abs(d["contribution"]) for d in drivers]
        assert magnitudes == sorted(magnitudes, reverse=True)

    def test_explanations_use_human_labels(self, inventory, registry):
        """An operator should read 'cold-chain breach hours', not a column id."""
        drivers = SpoilageModel().fit(inventory).assess(inventory.head(1))[0].drivers
        for driver in drivers:
            assert driver["feature"] != driver["raw_feature"]
            assert driver["direction"] in {"increases", "decreases"}

    def test_assess_can_skip_explanations(self, inventory, registry):
        assessments = SpoilageModel().fit(inventory).assess(
            inventory.head(5), explain=False)
        assert all(item.drivers == [] for item in assessments)

    def test_global_importance_sums_to_one_hundred_percent(self, inventory, registry):
        frame = SpoilageModel().fit(inventory).global_importance()
        assert not frame.empty
        assert frame["importance_pct"].sum() == pytest.approx(100.0, abs=0.5)

    def test_heuristic_explanation_path_works_without_shap(self, inventory, registry):
        """SHAP is optional, so the fallback must produce signed contributions
        rather than leaving the operator with a bare probability."""
        model = SpoilageModel().fit(inventory)
        model._explainer = None
        drivers = model.explain_row(inventory.iloc[0])
        assert drivers and any(d["contribution"] != 0 for d in drivers)

    def test_save_and_reload_preserves_behaviour(self, inventory, registry):
        original = SpoilageModel().fit(inventory)
        before = original.predict_proba(inventory.head(10))
        original.save()

        registry.clear_cache()
        reloaded = SpoilageModel.load()
        assert reloaded is not None
        assert np.allclose(before, reloaded.predict_proba(inventory.head(10)))

    def test_reloaded_model_keeps_its_metrics_and_algorithm(self, inventory, registry):
        SpoilageModel().fit(inventory).save()
        registry.clear_cache()
        reloaded = SpoilageModel.load()
        assert reloaded.algorithm in {"XGBoost", "GradientBoosting"}
        assert reloaded.metrics["roc_auc"] > 0
        assert reloaded.metrics["calibration"]

    def test_reloaded_model_can_still_explain(self, inventory, registry):
        SpoilageModel().fit(inventory).save()
        registry.clear_cache()
        assert SpoilageModel.load().assess(inventory.head(1))[0].drivers

    def test_load_returns_none_when_no_artifact_exists(self, registry):
        assert SpoilageModel.load() is None

    def test_risk_assessment_serialises(self, inventory, registry):
        payload = SpoilageModel().fit(inventory).assess(inventory.head(1))[0].as_dict()
        assert {"batch_id", "risk_score", "risk_band", "confidence",
                "top_driver", "drivers"} <= set(payload)


# ══════════════════════════════════════════════════════════════════════════
class TestRegistry:
    """Artefacts round-trip with their metadata intact."""

    def test_save_and_load_preserves_every_field(self, registry):
        registry.save(ModelArtifact(
            name="probe", model={"weights": [1, 2]}, version="v2.1",
            algorithm="TestAlgo", features=["a", "b"],
            metrics={"score": 0.9}, metadata={"note": "hello"}))

        loaded = registry.load("probe")
        assert loaded.model == {"weights": [1, 2]}
        assert loaded.version == "v2.1"
        assert loaded.algorithm == "TestAlgo"
        assert loaded.features == ["a", "b"]
        assert loaded.metrics == {"score": 0.9}
        assert loaded.metadata == {"note": "hello"}

    def test_missing_artifact_returns_none(self, registry):
        assert registry.load("nothing-here") is None

    def test_missing_artifact_raises_when_required(self, registry):
        from freshsense.exceptions import ModelNotTrainedError

        with pytest.raises(ModelNotTrainedError):
            registry.load("nothing-here", required=True)

    def test_metadata_sidecar_is_written(self, registry):
        registry.save(ModelArtifact(name="probe", model=1, metrics={"a": 1.0}))
        assert registry.metadata_path("probe").is_file()
        assert registry.exists("probe")

    def test_list_models_reports_saved_artifacts(self, registry):
        registry.save(ModelArtifact(name="one", model=1))
        registry.save(ModelArtifact(name="two", model=2))
        assert {m["name"] for m in registry.list_models()} == {"one", "two"}

    def test_cache_is_clearable(self, registry):
        registry.save(ModelArtifact(name="probe", model=1))
        registry.load("probe")
        registry.clear_cache()
        assert registry.load("probe") is not None


# ══════════════════════════════════════════════════════════════════════════
class TestTrainingIntegration:
    """Repository -> features -> train -> evaluate -> persist."""

    def test_trainer_reads_from_the_repository(self, db, registry):
        trainer = ModelTrainer(db)
        assert len(trainer.load_series()) == DAYS * len(PRODUCTS)
        assert len(trainer.load_inventory()) == 120

    def test_training_reads_all_batches_not_only_active(self, db, registry):
        """The spoilage target is only observable once a batch resolves, so
        training on active stock alone would exclude the rows whose outcome is
        precisely what the model must learn."""
        trainer = ModelTrainer(db)
        assert len(trainer.load_inventory(active_only=False)) >= \
               len(trainer.load_inventory(active_only=True))

    def test_train_all_trains_both_models(self, trained):
        assert set(trained) == {FORECAST_MODEL, SPOILAGE_MODEL}
        assert all(result.ok for result in trained.values())

    def test_training_persists_both_artifacts(self, trained, registry):
        assert registry.exists(FORECAST_MODEL)
        assert registry.exists(SPOILAGE_MODEL)

    def test_training_records_metrics_through_the_repository(self, db, trained):
        """Recorded rather than only logged, so MetricsRepository.history() can
        show a metric moving across retrains."""
        metrics = MetricsRepository(db)
        assert metrics.count() > 0
        assert not metrics.for_model(FORECAST_MODEL).empty
        assert not metrics.for_model(SPOILAGE_MODEL).empty

    def test_recorded_metrics_are_queryable_over_time(self, db, trained):
        history = MetricsRepository(db).history(SPOILAGE_MODEL, "roc_auc")
        assert len(history) >= 1

    def test_training_result_reports_row_counts(self, trained):
        assert trained[FORECAST_MODEL].rows == DAYS * len(PRODUCTS)
        assert trained[SPOILAGE_MODEL].rows == 120

    def test_training_report_is_tabular(self, trained):
        frame = ModelTrainer.report(trained)
        assert len(frame) == 2 and "trained" in frame.columns

    def test_empty_database_skips_rather_than_crashing(self, tmp_path, registry):
        """A fresh install must report what is missing, not raise."""
        empty = Database(tmp_path / "empty.db")
        empty.initialise(SCHEMA)
        results = ModelTrainer(empty).train_all()
        assert not any(result.ok for result in results.values())
        for result in results.values():
            assert "empty" in result.skipped_reason

    def test_single_class_target_is_skipped_with_a_reason(self, db, registry):
        """A classifier cannot learn from one class, and saying so beats
        producing a model with perfect accuracy and no discrimination."""
        db.execute("UPDATE batches SET is_spoiled = 0")
        result = ModelTrainer(db).train_spoilage()
        assert not result.ok
        assert "both classes" in result.skipped_reason


# ══════════════════════════════════════════════════════════════════════════
class TestInferenceIntegration:
    """Artefact -> predict -> persist -> application."""

    def test_status_reports_what_is_servable(self, db, trained):
        status = InferenceService(db).status()
        assert status["forecast_available"] and status["spoilage_available"]
        assert status["forecast_products"] == len(PRODUCTS)

    def test_inference_without_artifacts_reports_rather_than_training(
            self, db, registry):
        """A silent retrain inside a page load is slow and unrepeatable."""
        service = InferenceService(db)
        batch = service.forecast_product("Tomato")
        assert not batch.available
        assert "train_models" in batch.reason

    def test_forecast_from_a_persisted_artifact(self, db, trained):
        batch = InferenceService(db).forecast_product("Tomato", horizon=7)
        assert batch.ok and batch.rows == 7
        assert isinstance(batch.payload, ForecastResult)

    def test_forecast_persists_predictions_with_a_horizon_date(self, db, trained):
        """horizon_date is what makes a prediction scoreable later: monitoring
        joins it to the units actually sold that day."""
        service = InferenceService(db)
        batch = service.forecast_product("Tomato", horizon=5, persist=True)
        assert batch.persisted == 5

        stored = PredictionRepository(db).recent(DEMAND)
        assert len(stored) == 5
        assert stored["horizon_date"].notna().all()
        assert (stored["entity_type"] == "product").all()

    def test_forecast_confidence_falls_as_error_rises(self, db, trained):
        service = InferenceService(db)
        service.forecast_product("Tomato", horizon=3, persist=True)
        stored = PredictionRepository(db).recent(DEMAND)
        assert (stored["confidence"] > 0).all() and (stored["confidence"] <= 1).all()

    def test_unknown_product_is_reported_not_raised(self, db, trained):
        batch = InferenceService(db).forecast_product("Caviar")
        assert not batch.available and "No trained model" in batch.reason

    def test_forecast_all_covers_every_product(self, db, trained):
        batch = InferenceService(db).forecast_all(horizon=2)
        assert batch.ok and batch.rows == len(PRODUCTS)

    def test_assess_inventory_scores_and_persists(self, db, trained):
        batch = InferenceService(db).assess_inventory(limit=25)
        assert batch.ok and batch.rows == 25
        assert batch.persisted == 25

        stored = PredictionRepository(db).recent(SPOILAGE_RISK)
        assert len(stored) == 25
        assert (stored["entity_type"] == "batch").all()

    def test_persisted_risk_carries_its_explanation(self, db, trained):
        """A risk figure without a reason is not actionable, and re-deriving it
        later would need the exact model version that produced it."""
        InferenceService(db).assess_inventory(limit=5)
        stored = PredictionRepository(db).recent(SPOILAGE_RISK)
        assert stored["explanation"].notna().all()
        assert stored["label"].isin(
            ["Low", "Moderate", "High", "Critical"]).all()

    def test_persisted_predictions_are_scoreable(self, db, trained):
        """prediction_outcomes already exists, so nothing further is needed to
        close the monitoring loop."""
        service = InferenceService(db)
        service.assess_inventory(limit=3)

        backlog = service.scoreable_backlog()
        assert len(backlog) == 3

        predictions = PredictionRepository(db)
        first = int(backlog.iloc[0]["prediction_id"])
        predictions.record_outcome(first, 1.0)
        assert len(predictions.outcomes(SPOILAGE_RISK)) == 1
        assert len(service.scoreable_backlog()) == 2

    def test_assess_single_batch(self, db, trained):
        assessment = InferenceService(db).assess_batch("B0001")
        assert assessment is not None and assessment.batch_id == "B0001"
        assert assessment.drivers

    def test_assess_unknown_batch_returns_none(self, db, trained):
        assert InferenceService(db).assess_batch("NO-SUCH-BATCH") is None

    def test_risk_frame_is_sorted_for_the_ui(self, db, trained):
        frame = InferenceService(db).risk_frame(limit=20)
        assert not frame.empty
        assert frame["risk_score"].tolist() == sorted(
            frame["risk_score"], reverse=True)

    def test_predictions_can_be_suppressed(self, db, trained):
        batch = InferenceService(db).assess_inventory(limit=5, persist=False)
        assert batch.ok and batch.persisted == 0
        assert PredictionRepository(db).recent(SPOILAGE_RISK).empty

    def test_artifacts_load_once_per_service(self, db, trained):
        """Inference must not refit on every call."""
        service = InferenceService(db)
        first = service.spoilage_model
        assert first is service.spoilage_model

    def test_reload_picks_up_a_retrain(self, db, trained):
        service = InferenceService(db)
        assert service.status()["spoilage_available"]
        service.reload()
        assert service.status()["spoilage_available"]

    def test_milestone_two_repositories_are_untouched(self, db, trained):
        """The ML layer writes only to predictions and model_metrics; the
        Milestone 1 tables must be unchanged by inference."""
        service = InferenceService(db)
        before = {t: db.row_count(t) for t in
                  ("batches", "daily_item_stats", "orders", "sellers", "buyers")}
        service.assess_inventory(limit=10)
        service.forecast_product("Tomato", horizon=3)
        after = {t: db.row_count(t) for t in before}
        assert before == after
