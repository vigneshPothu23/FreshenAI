"""
Sprint 2 Model Training.
"""

from __future__ import annotations

from freshsense.models.demand_forecasting import DemandForecaster
from freshsense.models.spoilage_prediction import SpoilagePredictor
from freshsense.logging_config import get_logger

LOG = get_logger(__name__)


def main():

    LOG.info("Training Demand Forecasting model...")

    DemandForecaster().train()

    LOG.info("Training Spoilage Prediction model...")

    SpoilagePredictor().train()

    LOG.info("All models trained successfully.")


if __name__ == "__main__":
    main()