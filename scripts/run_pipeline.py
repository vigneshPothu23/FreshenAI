"""
Sprint 1 Pipeline Runner.

Loads raw CSV files, executes the complete data pipeline,
and persists the processed data into the normalized SQLite database.
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from freshsense.data.loaders import RawDataLoader
from freshsense.data.pipeline import DataPipeline
from freshsense.db.database import Database
from freshsense.db.loader import DatabaseLoader
from freshsense.logging_config import get_logger

LOG = get_logger(__name__)


def main() -> None:
    LOG.info("=" * 80)
    LOG.info("Starting FreshSense AI pipeline...")
    LOG.info("=" * 80)

    # ------------------------------------------------------------------
    # Execute Sprint-1 data pipeline
    # ------------------------------------------------------------------

    loader = RawDataLoader()

    pipeline = DataPipeline(loader=loader)

    processed = pipeline.run(
        persist=True,
        strict=False,
    )

    LOG.info("Pipeline completed successfully.")

    # ------------------------------------------------------------------
    # Initialise database
    # ------------------------------------------------------------------

    db = Database()
    db.initialise()

    LOG.info("Database initialized.")

    # ------------------------------------------------------------------
    # Load processed data into normalized schema
    # ------------------------------------------------------------------

    database_loader = DatabaseLoader(db)

    report = database_loader.load(
        processed.tables,
        mode="replace",
    )

    LOG.info("=" * 80)
    LOG.info("DATABASE LOAD SUMMARY")
    LOG.info("=" * 80)

    print()
    print(report.summary())

    print("\n")
    print("=" * 80)
    print("ROW COUNTS")
    print("=" * 80)
    print(database_loader.row_counts())

    print("\n")
    print("=" * 80)
    print("REFERENTIAL INTEGRITY")
    print("=" * 80)
    print(database_loader.integrity_check())

    LOG.info("SQLite database populated successfully.")
    LOG.info("FreshSense AI setup completed successfully.")


if __name__ == "__main__":
    main()