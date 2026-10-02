"""
Logging configuration for FreshSense AI.
"""

from __future__ import annotations

import logging
from pathlib import Path

from freshsense.paths import PATHS

LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)


def configure_logging(level: int = logging.INFO) -> None:
    """
    Configure application logging.
    """

    PATHS.logs_dir.mkdir(parents=True, exist_ok=True)

    log_file = PATHS.logs_dir / "freshsense.log"

    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger for the given module.
    """
    return logging.getLogger(name)


configure_logging()

__all__ = [
    "configure_logging",
    "get_logger",
]