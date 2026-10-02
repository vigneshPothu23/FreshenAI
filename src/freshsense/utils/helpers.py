"""
Common helper functions.
"""

from __future__ import annotations

from datetime import datetime


def current_timestamp() -> str:
    """Return current timestamp."""
    return datetime.now().isoformat(timespec="seconds")


def safe_divide(a: float, b: float) -> float:
    """Avoid division-by-zero."""
    return 0.0 if b == 0 else a / b