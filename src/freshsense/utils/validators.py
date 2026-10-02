"""
Validation helpers.
"""

from __future__ import annotations


def is_positive(value: float) -> bool:
    return value >= 0


def within_range(value: float, minimum: float, maximum: float) -> bool:
    return minimum <= value <= maximum