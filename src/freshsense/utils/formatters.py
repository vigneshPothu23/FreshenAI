"""
Formatting helpers.
"""

from __future__ import annotations


def currency(value: float) -> str:
    return f"₹{value:,.2f}"


def percentage(value: float) -> str:
    return f"{value:.2f}%"