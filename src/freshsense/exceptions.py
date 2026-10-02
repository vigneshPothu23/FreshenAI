"""
Custom exceptions used throughout FreshSense AI.
"""


class FreshSenseError(Exception):
    """Base exception for the project."""


# ─────────────────────────────────────────────────────────────
# Data Exceptions
# ─────────────────────────────────────────────────────────────

class DataNotFoundError(FreshSenseError):
    """Raised when required data files are missing."""


class DataValidationError(FreshSenseError):
    """Raised when data validation fails."""


class DataCleaningError(FreshSenseError):
    """Raised when cleaning pipeline fails."""


# ─────────────────────────────────────────────────────────────
# Database Exceptions
# ─────────────────────────────────────────────────────────────

class DatabaseError(FreshSenseError):
    """Raised for SQLite/database related errors."""


# ─────────────────────────────────────────────────────────────
# Model Exceptions
# ─────────────────────────────────────────────────────────────

class ModelError(FreshSenseError):
    """Base model exception."""


class ModelNotTrainedError(ModelError):
    """Raised when prediction is attempted before training."""


class ModelLoadError(ModelError):
    """Raised when a saved model cannot be loaded."""


# ─────────────────────────────────────────────────────────────
# RAG Exceptions
# ─────────────────────────────────────────────────────────────

class RetrievalError(FreshSenseError):
    """Raised during retrieval failures."""


class VectorStoreError(FreshSenseError):
    """Raised when vector database operations fail."""


class LLMError(FreshSenseError):
    """Raised when LLM generation fails."""


# ─────────────────────────────────────────────────────────────
# Recommendation Exceptions
# ─────────────────────────────────────────────────────────────

class RecommendationError(FreshSenseError):
    """Raised by recommendation engine."""


__all__ = [
    "FreshSenseError",
    "DataNotFoundError",
    "DataValidationError",
    "DataCleaningError",
    "DatabaseError",
    "ModelError",
    "ModelNotTrainedError",
    "ModelLoadError",
    "RetrievalError",
    "VectorStoreError",
    "LLMError",
    "RecommendationError",
]