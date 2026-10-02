"""
Project path management.

Provides a single source of truth for all filesystem locations used by
FreshSense AI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """Centralised project paths."""

    root: Path = Path(__file__).resolve().parents[2]

    @property
    def config_dir(self) -> Path:
        return self.root / "config"

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"

    @property
    def documents_dir(self) -> Path:
        return self.data_dir / "documents"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def vectorstore_dir(self) -> Path:
        return self.root / "data" / "vectorstore"

    @property
    def database(self) -> Path:
        return self.root / "freshsense.db"

    @property
    def schema_file(self) -> Path:
        return self.root / "database" / "schema.sql"

    def relative(self, path: Path) -> str:
        """Return project-relative path."""
        try:
            return str(path.relative_to(self.root))
        except ValueError:
            return str(path)


PATHS = ProjectPaths()

__all__ = ["PATHS", "ProjectPaths"]