"""Model registry.

Artefacts are trained offline by ``scripts/train_models.py`` and persisted to
``models/``. The Streamlit application loads them once at process start and
never trains at request time.

Every artefact is stored with its metrics, feature list, training timestamp and
version so that a prediction row can be traced back to the exact model that
produced it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib

from freshsense.exceptions import ModelNotTrainedError
from freshsense.logging_config import get_logger
from freshsense.paths import PATHS

LOG = get_logger(__name__)


@dataclass
class ModelArtifact:
    """A trained model plus everything needed to interpret its output."""

    name: str
    model: Any
    version: str = "v1.0"
    algorithm: str = ""
    features: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    trained_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def summary(self) -> dict[str, Any]:
        """Compact description used by the Monitoring and Settings pages."""
        return {
            "name": self.name,
            "version": self.version,
            "algorithm": self.algorithm,
            "n_features": len(self.features),
            "trained_at": self.trained_at,
            **{k: v for k, v in self.metrics.items() if isinstance(v, (int, float))},
        }


class ModelRegistry:
    """Filesystem-backed store for trained artefacts."""

    def __init__(self, models_dir: Path | None = None) -> None:
        self.models_dir = models_dir or PATHS.models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, ModelArtifact] = {}

    # ── paths ─────────────────────────────────────────────────────────
    def artifact_path(self, name: str) -> Path:
        return self.models_dir / f"{name}.pkl"

    def metadata_path(self, name: str) -> Path:
        return self.models_dir / f"{name}_metadata.json"

    # ── persistence ───────────────────────────────────────────────────
    def save(self, artifact: ModelArtifact) -> Path:
        """Persist an artefact and a human-readable metadata sidecar."""
        path = self.artifact_path(artifact.name)
        joblib.dump(
            {
                "model": artifact.model,
                "version": artifact.version,
                "algorithm": artifact.algorithm,
                "features": artifact.features,
                "metrics": artifact.metrics,
                "metadata": artifact.metadata,
                "trained_at": artifact.trained_at,
            },
            path,
            compress=3,
        )
        self.metadata_path(artifact.name).write_text(
            json.dumps(artifact.summary(), indent=2), encoding="utf-8"
        )
        self._cache[artifact.name] = artifact
        LOG.info("Saved model '%s' (%s) -> %s",
                 artifact.name, artifact.algorithm, PATHS.relative(path))
        return path

    def load(self, name: str, *, required: bool = False) -> ModelArtifact | None:
        """Load an artefact, returning ``None`` when absent unless ``required``."""
        if name in self._cache:
            return self._cache[name]

        path = self.artifact_path(name)
        if not path.is_file():
            if required:
                raise ModelNotTrainedError(
                    f"Model '{name}' has not been trained. "
                    f"Run `python scripts/train_models.py` first."
                )
            LOG.warning("Model artefact '%s' not found at %s",
                        name, PATHS.relative(path))
            return None

        try:
            blob = joblib.load(path)
        except Exception as exc:                         # pragma: no cover
            LOG.error("Failed to load model '%s': %s", name, exc)
            if required:
                raise ModelNotTrainedError(f"Model '{name}' is corrupt: {exc}") from exc
            return None

        artifact = ModelArtifact(
            name=name,
            model=blob["model"],
            version=blob.get("version", "v1.0"),
            algorithm=blob.get("algorithm", ""),
            features=blob.get("features", []),
            metrics=blob.get("metrics", {}),
            metadata=blob.get("metadata", {}),
            trained_at=blob.get("trained_at", ""),
        )
        self._cache[name] = artifact
        LOG.info("Loaded model '%s' (%s, trained %s)",
                 name, artifact.algorithm, artifact.trained_at)
        return artifact

    def exists(self, name: str) -> bool:
        return self.artifact_path(name).is_file()

    def delete(self, name: str) -> None:
        for path in (self.artifact_path(name), self.metadata_path(name)):
            if path.exists():
                path.unlink()
        self._cache.pop(name, None)
        LOG.warning("Deleted model artefact '%s'", name)

    def list_models(self) -> list[dict[str, Any]]:
        """Summaries of every artefact on disk — powers the Settings page."""
        summaries: list[dict[str, Any]] = []
        for path in sorted(self.models_dir.glob("*_metadata.json")):
            try:
                summaries.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        return summaries

    def clear_cache(self) -> None:
        self._cache.clear()


_REGISTRY: ModelRegistry | None = None


def get_registry() -> ModelRegistry:
    """Return the process-wide :class:`ModelRegistry` singleton."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = ModelRegistry()
    return _REGISTRY


__all__ = ["ModelArtifact", "ModelRegistry", "get_registry"]