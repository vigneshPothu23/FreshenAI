"""Typed configuration loader.

``config/application.yaml`` supplies domain constants; ``.env`` supplies secrets
and environment-specific overrides. The two are merged into a single immutable
:class:`Settings` object exposed as the module-level ``SETTINGS`` singleton.

Every threshold, weight, bound and seed used anywhere in FreshSense AI is read
through this object. Nothing else parses YAML, and nothing hardcodes a value
that belongs in the configuration file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml

from freshsense.paths import PATHS

#: Filenames searched for, in order, when ``PATHS.config_file`` does not exist.
#: The project has used both names; accepting either keeps the loader working
#: regardless of which one a given checkout carries.
_CONFIG_CANDIDATES: tuple[str, ...] = ("application.yaml", "config.yaml",
                                       "application.yml", "config.yml")


def _resolve_config_path() -> Path:
    """Locate the configuration file.

    Prefers ``PATHS.config_file`` so path resolution stays in one place, and
    falls back to a search of the config directory only when that file is
    absent. Raising here with the paths tried is far more useful than a
    ``FileNotFoundError`` naming a single candidate.

    Returns:
        Path to an existing configuration file.

    Raises:
        FileNotFoundError: If no candidate exists.
    """
    configured = getattr(PATHS, "config_file", None)
    if configured is not None and Path(configured).is_file():
        return Path(configured)

    directory = getattr(PATHS, "config_dir", None)
    if directory is None:
        directory = Path(configured).parent if configured is not None \
            else Path(PATHS.root) / "config"

    for name in _CONFIG_CANDIDATES:
        candidate = Path(directory) / name
        if candidate.is_file():
            return candidate

    tried = ", ".join(str(Path(directory) / n) for n in _CONFIG_CANDIDATES)
    raise FileNotFoundError(
        f"No FreshSense configuration file found. Tried: "
        f"{configured if configured else '(no PATHS.config_file)'}, {tried}."
    )


def _load_dotenv(path: Path) -> None:
    """Minimal ``.env`` reader.

    Implemented locally rather than depending on ``python-dotenv`` so the
    package remains importable in a bare interpreter. Existing environment
    variables always win, which is the standard precedence rule: a value already
    exported by the shell or the container must not be silently overwritten by a
    file checked into the repository.
    """
    if not path or not Path(path).is_file():
        return
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv(getattr(PATHS, "env_file", Path(PATHS.root) / ".env"))


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def _env_float(key: str, default: float) -> float:
    try:
        return float(_env(key) or default)
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    try:
        return int(float(_env(key) or default))
    except ValueError:
        return default


@dataclass(frozen=True)
class LLMSettings:
    """Configuration for the single LLM egress point (``rag.llm_wrapper``)."""

    provider: str = "auto"
    base_url: str = ""
    api_key: str = ""
    model: str = "gpt-4o-mini"
    timeout: int = 45
    max_tokens: int = 800
    temperature: float = 0.2

    @property
    def is_configured(self) -> bool:
        """True when a remote endpoint has enough detail to be attempted."""
        return bool(self.base_url and self.api_key) or bool(
            self.api_key and self.provider == "openai_compatible"
        )


@dataclass(frozen=True)
class Settings:
    """Immutable application settings."""

    raw: Mapping[str, Any] = field(repr=False)
    llm: LLMSettings = field(default_factory=LLMSettings)
    environment: str = "development"
    log_level: str = "INFO"
    embedding_backend: str = "auto"
    embedding_model: str = "all-MiniLM-L6-v2"
    vectorstore_backend: str = "auto"

    # ── generic accessors ─────────────────────────────────────────────
    def section(self, name: str) -> dict[str, Any]:
        """Return a top-level configuration section as a plain dictionary.

        A missing section yields ``{}`` rather than raising, so a caller reading
        an optional section with ``.get(key, default)`` behaves identically
        whether the section is absent or merely incomplete.
        """
        value = self.raw.get(name, {})
        return dict(value) if isinstance(value, Mapping) else {}

    def get(self, dotted: str, default: Any = None) -> Any:
        """Fetch a nested value using dotted notation, e.g. ``"pricing.max_discount"``."""
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return node

    # ── section properties ────────────────────────────────────────────
    @property
    def app(self) -> dict[str, Any]:
        return self.section("application")

    @property
    def application(self) -> dict[str, Any]:
        """Alias of :attr:`app`, matching the section name in the YAML file."""
        return self.section("application")

    @property
    def database(self) -> dict[str, Any]:
        return self.section("database")

    @property
    def data(self) -> dict[str, Any]:
        return self.section("data")

    @property
    def cleaning(self) -> dict[str, Any]:
        return self.section("cleaning")

    @property
    def features(self) -> dict[str, Any]:
        return self.section("features")

    @property
    def pricing(self) -> dict[str, Any]:
        return self.section("pricing")

    @property
    def grading(self) -> dict[str, Any]:
        return self.section("grading")

    @property
    def forecasting(self) -> dict[str, Any]:
        return self.section("forecasting")

    @property
    def spoilage(self) -> dict[str, Any]:
        return self.section("spoilage")

    @property
    def recommendation(self) -> dict[str, Any]:
        return self.section("recommendation")

    @property
    def rag(self) -> dict[str, Any]:
        return self.section("rag")

    @property
    def agents(self) -> dict[str, Any]:
        return self.section("agents")

    @property
    def monitoring(self) -> dict[str, Any]:
        return self.section("monitoring")

    @property
    def ui(self) -> dict[str, Any]:
        return self.section("ui")

    # ── derived values ────────────────────────────────────────────────
    @property
    def currency(self) -> str:
        """Currency symbol used in every rendered figure."""
        return str(self.app.get("currency_symbol", "\u20b9"))

    @property
    def seed(self) -> int:
        """Reproducibility seed, read from ``cleaning.random_seed``.

        The models reference ``SETTINGS.seed`` when splitting train and test and
        when seeding the estimators. Exposing it as a property rather than
        adding a new configuration key keeps a single source: the value still
        lives in ``application.yaml`` under ``cleaning.random_seed``, and no
        module hardcodes it.

        Without this, ``SpoilageModel.fit`` raised ``AttributeError`` before it
        ever reached ``train_test_split``, so the classifier could not be
        trained at all.
        """
        return int(self.cleaning.get("random_seed", 42))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and cache application settings.

    Cached because Streamlit re-executes the whole script on every widget
    interaction; re-parsing YAML on each rerun would be pure waste.
    """
    with _resolve_config_path().open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    # MaaS credentials take precedence when present: the lab endpoint is the
    # preferred provider once it exists.
    maas_url, maas_key = _env("MAAS_BASE_URL"), _env("MAAS_API_KEY")
    if maas_url and maas_key:
        base_url, api_key = maas_url, maas_key
        model = _env("MAAS_MODEL") or _env("FRESHSENSE_LLM_MODEL", "gpt-4o-mini")
        provider = "maas"
    else:
        base_url = _env("FRESHSENSE_LLM_BASE_URL")
        api_key = _env("FRESHSENSE_LLM_API_KEY")
        model = _env("FRESHSENSE_LLM_MODEL", "gpt-4o-mini")
        provider = _env("FRESHSENSE_LLM_PROVIDER", "auto")

    llm = LLMSettings(
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout=_env_int("FRESHSENSE_LLM_TIMEOUT", 45),
        max_tokens=_env_int("FRESHSENSE_LLM_MAX_TOKENS", 800),
        temperature=_env_float("FRESHSENSE_LLM_TEMPERATURE", 0.2),
    )

    return Settings(
        raw=raw,
        llm=llm,
        environment=_env("FRESHSENSE_ENV", "development"),
        log_level=_env("FRESHSENSE_LOG_LEVEL", "INFO").upper(),
        embedding_backend=_env("FRESHSENSE_EMBEDDING_BACKEND", "auto"),
        embedding_model=_env("FRESHSENSE_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        vectorstore_backend=_env("FRESHSENSE_VECTORSTORE", "auto"),
    )


SETTINGS = get_settings()

__all__ = ["SETTINGS", "Settings", "LLMSettings", "get_settings"]