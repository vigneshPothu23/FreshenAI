"""Sprint 4 — the single LLM egress point.

**No other module in FreshSense AI may call a language-model provider.** Every
RAG component and every agent goes through :func:`ask_llm`. That constraint is
what makes provider migration a one-file change.

Three providers sit behind one interface:

``StubProvider``
    Deterministic, offline, template-based composition over supplied context.
    **This is the default, not a fallback.** The application is fully functional
    with no credentials, no network and no MaaS endpoint — which matters when
    the endpoint specification is not yet published.

``OpenAICompatProvider``
    Any endpoint speaking the OpenAI ``/chat/completions`` schema: OpenAI
    itself, Ollama, vLLM, LM Studio, or an internal gateway.

``MaaSProvider``
    TCS Model-as-a-Service. Subclasses the OpenAI-compatible provider because
    the schema is expected to match; if it does not, only ``_build_payload`` and
    ``_parse_response`` need overriding.

Provider selection is automatic: a configured remote endpoint is preferred, and
the stub takes over whenever the remote is unreachable, times out, or returns an
error. A request never raises purely because an LLM was unavailable.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

from freshsense.config import SETTINGS
from freshsense.logging_config import get_logger

LOG = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════
@dataclass
class LLMResponse:
    """Uniform response envelope regardless of which provider served the call."""

    text: str
    provider: str
    model: str = ""
    latency_ms: int = 0
    success: bool = True
    error: str = ""
    prompt_chars: int = 0
    completion_chars: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_stub(self) -> bool:
        return self.provider == "stub"

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text, "provider": self.provider, "model": self.model,
            "latency_ms": self.latency_ms, "success": self.success,
            "error": self.error, "prompt_chars": self.prompt_chars,
            "completion_chars": self.completion_chars,
        }


# ══════════════════════════════════════════════════════════════════════════
class LLMProvider(ABC):
    """Interface every language-model backend must satisfy."""

    name: str = "abstract"

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        context: Sequence[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Produce a completion for ``prompt``."""

    @abstractmethod
    def is_available(self) -> bool:
        """Whether this provider can currently service a request."""

    def describe(self) -> dict[str, Any]:
        return {"provider": self.name, "available": self.is_available()}


# ══════════════════════════════════════════════════════════════════════════
class StubProvider(LLMProvider):
    """Deterministic offline provider.

    Composes an answer extractively from the retrieved context rather than
    generating text. The retrieval, grounding, refusal and logging paths are
    identical to the remote providers — only the final composition step differs.
    Consequently every screen in the application remains demonstrable with no
    credentials configured.
    """

    name = "stub"

    _STOPWORDS = frozenset({
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "what", "which", "who", "whom", "how", "when", "where", "why", "do",
        "does", "did", "can", "could", "should", "would", "will", "shall",
        "for", "of", "in", "to", "on", "at", "by", "with", "from", "and",
        "or", "but", "if", "then", "than", "that", "this", "these", "those",
        "it", "its", "my", "our", "your", "their", "i", "we", "you", "they",
        "me", "us", "them", "as", "so", "not", "no", "yes", "please", "tell",
        "give", "show", "about", "into", "over", "under", "any", "all",
    })

    def is_available(self) -> bool:
        return True

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        context: Sequence[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        started = time.perf_counter()
        question = self._extract_question(prompt)

        if context:
            text = self._compose_from_context(question, context)
        else:
            text = self._compose_from_prompt(prompt)

        latency = int((time.perf_counter() - started) * 1000)
        return LLMResponse(
            text=text, provider=self.name, model="deterministic-extractive",
            latency_ms=latency, success=True,
            prompt_chars=len(prompt), completion_chars=len(text),
            metadata={"mode": "extractive"},
        )

    @staticmethod
    def _extract_question(prompt: str) -> str:
        """Recover the user question from an assembled prompt."""
        for marker in ("QUESTION:", "REQUEST:", "USER:"):
            if marker in prompt:
                return prompt.split(marker)[-1].strip()
        return prompt.strip()

    def _keywords(self, text: str) -> set[str]:
        words = re.findall(r"[a-z0-9]+", text.lower())
        return {w for w in words if w not in self._STOPWORDS and len(w) > 2}

    def _compose_from_context(
        self, question: str, context: Sequence[dict[str, Any]]
    ) -> str:
        """Select and order the sentences that best answer the question."""
        keywords = self._keywords(question)
        scored: list[tuple[float, str, str]] = []

        for chunk in context:
            source = str(chunk.get("source", chunk.get("title", "knowledge base")))
            for sentence in re.split(r"(?<=[.!?])\s+", str(chunk.get("text", ""))):
                sentence = sentence.strip()
                if len(sentence) < 25:
                    continue
                overlap = len(keywords & self._keywords(sentence))
                if overlap:
                    # Normalise by length so a long sentence does not win purely
                    # by containing more words.
                    score = overlap / (1 + len(sentence) / 220)
                    scored.append((score, sentence, source))

        if not scored:
            first = str(context[0].get("text", "")).strip()
            return first[:600] if first else (
                "The knowledge base does not contain information on that topic."
            )

        scored.sort(key=lambda item: -item[0])
        selected: list[str] = []
        seen: set[str] = set()
        for _, sentence, _ in scored:
            normalised = sentence.lower()
            if normalised in seen:
                continue
            seen.add(normalised)
            selected.append(sentence)
            if len(selected) >= 3:
                break

        answer = " ".join(selected)
        sources = sorted({str(c.get("source", "")) for c in context if c.get("source")})
        if sources:
            answer += f"\n\nBased on: {', '.join(sources[:3])}."
        return answer

    @staticmethod
    def _compose_from_prompt(prompt: str) -> str:
        """Fallback used when an agent calls the LLM with no retrieved context."""
        return (
            "Language generation is running in offline mode, so this response is "
            "composed deterministically from the structured data above rather "
            "than by a language model. Configure FRESHSENSE_LLM_BASE_URL and "
            "FRESHSENSE_LLM_API_KEY (or the MaaS equivalents) to enable "
            "natural-language generation. Every analytical result shown in the "
            "application is unaffected by this setting."
        )


# ══════════════════════════════════════════════════════════════════════════
class OpenAICompatProvider(LLMProvider):
    """Any endpoint speaking the OpenAI ``/chat/completions`` schema.

    Implemented with :mod:`urllib` rather than the ``openai`` SDK so the wrapper
    has no hard third-party dependency and works against gateways that deviate
    slightly from the official client's expectations.
    """

    name = "openai_compatible"

    def __init__(
        self,
        *,
        base_url: str = "",
        api_key: str = "",
        model: str = "",
        timeout: int = 45,
    ) -> None:
        settings = SETTINGS.llm
        self.base_url = (base_url or settings.base_url).rstrip("/")
        self.api_key = api_key or settings.api_key
        self.model = model or settings.model
        self.timeout = timeout or settings.timeout

    @property
    def endpoint(self) -> str:
        if not self.base_url:
            return ""
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        if self.base_url.endswith("/v1"):
            return f"{self.base_url}/chat/completions"
        return f"{self.base_url}/v1/chat/completions"

    def is_available(self) -> bool:
        return bool(self.base_url and self.api_key)

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _build_payload(
        self, prompt: str, system: str | None, max_tokens: int, temperature: float
    ) -> dict[str, Any]:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

    @staticmethod
    def _parse_response(payload: dict[str, Any]) -> str:
        choices = payload.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        content = message.get("content", "")
        # Some gateways return content as a list of typed parts.
        if isinstance(content, list):
            return "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        return str(content)

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        context: Sequence[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        settings = SETTINGS.llm
        max_tokens = int(max_tokens or settings.max_tokens)
        temperature = float(temperature if temperature is not None else settings.temperature)

        started = time.perf_counter()
        body = json.dumps(
            self._build_payload(prompt, system, max_tokens, temperature)
        ).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint, data=body, headers=self._headers(), method="POST"
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            text = self._parse_response(payload)
            latency = int((time.perf_counter() - started) * 1000)
            return LLMResponse(
                text=text, provider=self.name, model=self.model,
                latency_ms=latency, success=bool(text),
                prompt_chars=len(prompt), completion_chars=len(text),
                metadata={"usage": payload.get("usage", {})},
            )
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
                json.JSONDecodeError, OSError) as exc:
            latency = int((time.perf_counter() - started) * 1000)
            LOG.warning("%s call failed (%s) — falling back to stub", self.name, exc)
            return LLMResponse(
                text="", provider=self.name, model=self.model,
                latency_ms=latency, success=False, error=str(exc)[:300],
                prompt_chars=len(prompt),
            )


# ══════════════════════════════════════════════════════════════════════════
class MaaSProvider(OpenAICompatProvider):
    """TCS Model-as-a-Service.

    Assumed OpenAI-schema compatible. If the published specification differs,
    override :meth:`_build_payload`, :meth:`_parse_response` and
    :meth:`_headers` here — nothing outside this class needs to change.
    """

    name = "maas"

    def _headers(self) -> dict[str, str]:
        headers = super()._headers()
        headers["api-key"] = self.api_key      # some gateways expect this form
        return headers


# ══════════════════════════════════════════════════════════════════════════
class LLMWrapper:
    """Provider selection, fallback and call accounting."""

    def __init__(self) -> None:
        self.settings = SETTINGS.llm
        self.stub = StubProvider()
        self.remote: LLMProvider | None = self._build_remote()
        self._calls: list[dict[str, Any]] = []

    def _build_remote(self) -> LLMProvider | None:
        provider = (self.settings.provider or "auto").lower()
        if provider == "stub":
            return None
        if provider == "maas" or (provider == "auto" and self.settings.base_url):
            candidate: LLMProvider = (
                MaaSProvider() if provider == "maas" else OpenAICompatProvider()
            )
            return candidate if candidate.is_available() else None
        if provider == "openai_compatible":
            candidate = OpenAICompatProvider()
            return candidate if candidate.is_available() else None
        return None

    # ── introspection ─────────────────────────────────────────────────
    @property
    def active_provider(self) -> str:
        return self.remote.name if self.remote else self.stub.name

    def status(self) -> dict[str, Any]:
        """Rendered on the Settings page and the health endpoint."""
        return {
            "active_provider": self.active_provider,
            "remote_configured": self.remote is not None,
            "remote_endpoint": getattr(self.remote, "endpoint", "") if self.remote else "",
            "model": getattr(self.remote, "model", "deterministic-extractive")
            if self.remote else "deterministic-extractive",
            "offline_capable": True,
            "calls_made": len(self._calls),
            "note": (
                "A remote provider is configured; the deterministic stub remains "
                "available as an automatic fallback."
                if self.remote else
                "Running fully offline. Every feature works; language generation "
                "is composed deterministically from retrieved context."
            ),
        }

    def call_log(self) -> list[dict[str, Any]]:
        return list(self._calls)

    # ── generation ────────────────────────────────────────────────────
    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        context: Sequence[dict[str, Any]] | None = None,
        allow_fallback: bool = True,
    ) -> LLMResponse:
        """Generate a completion, falling back to the stub on any remote failure."""
        if self.remote is not None:
            response = self.remote.generate(
                prompt, system=system, max_tokens=max_tokens,
                temperature=temperature, context=context,
            )
            self._record(response)
            if response.success and response.text.strip():
                return response
            if not allow_fallback:
                return response
            LOG.info("Remote provider returned no usable text; using stub")

        response = self.stub.generate(
            prompt, system=system, max_tokens=max_tokens,
            temperature=temperature, context=context,
        )
        self._record(response)
        return response

    def _record(self, response: LLMResponse) -> None:
        self._calls.append(response.as_dict())
        try:
            from freshsense.db.repository import MonitoringRepository
            MonitoringRepository().log_llm_call(
                provider=response.provider, model=response.model,
                prompt_chars=response.prompt_chars,
                completion_chars=response.completion_chars,
                latency_ms=response.latency_ms,
                status="ok" if response.success else "error",
                error=response.error,
            )
        except Exception:                                # pragma: no cover
            # Observability must never break generation.
            pass


_WRAPPER: LLMWrapper | None = None


def get_llm() -> LLMWrapper:
    """Return the process-wide LLM wrapper."""
    global _WRAPPER
    if _WRAPPER is None:
        _WRAPPER = LLMWrapper()
        LOG.info("LLM wrapper initialised — active provider: %s",
                 _WRAPPER.active_provider)
    return _WRAPPER


def reset_llm() -> LLMWrapper:
    """Rebuild the wrapper after a configuration change (Settings page)."""
    global _WRAPPER
    SETTINGS.__class__  # settings are immutable; caller reloads the process env
    _WRAPPER = LLMWrapper()
    return _WRAPPER


def ask_llm(
    prompt: str,
    *,
    system: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    context: Sequence[dict[str, Any]] | None = None,
) -> LLMResponse:
    """The single entry point for language generation across the whole codebase."""
    return get_llm().generate(
        prompt, system=system, max_tokens=max_tokens,
        temperature=temperature, context=context,
    )


__all__ = [
    "LLMResponse", "LLMProvider", "StubProvider", "OpenAICompatProvider",
    "MaaSProvider", "LLMWrapper", "get_llm", "reset_llm", "ask_llm",
]