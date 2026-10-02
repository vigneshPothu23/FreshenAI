"""Sprint 5 — the agent contract.

Two types every specialist agent depends on:

``AgentResult``
    The uniform envelope returned by every action. It separates four things that
    are routinely conflated in agent frameworks:

    * ``narrative`` — one sentence a human reads. This is what reaches the final
      answer, so it must contain the actual numbers rather than a description of
      the work performed.
    * ``facts`` — small values merged into the shared state for later agents to
      branch on.
    * ``artifacts`` — heavy payloads for the UI that must never enter a prompt.
    * ``data`` — the agent's own structured return value, consumed by the
      coordinator (the planner returns its task list here).

``BaseAgent``
    Dispatch, timing and error containment, so a specialist only writes the
    domain method. Actions map to methods named ``action_<name>``; declaring an
    action without implementing it fails loudly at construction rather than
    silently at runtime.

An agent is a **thin adapter**. It translates a task into a call against an
already-built service — a repository, a model, the recommendation engine, the
RAG chain — and translates the return value into an ``AgentResult``. Business
logic living in an agent is a design error: it would be unreachable from the UI
and untestable without an orchestration run.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from freshsense.agents.state import AgentState, AgentTask
from freshsense.logging_config import get_logger

LOG = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════
@dataclass
class AgentResult:
    """Uniform return envelope for every agent action."""

    agent: str
    action: str
    success: bool = True
    narrative: str = ""
    facts: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    tool_called: str = ""
    latency_ms: int = 0

    @property
    def is_empty(self) -> bool:
        """Whether the action succeeded but found nothing to report.

        The coordinator treats this the same as a failure — both trigger
        replanning — because a search that returns zero rows is exactly the
        situation where a different approach is needed. Distinguishing "broke"
        from "found nothing" still matters for the trace, so they stay separate
        fields.
        """
        if not self.success:
            return True
        return not (self.narrative or self.facts or self.artifacts or self.data)

    @property
    def status(self) -> str:
        """Trace status string, matching the ``agent_steps.status`` vocabulary."""
        if not self.success:
            return "error"
        return "empty" if self.is_empty else "ok"

    def as_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "action": self.action,
            "success": self.success,
            "status": self.status,
            "narrative": self.narrative,
            "facts": self.facts,
            "artifacts": sorted(self.artifacts),
            "error": self.error,
            "tool_called": self.tool_called,
            "latency_ms": self.latency_ms,
        }

    def __str__(self) -> str:                            # pragma: no cover
        return (f"{self.agent}.{self.action} [{self.status}] "
                f"{self.narrative or self.error}")


# ══════════════════════════════════════════════════════════════════════════
class BaseAgent(ABC):
    """Base class for every specialist agent.

    A subclass declares its identity and its actions, then implements one
    ``action_<name>`` method per declared action::

        class InventoryAgent(BaseAgent):
            name = "inventory"
            description = "Queries live stock and surfaces batches at risk."
            supported_actions = ("find_at_risk", "search_inventory")

            def action_find_at_risk(self, state, task) -> AgentResult:
                ...
                return self.ok("12 batches at risk", facts={...})

    :meth:`execute` handles dispatch, timing and error containment. Subclasses
    never catch their own exceptions for the purpose of returning a failure
    envelope — raising is the correct behaviour, and the base class converts it.
    """

    #: Logical name, matching the key used in ``AGENT_REGISTRY``.
    name: str = "agent"

    #: One line describing the agent's remit, shown on the Agents page.
    description: str = ""

    #: Actions this agent accepts. Every entry must have a matching
    #: ``action_<name>`` method.
    supported_actions: tuple[str, ...] = ()

    def __init__(self) -> None:
        missing = [
            action for action in self.supported_actions
            if not callable(getattr(self, f"action_{action}", None))
        ]
        if missing:
            raise NotImplementedError(
                f"{type(self).__name__} declares action(s) {missing} in "
                f"supported_actions but implements no matching "
                f"action_<name> method."
            )
        self._log = get_logger(f"freshsense.agents.{self.name}")

    # ── dispatch ──────────────────────────────────────────────────────
    def handler_for(self, action: str) -> Callable[..., AgentResult] | None:
        """Resolve the method implementing ``action``, if it is supported."""
        if action not in self.supported_actions:
            return None
        return getattr(self, f"action_{action}", None)

    def execute(self, state: AgentState, task: AgentTask) -> AgentResult:
        """Run one task, returning a result envelope in every circumstance.

        Args:
            state: The shared blackboard for this run. Read prior facts from it;
                do not mutate it — return values in ``facts`` and ``artifacts``
                so the coordinator applies them at a single, traceable point.
            task: The work to perform.

        Returns:
            An :class:`AgentResult`. Exceptions raised by the action method are
            caught, logged and converted into a failed result, because one
            specialist raising must not abort the whole orchestration.
        """
        handler = self.handler_for(task.action)
        if handler is None:
            return self.fail(
                f"'{task.action}' is not an action this agent supports "
                f"(available: {', '.join(self.supported_actions) or 'none'})",
                action=task.action,
            )

        started = time.perf_counter()
        try:
            result = handler(state, task)
        except KeyError as exc:
            # Raised by AgentTask.require() when the planner omitted a parameter.
            result = self.fail(f"missing input — {exc}", action=task.action)
        except Exception as exc:
            self._log.warning("Action '%s' failed: %s", task.action, exc,
                              exc_info=self._log.isEnabledFor(10))
            result = self.fail(str(exc)[:300], action=task.action)

        if not isinstance(result, AgentResult):          # pragma: no cover
            result = self.fail(
                f"action_{task.action} returned {type(result).__name__}, "
                f"expected AgentResult",
                action=task.action,
            )

        result.agent = self.name
        result.action = task.action
        if not result.latency_ms:
            result.latency_ms = int((time.perf_counter() - started) * 1000)

        self._log.debug("%s.%s -> %s in %d ms",
                        self.name, task.action, result.status, result.latency_ms)
        return result

    # ── result constructors ───────────────────────────────────────────
    def ok(
        self,
        narrative: str,
        *,
        facts: dict[str, Any] | None = None,
        artifacts: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        tool_called: str = "",
        action: str = "",
    ) -> AgentResult:
        """A successful result.

        ``narrative`` reaches the user, so it should state the finding, not the
        activity: "12 batches worth ₹8,420 expire within 2 days", never
        "inventory search completed".
        """
        return AgentResult(
            agent=self.name, action=action, success=True,
            narrative=narrative, facts=facts or {}, artifacts=artifacts or {},
            data=data or {}, tool_called=tool_called,
        )

    def empty(
        self,
        narrative: str = "",
        *,
        tool_called: str = "",
        action: str = "",
    ) -> AgentResult:
        """The action ran correctly but found nothing.

        Returned rather than ``ok`` so the coordinator can replan. The narrative
        is deliberately discarded from the envelope's truthiness check; supply
        one anyway, because it explains the gap in the trace.
        """
        result = AgentResult(
            agent=self.name, action=action, success=True,
            tool_called=tool_called,
        )
        result.error = narrative or "no results"
        return result

    def fail(
        self,
        error: str,
        *,
        tool_called: str = "",
        action: str = "",
    ) -> AgentResult:
        """The action could not complete."""
        return AgentResult(
            agent=self.name, action=action, success=False,
            error=error, tool_called=tool_called,
        )

    # ── introspection ─────────────────────────────────────────────────
    def describe(self) -> dict[str, Any]:
        """Roster entry for the Agents page."""
        return {
            "name": self.name,
            "class": type(self).__name__,
            "description": self.description,
            "actions": list(self.supported_actions),
        }

    def __repr__(self) -> str:                           # pragma: no cover
        return (f"<{type(self).__name__} name={self.name!r} "
                f"actions={len(self.supported_actions)}>")


# ══════════════════════════════════════════════════════════════════════════
class ToolAgent(BaseAgent, ABC):
    """Convenience base for agents that wrap exactly one service object.

    The service is constructed lazily on first use rather than in ``__init__``.
    That matters because the coordinator builds every agent at import time,
    while services such as the forecast bundle or the vector index are expensive
    and may not exist yet — an agent that loads a 20 MB artefact just to be
    listed on a roster is a poor citizen.
    """

    def __init__(self) -> None:
        super().__init__()
        self._service: Any = None

    @abstractmethod
    def build_service(self) -> Any:
        """Construct the underlying service. Called once, on first access."""

    @property
    def service(self) -> Any:
        if self._service is None:
            self._service = self.build_service()
        return self._service

    @property
    def is_ready(self) -> bool:
        """Whether the underlying service can currently answer.

        Override where readiness is more than "it constructed" — for example a
        model registry entry that exists but holds no trained artefact.
        """
        try:
            return self.service is not None
        except Exception as exc:                         # pragma: no cover
            self._log.warning("Service unavailable for '%s': %s", self.name, exc)
            return False


__all__ = ["AgentResult", "BaseAgent", "ToolAgent"]