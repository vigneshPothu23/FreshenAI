"""Sprint 5 — shared agent state, tasks and trace records.

Three small types carry everything the agent system passes around:

``AgentTask``
    One unit of work addressed to a named agent. Created by the planner,
    consumed by the coordinator, executed by a specialist.

``AgentState``
    The shared blackboard for a single run. Agents read what earlier agents
    established and write what they discover, so the recommendation agent can
    use the inventory agent's findings without either importing the other.

``AgentStep``
    An immutable record of something that happened. The step list is the audit
    trail persisted to ``agent_steps``.

Deliberate design points:

* **Facts and artifacts are separate.** ``facts`` holds small scalars and short
  lists that steer control flow and fit in a prompt. ``artifacts`` holds heavy
  payloads — DataFrames, recommendation objects, forecast results — that the UI
  renders but no prompt should ever contain. Conflating them is how agent
  systems end up with unusable context windows.
* **Tasks carry an id from birth.** The confirmation gate matches on
  ``task_id``, so it must exist before the task is ever shown to a user.
* **State is mutable, steps are append-only.** A run's history cannot be
  rewritten, which is what makes the trace trustworthy.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Iterable


class TaskStatus(str, Enum):
    """Lifecycle of a single task.

    Subclasses ``str`` so the value serialises directly into SQLite and JSON
    without a converter.
    """

    PENDING = "pending"
    RUNNING = "running"
    OK = "ok"
    EMPTY = "empty"
    ERROR = "error"
    REPLAN = "replan"
    SKIPPED = "skipped"
    AWAITING_CONFIRMATION = "awaiting_confirmation"

    @property
    def is_terminal(self) -> bool:
        return self in (TaskStatus.OK, TaskStatus.ERROR, TaskStatus.SKIPPED)

    @property
    def is_success(self) -> bool:
        return self is TaskStatus.OK


# ══════════════════════════════════════════════════════════════════════════
@dataclass
class AgentTask:
    """One unit of work addressed to a named agent.

    Attributes:
        agent: Logical agent name, matching a key in ``AGENT_REGISTRY``.
        action: The operation requested, matching one of the agent's
            ``supported_actions``.
        params: Arguments for the action. Kept JSON-serialisable so a plan can
            be persisted, replayed and diffed.
        rationale: Why the planner included this task. Surfaced verbatim at the
            confirmation gate, so it must read as an explanation to a human
            rather than as an internal note.
        task_id: Stable identifier assigned at construction. The confirmation
            gate matches on this, so it cannot be assigned later.
        depends_on: Task ids whose facts this task expects. Advisory — the
            coordinator executes in plan order — but it lets an agent detect a
            missing prerequisite and report it rather than fail obscurely.
        priority: Lower runs first when a planner returns an unordered set.
    """

    agent: str
    action: str
    params: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    task_id: str = field(default_factory=lambda: f"T-{uuid.uuid4().hex[:8].upper()}")
    depends_on: list[str] = field(default_factory=list)
    priority: int = 100
    status: TaskStatus = TaskStatus.PENDING

    def param(self, key: str, default: Any = None) -> Any:
        """Fetch one parameter with a default."""
        return self.params.get(key, default)

    def require(self, key: str) -> Any:
        """Fetch a parameter that the action cannot run without.

        Raises:
            KeyError: If the planner omitted a required parameter. Failing here
                produces a clear message in the trace rather than a downstream
                ``TypeError`` inside a service call.
        """
        if key not in self.params:
            raise KeyError(
                f"Task '{self.action}' for agent '{self.agent}' is missing "
                f"required parameter '{key}'"
            )
        return self.params[key]

    def describe(self) -> str:
        """Human-readable one-liner used in plan displays and logs."""
        readable = self.action.replace("_", " ")
        return f"{self.agent} › {readable}" + (f" — {self.rationale}" if self.rationale else "")

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "agent": self.agent,
            "action": self.action,
            "params": self.params,
            "rationale": self.rationale,
            "depends_on": self.depends_on,
            "priority": self.priority,
            "status": self.status.value,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AgentTask":
        """Rebuild a task from a persisted plan row."""
        return cls(
            agent=str(payload["agent"]),
            action=str(payload["action"]),
            params=dict(payload.get("params", {})),
            rationale=str(payload.get("rationale", "")),
            task_id=str(payload.get("task_id") or f"T-{uuid.uuid4().hex[:8].upper()}"),
            depends_on=list(payload.get("depends_on", [])),
            priority=int(payload.get("priority", 100)),
            status=TaskStatus(payload.get("status", "pending")),
        )


# ══════════════════════════════════════════════════════════════════════════
@dataclass
class AgentStep:
    """An append-only record of one thing that happened during a run.

    The field names mirror the ``agent_steps`` table exactly, so
    :meth:`as_dict` can be handed straight to
    ``MonitoringRepository.log_steps`` with no mapping layer.
    """

    step_index: int
    agent_name: str
    action: str
    tool_called: str = ""
    detail: str = ""
    status: str = "ok"
    latency_ms: int = 0
    created_at: str = field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    @property
    def succeeded(self) -> bool:
        return self.status == "ok"

    def as_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "agent_name": self.agent_name,
            "action": self.action,
            "tool_called": self.tool_called,
            "detail": self.detail,
            "status": self.status,
            "latency_ms": self.latency_ms,
            "created_at": self.created_at,
        }

    def __str__(self) -> str:                            # pragma: no cover
        return (f"[{self.step_index}] {self.agent_name}.{self.action} "
                f"({self.status}, {self.latency_ms} ms)")


# ══════════════════════════════════════════════════════════════════════════
@dataclass
class AgentState:
    """Shared blackboard for one orchestration run.

    Agents communicate only through this object. That indirection is what keeps
    the specialists decoupled: the recommendation agent consumes
    ``facts["at_risk_batch_ids"]`` without knowing which agent produced it, or
    whether one did at all.
    """

    run_id: str
    query: str
    session_id: str = ""
    facts: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    steps: list[AgentStep] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    started_at: str = field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    # ── facts: small, promptable, control-flow relevant ───────────────
    def set_fact(self, key: str, value: Any) -> None:
        """Record a scalar or short list that later agents may branch on."""
        self.facts[key] = value

    def update_facts(self, values: dict[str, Any]) -> None:
        self.facts.update(values)

    def fact(self, key: str, default: Any = None) -> Any:
        return self.facts.get(key, default)

    def has_fact(self, key: str) -> bool:
        """Whether a fact exists and carries a usable value.

        An empty list or an empty string counts as absent: a downstream agent
        asking "do I have batches to work with?" wants to know whether there is
        anything to act on, not whether a key was written.
        """
        value = self.facts.get(key)
        if value is None:
            return False
        if isinstance(value, (list, tuple, dict, str)) and len(value) == 0:
            return False
        return True

    # ── artifacts: heavy payloads that must never enter a prompt ──────
    def set_artifact(self, key: str, value: Any) -> None:
        """Store a DataFrame, model result or recommendation list for the UI."""
        self.artifacts[key] = value

    def artifact(self, key: str, default: Any = None) -> Any:
        return self.artifacts.get(key, default)

    # ── trace ─────────────────────────────────────────────────────────
    def record(
        self,
        agent_name: str,
        action: str,
        *,
        detail: str = "",
        status: str = "ok",
        latency_ms: int = 0,
        tool_called: str = "",
    ) -> AgentStep:
        """Append a step, assigning the next index automatically."""
        step = AgentStep(
            step_index=len(self.steps),
            agent_name=agent_name,
            action=action,
            tool_called=tool_called,
            detail=str(detail)[:500],
            status=status,
            latency_ms=int(latency_ms),
        )
        self.steps.append(step)
        if status == "error" and detail:
            self.errors.append(f"{agent_name}.{action}: {detail}"[:300])
        return step

    def steps_by(self, agent_name: str) -> list[AgentStep]:
        return [s for s in self.steps if s.agent_name == agent_name]

    def narratives(self, *, successful_only: bool = True) -> list[str]:
        """Detail lines suitable for final synthesis."""
        return [
            s.detail for s in self.steps
            if s.detail and (s.succeeded or not successful_only)
        ]

    # ── introspection ─────────────────────────────────────────────────
    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def has_errors(self) -> bool:
        return bool(self.errors)

    @property
    def total_latency_ms(self) -> int:
        return sum(s.latency_ms for s in self.steps)

    def prompt_context(self, *, max_chars: int = 1200) -> str:
        """Render the facts as compact text for an LLM prompt.

        Artifacts are excluded by construction — that is the whole reason for
        the fact/artifact split. Long values are truncated rather than dropped,
        so the model sees that something exists even when it cannot see all
        of it.
        """
        if not self.facts:
            return ""
        lines: list[str] = []
        for key, value in self.facts.items():
            # An empty fact tells the model nothing and costs it context.
            if not self.has_fact(key):
                continue
            if isinstance(value, (list, tuple)):
                rendered = ", ".join(str(v) for v in list(value)[:8])
                if len(value) > 8:
                    rendered += f", … (+{len(value) - 8} more)"
            elif isinstance(value, dict):
                rendered = ", ".join(f"{k}={v}" for k, v in list(value.items())[:6])
            else:
                rendered = str(value)
            lines.append(f"{key.replace('_', ' ')}: {rendered[:200]}")

        text = "\n".join(lines)
        return text if len(text) <= max_chars else text[:max_chars] + " …"

    def summary(self) -> dict[str, Any]:
        """Compact run summary for logging and the Agents page."""
        return {
            "run_id": self.run_id,
            "query": self.query,
            "session_id": self.session_id,
            "steps": self.step_count,
            "facts": len(self.facts),
            "artifacts": sorted(self.artifacts),
            "errors": len(self.errors),
            "total_latency_ms": self.total_latency_ms,
            "started_at": self.started_at,
        }

    def trace_dicts(self) -> list[dict[str, Any]]:
        """Whole trace in the shape ``MonitoringRepository.log_steps`` expects."""
        return [s.as_dict() for s in self.steps]


# ══════════════════════════════════════════════════════════════════════════
def order_tasks(tasks: Iterable[AgentTask]) -> list[AgentTask]:
    """Sort tasks by priority, preserving planner order within a priority band.

    Python's sort is stable, so a planner that returns an already-correct
    sequence and leaves every priority at the default gets that sequence back
    untouched.
    """
    return sorted(tasks, key=lambda t: t.priority)


__all__ = ["TaskStatus", "AgentTask", "AgentStep", "AgentState", "order_tasks"]