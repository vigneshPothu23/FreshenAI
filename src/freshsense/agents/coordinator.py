"""Sprint 5 — multi-agent orchestration.

The coordinator owns **control flow only**. It holds no domain knowledge: every
figure it reports was produced by a specialist agent, which in turn delegates to
the already-built services (repositories, forecasting, spoilage, recommendation
engine, RAG chain). Nothing in this file recomputes a risk score, a price or a
match.

What makes this an agent system rather than a function router:

**Planning.** The planner decomposes a request into an ordered task list before
any execution begins, so the plan is inspectable and auditable up front.

**Observation between steps.** After each task the coordinator reads what came
back and decides what to do next. A pipeline runs a fixed sequence; this reads
results and reacts.

**Autonomous replanning.** When a task returns empty, fails, or invalidates a
downstream assumption, the coordinator amends the remaining plan and continues.
That single behaviour is the difference between orchestration and a call graph.

**A confirmation gate.** Tasks that mutate state or commit money never execute
without explicit human approval. The plan pauses, surfaces exactly what it
intends to do, and waits.

**A complete trace.** Every step — including replans and refusals — is persisted
to ``agent_runs`` and ``agent_steps`` so Sprint 6 can measure the system rather
than trust it.

Agent contract expected by this module (implemented in ``base.py`` and the four
agent modules)::

    class BaseAgent:
        name: str
        description: str
        supported_actions: tuple[str, ...]
        def execute(self, state: AgentState, task: AgentTask) -> AgentResult: ...

Agents are resolved dynamically at construction. A missing or unimportable agent
is recorded as a capability gap and the plan routes around it; one broken
specialist must not take down the whole system.
"""

from __future__ import annotations

import importlib
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from freshsense.config import SETTINGS
from freshsense.db.repository import MonitoringRepository
from freshsense.exceptions import AgentExecutionError
from freshsense.logging_config import get_logger

LOG = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# Agent discovery
# ══════════════════════════════════════════════════════════════════════════
#: Logical agent name -> (module suffix, class name). Adding a specialist means
#: adding a row here; the coordinator itself never changes.
AGENT_REGISTRY: dict[str, tuple[str, str]] = {
    "planner": ("planner_agent", "PlannerAgent"),
    "inventory": ("inventory_agent", "InventoryAgent"),
    "recommendation": ("recommendation_agent", "RecommendationAgent"),
    "rag": ("rag_agent", "RAGAgent"),
    "forecast": ("forecast_agent", "ForecastAgent"),
}

#: Actions that change persisted state or commit money. These always stop at the
#: confirmation gate, regardless of which agent proposed them.
WRITE_ACTIONS: frozenset[str] = frozenset({
    "apply_pricing",
    "apply_discount",
    "create_order",
    "reserve_stock",
    "adjust_quantity",
    "place_restock_order",
    "notify_buyers",
    "write_off_batch",
})


def _load_agent(module_suffix: str, class_name: str) -> Any | None:
    """Import one agent class, returning ``None`` when unavailable.

    Import failure is logged and tolerated. The coordinator reports the gap in
    its plan rather than raising, because a single unimportable specialist
    should degrade capability, not availability.
    """
    try:
        module = importlib.import_module(f"freshsense.agents.{module_suffix}")
        return getattr(module, class_name)
    except Exception as exc:                             # pragma: no cover
        LOG.warning("Agent '%s.%s' unavailable: %s", module_suffix, class_name, exc)
        return None


# ══════════════════════════════════════════════════════════════════════════
# Run outcome
# ══════════════════════════════════════════════════════════════════════════
@dataclass
class CoordinatorRun:
    """Everything one orchestration produced, ready for UI and for audit."""

    run_id: str
    query: str
    intent: str = "unknown"
    answer: str = ""
    status: str = "completed"          # completed | awaiting_confirmation | failed
    plan: list[dict[str, Any]] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
    pending_confirmations: list[dict[str, Any]] = field(default_factory=list)
    replan_count: int = 0
    latency_ms: int = 0
    agents_used: list[str] = field(default_factory=list)
    capability_gaps: list[str] = field(default_factory=list)

    @property
    def needs_confirmation(self) -> bool:
        return self.status == "awaiting_confirmation"

    @property
    def succeeded(self) -> bool:
        return self.status in ("completed", "awaiting_confirmation")

    def trace_frame(self):
        """Trace as a DataFrame for the Agents page timeline."""
        import pandas as pd

        if not self.trace:
            return pd.DataFrame(
                columns=["step_index", "agent_name", "action", "status",
                         "latency_ms", "detail"]
            )
        return pd.DataFrame(self.trace)[
            ["step_index", "agent_name", "action", "status", "latency_ms", "detail"]
        ]

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id, "query": self.query, "intent": self.intent,
            "answer": self.answer, "status": self.status,
            "plan": self.plan, "trace": self.trace,
            "replan_count": self.replan_count, "latency_ms": self.latency_ms,
            "agents_used": self.agents_used,
            "pending_confirmations": self.pending_confirmations,
            "capability_gaps": self.capability_gaps,
        }


# ══════════════════════════════════════════════════════════════════════════
# Coordinator
# ══════════════════════════════════════════════════════════════════════════
class Coordinator:
    """Orchestrates the specialist agents through a plan-execute-observe loop."""

    def __init__(
        self,
        *,
        agents: dict[str, Any] | None = None,
        monitoring: MonitoringRepository | None = None,
    ) -> None:
        config = SETTINGS.agents
        self.max_steps = int(config.get("max_steps", 12))
        self.enable_replanning = bool(config.get("enable_replanning", True))
        self.require_confirmation = bool(config.get("human_confirmation_required", True))
        self.monitoring = monitoring or MonitoringRepository()

        self.agents: dict[str, Any] = agents if agents is not None else self._build_agents()
        self.capability_gaps: list[str] = [
            name for name in AGENT_REGISTRY if name not in self.agents
        ]
        if self.capability_gaps:
            LOG.warning("Coordinator starting with capability gaps: %s",
                        ", ".join(self.capability_gaps))
        LOG.info("Coordinator ready with agents: %s",
                 ", ".join(sorted(self.agents)) or "none")

    # ── construction ──────────────────────────────────────────────────
    def _build_agents(self) -> dict[str, Any]:
        """Instantiate every registered agent that imports cleanly."""
        built: dict[str, Any] = {}
        for name, (module_suffix, class_name) in AGENT_REGISTRY.items():
            agent_class = _load_agent(module_suffix, class_name)
            if agent_class is None:
                continue
            try:
                built[name] = agent_class()
            except Exception as exc:                     # pragma: no cover
                LOG.warning("Agent '%s' failed to initialise: %s", name, exc)
        return built

    def available_agents(self) -> list[dict[str, Any]]:
        """Agent roster for the Agents page."""
        return [
            {
                "name": name,
                "class": type(agent).__name__,
                "description": getattr(agent, "description", ""),
                "actions": list(getattr(agent, "supported_actions", ())),
            }
            for name, agent in sorted(self.agents.items())
        ]

    # ── planning ──────────────────────────────────────────────────────
    def _plan(self, state: Any) -> list[Any]:
        """Delegate decomposition to the planner agent.

        The planner is itself an agent, so planning is traced like any other
        step. If the planner is unavailable the coordinator falls back to a
        single broad task rather than failing: a degraded answer beats none.
        """
        planner = self.agents.get("planner")
        if planner is None:
            LOG.warning("Planner unavailable — falling back to a single RAG task")
            return [self._fallback_task(state)]

        from freshsense.agents.state import AgentTask  # local: avoids a cycle

        result = planner.execute(state, AgentTask(
            agent="planner", action="plan", params={"query": state.query},
            rationale="Decompose the request into an ordered task list",
        ))
        self._record(state, "planner", "plan", result)

        tasks = list(result.data.get("tasks", [])) if result.data else []
        if not tasks:
            return [self._fallback_task(state)]

        # Drop tasks addressed to agents that failed to load, so the run
        # proceeds with whatever capability is actually present.
        routable = [t for t in tasks if t.agent in self.agents]
        dropped = [t.agent for t in tasks if t.agent not in self.agents]
        if dropped:
            LOG.warning("Dropping %d task(s) for unavailable agent(s): %s",
                        len(dropped), ", ".join(sorted(set(dropped))))
        return routable or [self._fallback_task(state)]

    @staticmethod
    def _fallback_task(state: Any) -> Any:
        """Single knowledge-base task used when planning is unavailable."""
        from freshsense.agents.state import AgentTask

        return AgentTask(
            agent="rag", action="answer_question",
            params={"question": state.query},
            rationale="Planner unavailable; answering directly from the knowledge base",
        )

    # ── replanning ────────────────────────────────────────────────────
    def _replan(self, state: Any, failed_task: Any, result: Any) -> list[Any]:
        """Ask the planner to amend the remaining plan after a bad result.

        This is the autonomous-behaviour requirement. The planner sees what
        failed and why, and returns replacement tasks — for example, widening a
        search radius after an empty match, or substituting a knowledge-base
        lookup when a model artefact is missing.
        """
        if not self.enable_replanning:
            return []

        planner = self.agents.get("planner")
        if planner is None:
            return []

        from freshsense.agents.state import AgentTask

        amendment = planner.execute(state, AgentTask(
            agent="planner", action="replan",
            params={
                "query": state.query,
                "failed_agent": failed_task.agent,
                "failed_action": failed_task.action,
                "failure_reason": result.error or "no results returned",
                "facts": dict(state.facts),
            },
            rationale=f"Recover from '{failed_task.action}' returning no usable result",
        ))
        self._record(state, "planner", "replan", amendment, status="replan")

        tasks = list(amendment.data.get("tasks", [])) if amendment.data else []
        return [t for t in tasks if t.agent in self.agents]

    # ── execution ─────────────────────────────────────────────────────
    def run(
        self,
        query: str,
        *,
        session_id: str = "",
        confirmed_actions: Sequence[str] = (),
        context: dict[str, Any] | None = None,
    ) -> CoordinatorRun:
        """Execute one orchestration end to end.

        Args:
            query: The user's request in natural language.
            session_id: Chat session this run belongs to, for trace grouping.
            confirmed_actions: Task ids the user has explicitly approved. A
                write action absent from this list pauses the run instead of
                executing.
            context: Optional pre-seeded facts (for example a selected batch id
                from the Inventory page).

        Returns:
            A :class:`CoordinatorRun` carrying the answer, the plan, the full
            step trace and any pending confirmations.
        """
        from freshsense.agents.state import AgentState

        started = time.perf_counter()
        run_id = f"RUN-{uuid.uuid4().hex[:12].upper()}"

        state = AgentState(
            run_id=run_id,
            query=query.strip(),
            session_id=session_id,
            facts=dict(context or {}),
        )
        self.monitoring.start_run(run_id, state.query, session_id)

        outcome = CoordinatorRun(
            run_id=run_id,
            query=state.query,
            capability_gaps=list(self.capability_gaps),
        )

        if not state.query:
            outcome.status = "failed"
            outcome.answer = "Please describe what you would like me to do."
            self._finalise(state, outcome, started)
            return outcome

        try:
            queue: list[Any] = self._plan(state)
            outcome.plan = [self._describe_task(t) for t in queue]
            outcome.intent = str(state.facts.get("intent", "unknown"))

            executed = 0
            while queue and executed < self.max_steps:
                task = queue.pop(0)
                executed += 1

                # ── confirmation gate ─────────────────────────────────
                if self._needs_confirmation(task, confirmed_actions):
                    outcome.pending_confirmations.append(self._describe_task(task))
                    self._record_pending(state, task)
                    LOG.info("Run %s paused for confirmation on '%s'",
                             run_id, task.action)
                    continue

                agent = self.agents.get(task.agent)
                if agent is None:
                    LOG.warning("No agent registered for '%s'", task.agent)
                    continue

                result = self._execute_task(state, agent, task)
                if task.agent not in outcome.agents_used:
                    outcome.agents_used.append(task.agent)

                # ── observe, then decide ──────────────────────────────
                if not result.success or result.is_empty:
                    amendments = self._replan(state, task, result)
                    if amendments:
                        outcome.replan_count += 1
                        # Amendments run before the remaining plan, because a
                        # later task may depend on what the recovery produces.
                        queue = amendments + queue
                        outcome.plan.extend(
                            self._describe_task(t) | {"origin": "replan"}
                            for t in amendments
                        )
                    continue

                state.facts.update(result.facts)
                if result.artifacts:
                    state.artifacts.update(result.artifacts)

            if executed >= self.max_steps and queue:
                LOG.warning("Run %s hit the %d-step ceiling with %d task(s) left",
                            run_id, self.max_steps, len(queue))

            outcome.artifacts = dict(state.artifacts)
            outcome.status = (
                "awaiting_confirmation" if outcome.pending_confirmations else "completed"
            )
            outcome.answer = self._synthesise(state, outcome)

        except AgentExecutionError as exc:
            LOG.error("Run %s failed: %s", run_id, exc)
            outcome.status = "failed"
            outcome.answer = (
                "I could not complete that request. The step that failed was "
                f"'{exc}'. The partial results above are still valid."
            )
        except Exception as exc:                         # pragma: no cover
            LOG.exception("Run %s raised an unexpected error", run_id)
            outcome.status = "failed"
            outcome.answer = (
                "Something went wrong while coordinating the agents. The error "
                "has been logged; nothing was changed in your inventory."
            )

        self._finalise(state, outcome, started)
        return outcome

    def _execute_task(self, state: Any, agent: Any, task: Any) -> Any:
        """Run one task with timing, error containment and tracing."""
        step_started = time.perf_counter()
        try:
            result = agent.execute(state, task)
        except Exception as exc:
            from freshsense.agents.base import AgentResult

            LOG.warning("Agent '%s' raised during '%s': %s",
                        task.agent, task.action, exc)
            result = AgentResult(
                agent=task.agent, action=task.action,
                success=False, error=str(exc)[:300],
            )

        latency = int((time.perf_counter() - step_started) * 1000)
        self._record(state, task.agent, task.action, result, latency_ms=latency)
        return result

    def _needs_confirmation(self, task: Any, confirmed: Sequence[str]) -> bool:
        """Whether this task must stop and wait for a human."""
        if not self.require_confirmation:
            return False
        if task.action not in WRITE_ACTIONS:
            return False
        return task.task_id not in set(confirmed)

    # ── synthesis ─────────────────────────────────────────────────────
    def _synthesise(self, state: Any, outcome: CoordinatorRun) -> str:
        """Compose the final answer from what the agents actually returned.

        Agent narratives are the source of truth. The language model is used
        only to join them into readable prose — it is never asked to supply a
        fact, which is what keeps the output faithful to the computation.
        """
        # Read the live step list, not outcome.trace: _finalise populates the
        # latter after synthesis runs. Planner steps are excluded because they
        # describe the approach, and the user asked about their stock.
        narratives = [
            step.detail for step in state.steps
            if step.status == "ok" and step.detail and step.agent_name != "planner"
        ]

        if not narratives:
            if outcome.pending_confirmations:
                return self._confirmation_summary(outcome)
            return (
                "I could not find anything to report for that request. Try "
                "naming a product, a zone or a seller, or ask about food-safety "
                "and storage guidance."
            )

        findings = "\n".join(f"- {line}" for line in narratives)

        # Deterministic body first: it stands alone if generation is offline.
        body = self._deterministic_summary(state, outcome, narratives)

        try:
            from freshsense.rag.llm_wrapper import ask_llm

            response = ask_llm(
                "Rewrite the findings below as a short reply to the user's "
                "request. Use only the facts given — do not add numbers, "
                "products or recommendations that do not appear. Three or four "
                "sentences, plain prose, no bullet points.\n\n"
                f"REQUEST: {state.query}\n\nFINDINGS:\n{findings}",
                system=(
                    "You summarise the output of an inventory analysis system. "
                    "Every figure must come from the findings supplied. Never "
                    "invent a value."
                ),
                context=[{"text": findings, "source": "agent findings"}],
            )
            if response.success and response.text.strip() and not response.is_stub:
                text = response.text.strip()
                if outcome.pending_confirmations:
                    text += "\n\n" + self._confirmation_summary(outcome)
                return text
            if response.is_stub:
                # The stub is an extractive sentence ranker. Agent narratives
                # are already complete sentences, so passing them through it
                # reorders and truncates finished prose. The deterministic
                # summary is strictly better offline.
                LOG.debug("Stub provider active; using the deterministic summary")
        except Exception as exc:                         # pragma: no cover
            LOG.warning("Synthesis via LLM failed (%s); using the direct summary", exc)

        return body

    @staticmethod
    def _deterministic_summary(
        state: Any, outcome: CoordinatorRun, narratives: Iterable[str]
    ) -> str:
        """Readable summary built without any language model."""
        lines = list(narratives)
        parts = [f"Here is what I found for: {state.query}", ""]
        parts.extend(f"• {line}" for line in lines)
        if outcome.replan_count:
            parts.append(
                f"\n(I adjusted my approach {outcome.replan_count} time(s) when "
                f"a step returned nothing usable.)"
            )
        if outcome.pending_confirmations:
            parts.append("\n" + Coordinator._confirmation_summary(outcome))
        return "\n".join(parts)

    @staticmethod
    def _confirmation_summary(outcome: CoordinatorRun) -> str:
        """Describe exactly what is waiting on the user, and nothing more."""
        items = "\n".join(
            f"• {task['action'].replace('_', ' ')} — {task['rationale']}"
            for task in outcome.pending_confirmations
        )
        return (
            "The following action(s) change your data or commit an order, so I "
            f"have not run them:\n{items}\n"
            "Approve them to proceed."
        )

    # ── tracing ───────────────────────────────────────────────────────
    def _record(
        self,
        state: Any,
        agent_name: str,
        action: str,
        result: Any,
        *,
        latency_ms: int = 0,
        status: str | None = None,
    ) -> None:
        """Append one step to the in-memory trace on the shared state."""
        from freshsense.agents.state import AgentStep

        resolved = status or ("ok" if getattr(result, "success", False) else "error")
        detail = (
            getattr(result, "narrative", "")
            or getattr(result, "error", "")
            or f"{action} completed"
        )
        state.steps.append(AgentStep(
            step_index=len(state.steps),
            agent_name=agent_name,
            action=action,
            tool_called=getattr(result, "tool_called", "") or "",
            detail=str(detail)[:500],
            status=resolved,
            latency_ms=latency_ms or int(getattr(result, "latency_ms", 0)),
        ))

    def _record_pending(self, state: Any, task: Any) -> None:
        """Record a gated task in the trace so the pause is auditable."""
        from freshsense.agents.state import AgentStep

        state.steps.append(AgentStep(
            step_index=len(state.steps),
            agent_name=task.agent,
            action=task.action,
            tool_called="",
            detail=f"Held for human confirmation: {task.rationale}"[:500],
            status="pending",
            latency_ms=0,
        ))

    @staticmethod
    def _describe_task(task: Any) -> dict[str, Any]:
        """Plan-row representation of a task, for display and persistence."""
        return {
            "task_id": task.task_id,
            "agent": task.agent,
            "action": task.action,
            "rationale": task.rationale,
            "requires_confirmation": task.action in WRITE_ACTIONS,
            "origin": "plan",
        }

    def _finalise(self, state: Any, outcome: CoordinatorRun, started: float) -> None:
        """Persist the run and its steps, then close out the outcome object."""
        outcome.latency_ms = int((time.perf_counter() - started) * 1000)
        outcome.trace = [step.as_dict() for step in state.steps]

        try:
            self.monitoring.log_steps(outcome.run_id, outcome.trace)
            self.monitoring.finish_run(
                outcome.run_id,
                final_output=outcome.answer,
                status=outcome.status,
                latency_ms=outcome.latency_ms,
                replan_count=outcome.replan_count,
            )
            self.monitoring.log_event(
                component="coordinator",
                action="run",
                status="ok" if outcome.succeeded else "error",
                latency_ms=outcome.latency_ms,
                detail=outcome.query[:200],
                metadata={
                    "run_id": outcome.run_id,
                    "intent": outcome.intent,
                    "agents_used": outcome.agents_used,
                    "steps": len(outcome.trace),
                    "replans": outcome.replan_count,
                },
            )
        except Exception as exc:                         # pragma: no cover
            # Observability must never determine whether the user gets an answer.
            LOG.warning("Could not persist the agent trace for %s: %s",
                        outcome.run_id, exc)

        LOG.info(
            "Run %s %s in %d ms | %d step(s), %d replan(s), agents: %s",
            outcome.run_id, outcome.status, outcome.latency_ms,
            len(outcome.trace), outcome.replan_count,
            ", ".join(outcome.agents_used) or "none",
        )


_COORDINATOR: Coordinator | None = None


def get_coordinator() -> Coordinator:
    """Return the process-wide coordinator, building the agent roster once."""
    global _COORDINATOR
    if _COORDINATOR is None:
        _COORDINATOR = Coordinator()
    return _COORDINATOR


def reset_coordinator() -> Coordinator:
    """Rebuild the coordinator after agents or configuration change."""
    global _COORDINATOR
    _COORDINATOR = Coordinator()
    return _COORDINATOR


__all__ = [
    "Coordinator", "CoordinatorRun", "get_coordinator", "reset_coordinator",
    "AGENT_REGISTRY", "WRITE_ACTIONS",
]