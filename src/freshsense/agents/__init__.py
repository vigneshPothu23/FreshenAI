"""Sprint 5 — multi-agent system.

A coordinator plus four specialists, each a thin adapter over a service built in
an earlier sprint. No agent contains business logic: the inventory agent reads
:class:`InventoryRepository`, the recommendation agent calls the Sprint 3
engine, the forecast agent serves Sprint 2 models, and the knowledge agent runs
the Sprint 4 RAG chain. That constraint is what keeps every capability reachable
from the UI without an orchestration run, and testable without one.

Import order below is deliberate. ``state`` and ``base`` define the contract and
depend on nothing else in the package; the specialists depend on those; the
coordinator resolves specialists dynamically at construction and so imports none
of them at module level. The result is a package with no import cycle.

Typical use::

    from freshsense.agents import get_coordinator

    run = get_coordinator().run("What is at risk today in T Nagar?")
    print(run.answer)
    print(run.trace_frame())

Note that ``Coordinator`` discovers agents through
:data:`~freshsense.agents.coordinator.AGENT_REGISTRY` rather than through this
module, so a specialist that fails to import degrades to a reported capability
gap instead of breaking the package.
"""

from __future__ import annotations

from freshsense.agents.base import AgentResult, BaseAgent, ToolAgent
from freshsense.agents.coordinator import (AGENT_REGISTRY, WRITE_ACTIONS,
                                           Coordinator, CoordinatorRun,
                                           get_coordinator, reset_coordinator)
from freshsense.agents.forecast_agent import ForecastAgent
from freshsense.agents.inventory_agent import InventoryAgent
from freshsense.agents.planner_agent import ACTION_CATALOGUE, PlannerAgent
from freshsense.agents.rag_agent import RAGAgent
from freshsense.agents.recommendation_agent import RecommendationAgent
from freshsense.agents.state import (AgentState, AgentStep, AgentTask,
                                     TaskStatus, order_tasks)

#: Logical agent name -> class. Mirrors ``AGENT_REGISTRY`` but resolved at import
#: time, so tests can construct a specialist directly without going through the
#: coordinator's dynamic loader.
AGENT_CLASSES: dict[str, type[BaseAgent]] = {
    "planner": PlannerAgent,
    "inventory": InventoryAgent,
    "recommendation": RecommendationAgent,
    "forecast": ForecastAgent,
    "rag": RAGAgent,
}


def build_agents(*names: str) -> dict[str, BaseAgent]:
    """Instantiate named agents, or all of them when none are named.

    Intended for tests and for scripts that want a specific subset. The
    coordinator does not use this — it resolves agents dynamically so that one
    unimportable specialist cannot take down the whole system.

    Args:
        *names: Logical agent names. Empty means every registered agent.

    Returns:
        Mapping of agent name to a constructed instance.

    Raises:
        KeyError: If a requested name is not registered.
    """
    selected = names or tuple(AGENT_CLASSES)
    unknown = [n for n in selected if n not in AGENT_CLASSES]
    if unknown:
        raise KeyError(
            f"Unknown agent(s): {', '.join(unknown)}. "
            f"Registered: {', '.join(sorted(AGENT_CLASSES))}"
        )
    return {name: AGENT_CLASSES[name]() for name in selected}


def verify_action_catalogue() -> dict[str, list[str]]:
    """Check that every planner-emitted action has an implementing agent.

    The planner and the specialists agree on a vocabulary
    (:data:`ACTION_CATALOGUE`). Drift between the two is silent at import time
    and only surfaces as an unroutable task mid-run, so this makes the contract
    checkable — the test suite asserts it returns empty.

    Returns:
        Mapping of agent name to actions the catalogue declares but the agent
        does not implement. An empty mapping means the contract holds.
    """
    gaps: dict[str, list[str]] = {}
    for agent_name, actions in ACTION_CATALOGUE.items():
        agent_class = AGENT_CLASSES.get(agent_name)
        if agent_class is None:
            gaps[agent_name] = list(actions)
            continue
        implemented = set(agent_class.supported_actions)
        missing = [a for a in actions if a not in implemented]
        if missing:
            gaps[agent_name] = missing
    return gaps


__all__ = [
    # contract
    "AgentState", "AgentStep", "AgentTask", "TaskStatus", "order_tasks",
    "AgentResult", "BaseAgent", "ToolAgent",
    # specialists
    "PlannerAgent", "InventoryAgent", "RecommendationAgent", "ForecastAgent",
    "RAGAgent",
    # orchestration
    "Coordinator", "CoordinatorRun", "get_coordinator", "reset_coordinator",
    "AGENT_REGISTRY", "WRITE_ACTIONS", "ACTION_CATALOGUE",
    # helpers
    "AGENT_CLASSES", "build_agents", "verify_action_catalogue",
]