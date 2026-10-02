"""Sprint 5 — the planner agent.

Decomposes a request into an ordered task list, and amends that list when a step
comes back empty or broken. It is the only agent that reasons about *the plan*
rather than about inventory.

Two deliberate choices:

**Rules first, model second.** Intent classification is a deterministic keyword
scorer. A language model is consulted only when the rules are genuinely
ambiguous, and its answer is constrained to the known intent vocabulary — an
unparseable reply falls back to the rule result. Planning therefore stays fully
functional offline, and identical input always produces an identical plan, which
is what makes agent behaviour testable at all.

**Replanning is bounded and specific.** Each recovery strategy is tied to a
concrete failure mode (an empty match, a missing model artefact, a refused
retrieval) and each may fire once per run. Unbounded self-correction is how
agent systems burn a hundred steps arriving nowhere; the recovery ledger in
``state.facts`` prevents a loop.

The planner emits actions from :data:`ACTION_CATALOGUE`. That catalogue is the
contract the specialist agents implement — a task naming an action outside it is
a planner bug, not an agent gap.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from freshsense.agents.base import AgentResult, BaseAgent
from freshsense.agents.state import AgentState, AgentTask
from freshsense.data.cleaning import (CANONICAL_PRODUCTS, CANONICAL_ZONES,
                                      PRODUCT_LOOKUP, ZONE_LOOKUP)
from freshsense.logging_config import get_logger

LOG = get_logger(__name__)


#: The action vocabulary. Every specialist agent's ``supported_actions`` must be
#: a subset of its row here.
ACTION_CATALOGUE: dict[str, tuple[str, ...]] = {
    "inventory": (
        "find_at_risk", "search_inventory", "inventory_summary",
        "expiring_soon", "low_stock",
    ),
    "forecast": ("forecast_demand", "forecast_summary"),
    "recommendation": (
        "recommend_actions", "recommend_pricing", "recommend_buyers",
        "match_sellers", "recommend_restocking",
    ),
    "rag": ("answer_question", "lookup_policy"),
}

#: Intent -> keyword weights. Multi-word phrases score higher because they are
#: far less likely to appear incidentally than a single common word.
INTENT_KEYWORDS: dict[str, dict[str, float]] = {
    "at_risk_review": {
        "at risk": 3.0, "expiring": 3.0, "expire": 2.5, "spoil": 2.5,
        "going bad": 3.0, "waste": 2.0, "urgent": 1.5, "today": 1.0,
        "near expiry": 3.0, "risk": 1.0, "wastage": 2.0,
    },
    "pricing": {
        "price": 2.5, "pricing": 3.0, "discount": 3.0, "markdown": 3.0,
        "how much should": 3.0, "reprice": 3.0, "margin": 1.5, "cheaper": 1.5,
    },
    "forecast_demand": {
        "forecast": 3.0, "demand": 2.5, "predict": 2.5, "how much will": 3.0,
        "next week": 2.0, "expected sales": 3.0, "projection": 2.0,
        "sell tomorrow": 3.0, "trend": 1.5,
    },
    "find_buyers": {
        "buyer": 3.0, "who will buy": 3.0, "sell to": 2.5, "offload": 2.5,
        "customers": 2.0, "clear stock": 2.5, "caterer": 2.0, "hotel": 1.5,
    },
    "source_supply": {
        "buy": 2.0, "source": 2.5, "supplier": 2.5, "need": 1.5,
        "looking for": 2.0, "purchase": 2.5, "procure": 2.5, "order": 1.5,
        "cheapest": 2.5, "find me": 2.0, "who has": 2.5,
    },
    "restocking": {
        "restock": 3.0, "reorder": 3.0, "replenish": 3.0, "run out": 2.5,
        "stock up": 2.5, "how much to order": 3.0, "purchase order": 2.0,
    },
    "knowledge": {
        "fssai": 3.0, "regulation": 3.0, "licence": 2.5, "license": 2.5,
        "policy": 2.5, "how long": 2.0, "shelf life": 2.5, "store": 1.5,
        "temperature": 2.0, "safe": 1.5, "rules": 2.0, "legal": 2.5,
        "compliance": 3.0, "guideline": 2.5, "what is": 1.0, "donate": 2.0,
    },
    "inventory_lookup": {
        "how much": 2.0, "do i have": 3.0, "in stock": 3.0, "show me": 2.0,
        "list": 1.5, "inventory": 2.0, "available": 1.5, "quantity": 1.5,
    },
    "daily_briefing": {
        "briefing": 3.0, "summary": 2.0, "overview": 2.5, "what should i do": 3.5,
        "morning": 2.0, "status": 1.5, "dashboard": 2.0, "priorities": 2.5,
        "walk me through": 3.0,
    },
}

#: Below this the rule scorer is not confident enough to plan on its own.
AMBIGUITY_THRESHOLD = 2.5


class PlannerAgent(BaseAgent):
    """Turns a natural-language request into an executable, ordered plan."""

    name = "planner"
    description = (
        "Classifies intent, extracts entities and decomposes a request into an "
        "ordered task list. Amends the plan when a step returns nothing usable."
    )
    supported_actions = ("plan", "replan")

    # ── entity extraction ─────────────────────────────────────────────
    @staticmethod
    def extract_product(text: str) -> str | None:
        """Resolve a product mention to its canonical catalogue name.

        Matching runs longest-first so "Brown Bread" is not swallowed by
        "Bread", and reuses the Sprint 1 lookup rather than a second vocabulary.
        """
        lowered = text.lower()
        for product in sorted(CANONICAL_PRODUCTS, key=len, reverse=True):
            if product.lower() in lowered:
                return product
        # Fall back to the whitespace-insensitive lookup ("whitebread").
        for token in re.findall(r"[a-z]{4,}", lowered):
            if match := PRODUCT_LOOKUP.get(token):
                return match
        return None

    @staticmethod
    def extract_zone(text: str) -> str | None:
        """Resolve a zone mention, tolerating the spelling variants seen in the data."""
        lowered = text.lower()
        for zone in sorted(CANONICAL_ZONES, key=len, reverse=True):
            if zone.lower() in lowered:
                return zone
        for variant, canonical in ZONE_LOOKUP.items():
            if variant in lowered:
                return canonical
        return None

    @staticmethod
    def extract_quantity(text: str) -> float | None:
        """Pull a requested quantity, e.g. '40 kg of tomatoes'."""
        match = re.search(
            r"(\d+(?:\.\d+)?)\s*(kg|kilo|kilos|kilogram[s]?|l|litre[s]?|"
            r"liter[s]?|unit[s]?|packet[s]?|piece[s]?|dozen)\b",
            text.lower(),
        )
        if match:
            return float(match.group(1))
        # A bare number alongside a buying verb is still a quantity.
        if re.search(r"\b(need|want|buy|order|source|looking for)\b", text.lower()):
            if bare := re.search(r"\b(\d{1,4}(?:\.\d+)?)\b", text):
                return float(bare.group(1))
        return None

    @staticmethod
    def extract_days(text: str, default: int = 2) -> int:
        """Pull a time window, e.g. 'expiring in 3 days'."""
        if match := re.search(r"(\d+)\s*day", text.lower()):
            return max(0, min(int(match.group(1)), 30))
        lowered = text.lower()
        if "today" in lowered or "tonight" in lowered:
            return 0
        if "tomorrow" in lowered:
            return 1
        if "this week" in lowered or "week" in lowered:
            return 7
        return default

    # ── intent classification ─────────────────────────────────────────
    def score_intents(self, text: str) -> dict[str, float]:
        """Weighted keyword scores for every intent."""
        lowered = f" {text.lower()} "
        scores: dict[str, float] = {}
        for intent, keywords in INTENT_KEYWORDS.items():
            total = sum(weight for phrase, weight in keywords.items()
                        if phrase in lowered)
            if total:
                scores[intent] = round(total, 2)
        return dict(sorted(scores.items(), key=lambda kv: -kv[1]))

    def classify(self, text: str) -> tuple[str, float, dict[str, float]]:
        """Classify intent, consulting a model only when the rules are unsure.

        Returns:
            ``(intent, confidence, all_scores)``. Confidence is the margin
            between the top two candidates, so a request that fires two intents
            equally is correctly reported as ambiguous rather than as a
            confident pick of whichever sorted first.
        """
        scores = self.score_intents(text)
        if not scores:
            return "unknown", 0.0, scores

        ranked = list(scores.items())
        top_intent, top_score = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = top_score - runner_up

        if top_score >= AMBIGUITY_THRESHOLD and margin >= 1.0:
            return top_intent, round(margin, 2), scores

        resolved = self._disambiguate(text, [i for i, _ in ranked[:3]])
        return (resolved or top_intent), round(margin, 2), scores

    def _disambiguate(self, text: str, candidates: list[str]) -> str | None:
        """Ask the model to choose between candidate intents.

        The reply is constrained to the candidate list and validated. Anything
        unrecognised is discarded, so a hallucinated intent can never enter the
        plan.
        """
        if len(candidates) < 2:
            return None
        try:
            from freshsense.rag.llm_wrapper import ask_llm

            response = ask_llm(
                "Classify the request into exactly one of these categories. "
                "Reply with the category name only, nothing else.\n\n"
                f"CATEGORIES: {', '.join(candidates)}\n\nREQUEST: {text}",
                system="You are an intent classifier. Reply with one category "
                       "name from the supplied list and no other text.",
                max_tokens=16,
                temperature=0.0,
            )
            if not response.success:
                return None
            reply = response.text.strip().lower()
            for candidate in candidates:
                if candidate in reply:
                    LOG.debug("Model disambiguated to '%s'", candidate)
                    return candidate
        except Exception as exc:                         # pragma: no cover
            LOG.debug("Disambiguation unavailable (%s); using the rule result", exc)
        return None

    # ── planning ──────────────────────────────────────────────────────
    def action_plan(self, state: AgentState, task: AgentTask) -> AgentResult:
        """Classify the request and emit an ordered task list."""
        query = str(task.param("query") or state.query).strip()
        if not query:
            return self.fail("no query to plan for")

        intent, confidence, scores = self.classify(query)
        entities = {
            "product": self.extract_product(query),
            "zone": self.extract_zone(query),
            "quantity": self.extract_quantity(query),
            "days": self.extract_days(query),
        }
        entities = {k: v for k, v in entities.items() if v is not None}

        tasks = self._build_plan(intent, entities, query)
        if not tasks:
            return self.empty(f"no plan could be built for intent '{intent}'")

        # The coordinator reads intent and entities before the execution loop
        # begins, so the planner seeds them directly. It is the one agent that
        # writes to state, because its output *is* the run's starting context.
        state.set_fact("intent", intent)
        state.set_fact("intent_confidence", confidence)
        for key, value in entities.items():
            state.set_fact(key, value)

        described = ", ".join(f"{k}={v}" for k, v in entities.items()) or "none"
        return self.ok(
            f"Interpreted the request as '{intent.replace('_', ' ')}' "
            f"(entities: {described}) and planned {len(tasks)} step(s).",
            facts={"intent": intent, "intent_confidence": confidence, **entities},
            data={"tasks": tasks, "intent": intent, "scores": scores},
            tool_called="PlannerAgent.classify",
        )

    def _build_plan(
        self, intent: str, entities: dict[str, Any], query: str
    ) -> list[AgentTask]:
        """Map an intent plus entities onto a concrete task sequence.

        Composite intents produce genuinely multi-agent plans: a daily briefing
        touches inventory, forecasting, recommendation and the knowledge base,
        with each step consuming what the previous one established.
        """
        product = entities.get("product")
        zone = entities.get("zone")
        quantity = entities.get("quantity")
        days = int(entities.get("days", 2))

        builders = {
            "at_risk_review": lambda: [
                AgentTask(
                    agent="inventory", action="find_at_risk",
                    params={"days": days, "zone": zone, "product": product},
                    rationale=f"Identify stock expiring within {days} day(s)",
                    priority=10,
                ),
                AgentTask(
                    agent="recommendation", action="recommend_actions",
                    params={"top_n": 5},
                    rationale="Decide what to do with each at-risk batch",
                    priority=20,
                ),
            ],
            "pricing": lambda: [
                AgentTask(
                    agent="inventory", action="search_inventory",
                    params={"product": product, "zone": zone},
                    rationale="Locate the batches to be priced",
                    priority=10,
                ),
                AgentTask(
                    agent="forecast", action="forecast_demand",
                    params={"product": product, "horizon": 2},
                    rationale="Establish expected demand before setting a discount",
                    priority=20,
                ),
                AgentTask(
                    agent="recommendation", action="recommend_pricing",
                    params={"product": product},
                    rationale="Apply the decay curve against risk and surplus",
                    priority=30,
                ),
            ],
            "forecast_demand": lambda: [
                AgentTask(
                    agent="forecast", action="forecast_demand",
                    params={"product": product, "horizon": max(days, 7)},
                    rationale=f"Project demand for the next {max(days, 7)} day(s)",
                    priority=10,
                ),
            ],
            "find_buyers": lambda: [
                AgentTask(
                    agent="inventory", action="search_inventory",
                    params={"product": product, "zone": zone},
                    rationale="Find the stock that needs a buyer",
                    priority=10,
                ),
                AgentTask(
                    agent="recommendation", action="recommend_buyers",
                    params={"product": product, "top_n": 5},
                    rationale="Rank buyers by proximity, history and capacity",
                    priority=20,
                ),
            ],
            "source_supply": lambda: [
                AgentTask(
                    agent="recommendation", action="match_sellers",
                    params={"product": product, "quantity": quantity or 10.0,
                            "zone": zone},
                    rationale=(f"Find sellers who can supply "
                               f"{quantity or 10:.0f} unit(s) of "
                               f"{product or 'the requested item'}"),
                    priority=10,
                ),
            ],
            "restocking": lambda: [
                AgentTask(
                    agent="forecast", action="forecast_summary",
                    params={"horizon": 2},
                    rationale="Establish forward demand across the catalogue",
                    priority=10,
                ),
                AgentTask(
                    agent="recommendation", action="recommend_restocking",
                    params={"cover_days": max(days, 3), "top_n": 8},
                    rationale="Compute reorder quantities against current cover",
                    priority=20,
                ),
            ],
            "knowledge": lambda: [
                AgentTask(
                    agent="rag", action="answer_question",
                    params={"question": query, "product": product},
                    rationale="Answer from the food-safety and policy corpus",
                    priority=10,
                ),
            ],
            "inventory_lookup": lambda: [
                AgentTask(
                    agent="inventory", action="search_inventory",
                    params={"product": product, "zone": zone},
                    rationale="Read current stock matching the request",
                    priority=10,
                ),
            ],
            "daily_briefing": lambda: [
                AgentTask(
                    agent="inventory", action="inventory_summary",
                    params={"zone": zone},
                    rationale="Establish the current position across all stock",
                    priority=10,
                ),
                AgentTask(
                    agent="inventory", action="find_at_risk",
                    params={"days": 2, "zone": zone},
                    rationale="Surface what needs attention first",
                    priority=20,
                ),
                AgentTask(
                    agent="forecast", action="forecast_summary",
                    params={"horizon": 2},
                    rationale="Compare stock on hand against expected demand",
                    priority=30,
                ),
                AgentTask(
                    agent="recommendation", action="recommend_actions",
                    params={"top_n": 5},
                    rationale="Produce the prioritised action list for today",
                    priority=40,
                ),
            ],
        }

        if builder := builders.get(intent):
            return builder()

        # Unknown intent: try the knowledge base, and read inventory too when a
        # product was named, since the request is probably about that stock.
        fallback = [
            AgentTask(
                agent="rag", action="answer_question",
                params={"question": query},
                rationale="Intent was unclear — check the knowledge base",
                priority=10,
            ),
        ]
        if product:
            fallback.append(AgentTask(
                agent="inventory", action="search_inventory",
                params={"product": product, "zone": zone},
                rationale=f"'{product}' was mentioned — report current stock",
                priority=20,
            ))
        return fallback

    # ── replanning ────────────────────────────────────────────────────
    def action_replan(self, state: AgentState, task: AgentTask) -> AgentResult:
        """Produce recovery tasks after a step returned nothing usable.

        Each strategy fires at most once per run. The ledger lives in
        ``state.facts`` so it survives across coordinator iterations without the
        planner holding run-scoped state on itself — the agent stays a
        singleton, safe to share.
        """
        failed_agent = str(task.param("failed_agent", ""))
        failed_action = str(task.param("failed_action", ""))
        reason = str(task.param("failure_reason", "no results returned"))
        facts = dict(task.param("facts", {}) or {})

        ledger_key = f"{failed_agent}.{failed_action}"
        attempted: list[str] = list(state.fact("recovery_attempts", []))
        if ledger_key in attempted:
            return self.empty(
                f"already attempted recovery for {ledger_key}; not retrying",
            )
        attempted.append(ledger_key)
        state.set_fact("recovery_attempts", attempted)

        tasks = self._recovery_tasks(failed_agent, failed_action, facts, state)
        if not tasks:
            return self.empty(
                f"no recovery available for {ledger_key} ({reason})"
            )

        return self.ok(
            f"'{failed_action}' returned nothing ({reason}); "
            f"trying {len(tasks)} alternative step(s).",
            data={"tasks": tasks, "recovering": ledger_key},
            tool_called="PlannerAgent.replan",
        )

    def _recovery_tasks(
        self,
        failed_agent: str,
        failed_action: str,
        facts: dict[str, Any],
        state: AgentState,
    ) -> list[AgentTask]:
        """Concrete recovery strategy per known failure mode."""
        product = facts.get("product") or state.fact("product")
        zone = facts.get("zone") or state.fact("zone")
        quantity = facts.get("quantity") or state.fact("quantity") or 10.0
        days = int(facts.get("days") or state.fact("days", 2))

        # An empty at-risk search usually means the window or the zone filter
        # was too tight — widen both before concluding there is nothing.
        if failed_action in ("find_at_risk", "expiring_soon"):
            return [AgentTask(
                agent="inventory", action="find_at_risk",
                params={"days": min(days + 5, 14), "zone": None, "product": None},
                rationale=(f"Nothing found within {days} day(s)"
                           + (f" in {zone}" if zone else "")
                           + f" — widening to {min(days + 5, 14)} days across all zones"),
                priority=5,
            )]

        # A narrow inventory search: drop the zone constraint first, since
        # geography is the filter most likely to have been over-applied.
        if failed_action == "search_inventory":
            if zone:
                return [AgentTask(
                    agent="inventory", action="search_inventory",
                    params={"product": product, "zone": None},
                    rationale=f"No {product or 'stock'} in {zone} — searching every zone",
                    priority=5,
                )]
            return [AgentTask(
                agent="inventory", action="inventory_summary", params={},
                rationale="Specific search found nothing — reporting the overall position",
                priority=5,
            )]

        # No sellers matched: relax the radius and the grade floor, which are
        # the two hard constraints in the matcher.
        if failed_action == "match_sellers":
            return [AgentTask(
                agent="recommendation", action="match_sellers",
                params={"product": product, "quantity": quantity, "zone": None,
                        "relax_constraints": True},
                rationale="No seller met the distance and grade constraints — "
                          "widening the radius and accepting a lower grade",
                priority=5,
            )]

        # No buyer cleared the constraints: fall back to a pricing action, since
        # a sharper discount is the other lever for moving the same stock.
        if failed_action == "recommend_buyers":
            return [AgentTask(
                agent="recommendation", action="recommend_pricing",
                params={"product": product},
                rationale="No buyer matched — recommending a sharper discount instead",
                priority=5,
            )]

        # A missing or untrained forecast artefact must not stall the run: the
        # recommendation layer degrades to historical averages without it.
        if failed_agent == "forecast":
            return [AgentTask(
                agent="recommendation", action="recommend_actions",
                params={"top_n": 5, "use_historical_average": True},
                rationale="Forecast unavailable — proceeding on historical daily "
                          "averages instead",
                priority=5,
            )]

        # Retrieval found nothing: if a product was named, the question is
        # probably about that stock rather than about policy.
        if failed_agent == "rag" and product:
            return [AgentTask(
                agent="inventory", action="search_inventory",
                params={"product": product, "zone": zone},
                rationale=f"Knowledge base had no answer — reporting live "
                          f"{product} stock instead",
                priority=5,
            )]

        return []


__all__ = ["PlannerAgent", "ACTION_CATALOGUE", "INTENT_KEYWORDS",
           "AMBIGUITY_THRESHOLD"]