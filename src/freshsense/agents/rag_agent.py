"""Sprint 5 — the knowledge agent.

Answers questions from the document corpus through the Sprint 4
:class:`RAGChain`, and — where the question touches stock the platform actually
holds — grounds the answer in live database facts alongside the retrieved
passages.

That hybrid step is what makes retrieval part of the application rather than a
bolt-on. *"Paneer keeps five days at 2–4 °C [FSSAI], and you are holding 47 kg
across 6 batches, the oldest expiring today"* is an answer neither source
produces alone: the corpus knows the rule, the database knows the situation, and
only the join is operationally useful.

Three failure modes are kept distinct, because the right response differs for
each:

* **No index built** — a setup problem. Reported with the command that fixes it.
* **Corpus does not cover the question** — the chain refuses. Returned as
  ``empty`` so the planner can route to live inventory instead.
* **Retrieval succeeded but scored poorly** — answered, with the weak grounding
  stated openly rather than hidden behind confident prose.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from freshsense.agents.base import AgentResult, BaseAgent
from freshsense.agents.state import AgentState, AgentTask
from freshsense.config import SETTINGS
from freshsense.db.repository import InventoryRepository
from freshsense.logging_config import get_logger

LOG = get_logger(__name__)

#: Below this similarity the answer is served with an explicit weak-grounding
#: caveat. It sits above the chain's own refusal threshold, so this band covers
#: answers that retrieved something but not something convincing.
WEAK_GROUNDING_SIMILARITY = 0.25

#: Topics the policy lookup recognises, mapped to a retrieval phrasing that
#: matches how the corpus actually words them. A user asking about "discounts"
#: should reach the decay-pricing section even though the corpus never uses that
#: word in isolation.
POLICY_TOPICS: dict[str, str] = {
    "pricing": "dynamic decay pricing discount schedule near expiry",
    "grading": "quality grade A B C assignment criteria",
    "donation": "donation NGO surplus food redistribution policy",
    "storage": "cold chain storage temperature bands for perishables",
    "licensing": "FSSAI licence registration requirement for food business",
    "disputes": "dispute resolution refund buyer seller policy",
    "labelling": "labelling expiry date marking requirements",
    "hygiene": "food safety hygiene handling requirements",
}


class RAGAgent(BaseAgent):
    """Answers knowledge questions, grounded in documents and live stock."""

    name = "rag"
    description = (
        "Answers food-safety, storage, regulatory and platform-policy questions "
        "from the document corpus, grounded in live inventory where relevant."
    )
    supported_actions = ("answer_question", "lookup_policy")

    def __init__(
        self,
        *,
        chain: Any = None,
        inventory: InventoryRepository | None = None,
    ) -> None:
        super().__init__()
        self._chain = chain
        self._chain_loaded = chain is not None
        self._inventory = inventory
        self.currency = SETTINGS.currency

    # ── lazily-resolved dependencies ──────────────────────────────────
    @property
    def inventory(self) -> InventoryRepository:
        if self._inventory is None:
            self._inventory = InventoryRepository()
        return self._inventory

    @property
    def chain(self) -> Any:
        """The RAG chain, or ``None`` when the index cannot be loaded.

        Deferred because constructing it loads an embedding backend and a vector
        index off disk — work an agent should not do merely to be listed.
        """
        if not self._chain_loaded:
            self._chain_loaded = True
            try:
                from freshsense.rag.chain import get_rag_chain

                self._chain = get_rag_chain()
            except Exception as exc:                     # pragma: no cover
                LOG.warning("RAG chain unavailable: %s", exc)
                self._chain = None
        return self._chain

    @property
    def is_ready(self) -> bool:
        chain = self.chain
        return chain is not None and chain.retriever.is_ready

    @staticmethod
    def _no_index_message() -> str:
        return (
            "The knowledge index has not been built. Run "
            "`python scripts/build_knowledge_base.py` to ingest the corpus and "
            "create the vector index."
        )

    # ── live grounding ────────────────────────────────────────────────
    def _live_facts(self, product: str | None, zone: str | None) -> str:
        """Render current stock for the named product as prompt-ready text.

        Deliberately terse. This text enters the prompt, so it carries the
        handful of figures that change an answer — quantity, urgency, price
        band — and nothing that would merely consume context.
        """
        if not product:
            return ""

        try:
            batches = self.inventory.search(
                product=product, zone=zone, active_only=True
            )
        except Exception as exc:                         # pragma: no cover
            LOG.debug("Could not read live stock for grounding: %s", exc)
            return ""

        if batches.empty:
            scope = f" in {zone}" if zone else ""
            return f"Current stock of {product}{scope}: none on hand."

        quantity = float(batches["quantity_available"].sum())
        unit = str(batches["unit"].iloc[0]) if "unit" in batches.columns else "units"
        soonest = int(batches["days_to_expiry"].min())
        grades = ", ".join(sorted(batches["quality_grade"].dropna().unique()))
        low = float(batches["effective_price"].min())
        high = float(batches["effective_price"].max())
        at_risk = int((batches["days_to_expiry"] <= 2).sum())
        scope = f" in {zone}" if zone else ""

        return (
            f"Current stock of {product}{scope}: {quantity:,.0f} {unit} across "
            f"{len(batches)} batch(es) from "
            f"{int(batches['seller_id'].nunique())} seller(s). "
            f"Earliest expiry in {soonest} day(s); {at_risk} batch(es) expire "
            f"within 2 days. Grades present: {grades}. "
            f"Price range {self.currency}{low:.0f}–{self.currency}{high:.0f} "
            f"per {unit}."
        )

    @staticmethod
    def _cite(sources: list[dict[str, Any]], limit: int = 3) -> str:
        """Deduplicated source list, preserving retrieval order."""
        seen: list[str] = []
        for source in sources:
            label = str(source.get("source") or source.get("title") or "").strip()
            if label and label not in seen:
                seen.append(label)
            if len(seen) >= limit:
                break
        return ", ".join(seen)

    def _sources_frame(self, sources: list[dict[str, Any]]) -> pd.DataFrame:
        """Retrieved passages as a table, for the citation panel in the UI."""
        if not sources:
            return pd.DataFrame(columns=["source", "score", "excerpt"])
        return pd.DataFrame([
            {
                "source": s.get("source") or s.get("title", ""),
                "score": round(float(s.get("score", 0)), 3),
                "excerpt": str(s.get("text", ""))[:280],
            }
            for s in sources
        ])

    # ── actions ───────────────────────────────────────────────────────
    def action_answer_question(
        self, state: AgentState, task: AgentTask
    ) -> AgentResult:
        """Answer a free-form question, grounded in documents and live stock."""
        question = str(
            task.param("question") or state.query or ""
        ).strip()
        if not question:
            return self.fail("no question supplied")

        if not self.is_ready:
            return self.fail(self._no_index_message(),
                             tool_called="Retriever.load")

        product = task.param("product") or state.fact("product")
        zone = task.param("zone") or state.fact("zone")
        live_facts = self._live_facts(product, zone)

        answer = self.chain.answer(
            question,
            top_k=task.param("top_k"),
            live_facts=live_facts,
            session_id=state.session_id or state.run_id,
        )

        # The corpus genuinely does not cover this. Returned as empty rather
        # than as a failure, because the planner's recovery — reading live
        # inventory for the named product — is often the better answer anyway.
        if answer.refused:
            return self.empty(
                f"The knowledge base does not cover that question "
                f"(best passage scored {answer.max_similarity:.2f}, below the "
                f"{self.chain.threshold:.2f} grounding threshold).",
                tool_called="RAGChain.answer",
            )

        citations = self._cite(answer.sources)
        narrative = answer.answer.strip()

        if citations:
            narrative += f" [Sources: {citations}]"
        if live_facts:
            # Append the facts rather than asserting the answer reflects them.
            # The offline stub composes only from retrieved passages, so a claim
            # that live data informed the wording would be false under it. This
            # way the joint answer is delivered under every provider.
            narrative += f"\n\n{live_facts}"
        if answer.max_similarity < WEAK_GROUNDING_SIMILARITY:
            narrative += (
                f" Note: the supporting passages are only loosely related "
                f"(best match {answer.max_similarity:.2f}), so treat this as "
                f"indicative and verify against the source document."
            )

        return self.ok(
            narrative,
            facts={
                "rag_answered": True,
                "rag_similarity": round(answer.max_similarity, 3),
                "rag_sources": [s.get("source") for s in answer.sources][:5],
                "rag_used_live_data": bool(live_facts),
                "rag_provider": answer.provider,
                "rag_weakly_grounded": answer.max_similarity < WEAK_GROUNDING_SIMILARITY,
            },
            artifacts={
                "rag_answer": answer,
                "rag_sources_table": self._sources_frame(answer.sources),
            },
            tool_called="RAGChain.answer",
        )

    def action_lookup_policy(
        self, state: AgentState, task: AgentTask
    ) -> AgentResult:
        """Retrieve platform or regulatory policy on a named topic.

        Distinct from a free-form question: the topic is expanded into the
        phrasing the corpus actually uses, and more passages are retrieved,
        because policy answers are judged on completeness rather than brevity.
        """
        if not self.is_ready:
            return self.fail(self._no_index_message(),
                             tool_called="Retriever.load")

        topic = str(task.param("topic") or "").strip().lower()
        question = str(task.param("question") or state.query or "").strip()

        if topic and topic in POLICY_TOPICS:
            query = POLICY_TOPICS[topic]
            label = topic
        elif topic:
            query = topic
            label = topic
        elif question:
            # Infer the topic from the question so a bare question still routes
            # to the right section of the corpus.
            lowered = question.lower()
            matched = next(
                (name for name in POLICY_TOPICS if name in lowered), None
            )
            query = POLICY_TOPICS.get(matched, question)
            label = matched or "general"
        else:
            return self.fail(
                "no topic or question supplied — name a policy area such as "
                f"{', '.join(sorted(POLICY_TOPICS)[:4])}"
            )

        answer = self.chain.answer(
            query,
            top_k=int(task.param("top_k", 6)),
            session_id=state.session_id or state.run_id,
        )

        if answer.refused:
            return self.empty(
                f"No policy guidance on '{label}' is present in the corpus. "
                f"Recognised areas: {', '.join(sorted(POLICY_TOPICS))}.",
                tool_called="RAGChain.answer",
            )

        citations = self._cite(answer.sources, limit=4)
        narrative = (
            f"Policy on {label}: {answer.answer.strip()}"
            + (f" [Sources: {citations}]" if citations else "")
        )

        return self.ok(
            narrative,
            facts={
                "policy_topic": label,
                "policy_similarity": round(answer.max_similarity, 3),
                "policy_sources": [s.get("source") for s in answer.sources][:5],
            },
            artifacts={
                "policy_answer": answer,
                "policy_sources_table": self._sources_frame(answer.sources),
            },
            tool_called="RAGChain.answer",
        )


__all__ = ["RAGAgent", "POLICY_TOPICS", "WEAK_GROUNDING_SIMILARITY"]