"""Sprint 4 — retrieval, prompt assembly and the RAG chain.

The pipeline:

    query → embed → top-k retrieve → similarity threshold → assemble prompt
          → [+ live inventory facts] → ask_llm() → answer + visible sources → log

Two properties distinguish this from a naive implementation:

**Refusal.** When the best retrieved passage scores below the configured
threshold, the corpus does not contain the answer and the assistant says so
rather than generating. A system that knows what it does not know is worth more
than one that always answers.

**Hybrid grounding.** The prompt can carry both document passages *and* live
structured facts from SQLite. "Paneer keeps five days refrigerated [FSSAI], and
you currently hold 12 kg expiring in 2 days [inventory]" is an answer neither
source produces alone — and it is what makes the chatbot part of the application
rather than a bolt-on.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from freshsense.config import SETTINGS
from freshsense.db.repository import ChatRepository, DocumentRepository
from freshsense.logging_config import get_logger
from freshsense.rag.llm_wrapper import LLMResponse, ask_llm, get_llm
from freshsense.rag.document_loader import Chunk, Chunker, DocumentLoader
from freshsense.rag.vectorstore import (SearchHit, VectorStore,
                                        build_embedding_backend,
                                        build_vector_store)

LOG = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are the FreshSense AI assistant for a perishable inventory platform "
    "operating in Chennai. Answer strictly from the CONTEXT provided. If the "
    "context does not contain the answer, say so plainly rather than guessing. "
    "Be concise: two to four sentences. Prefer concrete numbers, temperatures "
    "and durations when the context supplies them. Never invent a regulation, "
    "a threshold or a figure."
)


# ══════════════════════════════════════════════════════════════════════════
@dataclass
class RAGAnswer:
    """A grounded answer plus everything needed to audit it."""

    question: str
    answer: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    refused: bool = False
    max_similarity: float = 0.0
    provider: str = ""
    latency_ms: int = 0
    used_live_data: bool = False

    @property
    def is_grounded(self) -> bool:
        return bool(self.sources) and not self.refused

    def as_dict(self) -> dict[str, Any]:
        return {
            "question": self.question, "answer": self.answer,
            "sources": self.sources, "refused": self.refused,
            "max_similarity": round(self.max_similarity, 4),
            "provider": self.provider, "latency_ms": self.latency_ms,
            "used_live_data": self.used_live_data,
        }


# ══════════════════════════════════════════════════════════════════════════
class Retriever:
    """Owns the index lifecycle and nearest-neighbour search."""

    def __init__(self, store: VectorStore | None = None) -> None:
        self.config = SETTINGS.rag
        self.top_k = int(self.config.get("top_k", 4))
        self.threshold = float(self.config.get("similarity_threshold", 0.12))
        self.store = store or build_vector_store()
        self._ready = False

    # ── index construction ────────────────────────────────────────────
    def build_index(self, *, persist_metadata: bool = True) -> dict[str, Any]:
        """Load the corpus, chunk it, embed it and persist both index and provenance."""
        documents = DocumentLoader().load_all()
        if not documents:
            self._ready = False
            return {"documents": 0, "chunks": 0, "backend": "", "store": self.store.name}

        chunks = Chunker().chunk_all(documents)
        if not chunks:
            self._ready = False
            return {"documents": len(documents), "chunks": 0,
                    "backend": "", "store": self.store.name}

        # Provenance is written first so chunk_id is available to the index.
        if persist_metadata:
            repository = DocumentRepository()
            repository.clear()
            by_title = {d.title: d for d in documents}
            document_ids: dict[str, int] = {}
            for document in documents:
                document_ids[document.title] = repository.add_document(
                    source=document.source, title=document.title,
                    uri=document.uri, doc_type=document.doc_type,
                    license_=document.license, checksum=document.checksum,
                )
            repository.add_chunks([
                {
                    "document_id": document_ids.get(c.document_title, 0),
                    "chunk_index": c.chunk_index,
                    "section": c.section,
                    "text": c.text,
                    "token_count": c.token_estimate,
                    "vector_id": f"vec_{i}",
                }
                for i, c in enumerate(chunks)
            ])
            stored = repository.all_chunks()
            for chunk, (_, row) in zip(chunks, stored.iterrows()):
                chunk.chunk_id = int(row["chunk_id"])

        backend = build_embedding_backend()
        count = self.store.build(chunks, backend)
        self._ready = count > 0

        summary = {
            "documents": len(documents),
            "chunks": count,
            "backend": backend.name,
            "store": self.store.name,
            "avg_chunk_tokens": round(
                sum(c.token_estimate for c in chunks) / max(len(chunks), 1), 1
            ),
        }
        LOG.info("Knowledge index built: %s", summary)
        return summary

    def load(self) -> bool:
        self._ready = self.store.load()
        return self._ready

    @property
    def is_ready(self) -> bool:
        return self._ready and self.store.size > 0

    @property
    def size(self) -> int:
        return self.store.size

    # ── search ────────────────────────────────────────────────────────
    def search(self, query: str, top_k: int | None = None) -> list[SearchHit]:
        if not self.is_ready and not self.load():
            LOG.warning("Retrieval attempted before the index was built")
            return []
        hits = self.store.search(query, top_k or self.top_k)
        return [h for h in hits if h.score > 0]

    def stats(self) -> dict[str, Any]:
        return {
            "ready": self.is_ready,
            "chunks": self.size,
            "store": self.store.name,
            "top_k": self.top_k,
            "similarity_threshold": self.threshold,
            **DocumentRepository().stats(),
        }


# ══════════════════════════════════════════════════════════════════════════
class PromptBuilder:
    """Assembles the final prompt from retrieved passages and live facts."""

    @staticmethod
    def build(
        question: str,
        hits: Sequence[SearchHit],
        live_facts: str = "",
    ) -> str:
        blocks = [
            f"[{i}] ({hit.source})\n{hit.text}"
            for i, hit in enumerate(hits, start=1)
        ]
        context = "\n\n".join(blocks)

        sections = [f"CONTEXT:\n{context}"]
        if live_facts:
            sections.append(
                "LIVE INVENTORY DATA (current, from the application database):\n"
                f"{live_facts}"
            )
        sections.append(f"QUESTION: {question}")
        return "\n\n".join(sections)

    @staticmethod
    def refusal(question: str) -> str:
        return (
            "I could not find that in the FreshSense knowledge base. The corpus "
            "covers food-safety and storage regulation, product shelf life, "
            "platform pricing and grading policy, and Chennai market context. "
            "Try rephrasing, or ask about one of those areas."
        )


# ══════════════════════════════════════════════════════════════════════════
class RAGChain:
    """End-to-end retrieval-augmented generation with grounding and refusal."""

    def __init__(self, retriever: Retriever | None = None) -> None:
        self.retriever = retriever or Retriever()
        self.builder = PromptBuilder()
        self.config = SETTINGS.rag
        self.threshold = float(self.config.get("similarity_threshold", 0.12))
        self.chat_repository = ChatRepository()

    def answer(
        self,
        question: str,
        *,
        top_k: int | None = None,
        live_facts: str = "",
        session_id: str = "",
        persist: bool = True,
    ) -> RAGAnswer:
        """Answer a question against the corpus, refusing when unsupported."""
        started = time.perf_counter()
        question = question.strip()
        if not question:
            return RAGAnswer(question="", answer="Please enter a question.",
                             refused=True)

        hits = self.retriever.search(question, top_k)
        max_similarity = max((h.score for h in hits), default=0.0)

        # Refusal path: the corpus does not support an answer.
        if not hits or max_similarity < self.threshold:
            latency = int((time.perf_counter() - started) * 1000)
            result = RAGAnswer(
                question=question,
                answer=self.builder.refusal(question),
                sources=[h.as_dict() for h in hits[:2]],
                refused=True,
                max_similarity=max_similarity,
                provider=get_llm().active_provider,
                latency_ms=latency,
            )
            if persist and session_id:
                self._persist(session_id, result)
            LOG.info("RAG refusal (max similarity %.3f < %.3f): %s",
                     max_similarity, self.threshold, question[:60])
            return result

        prompt = self.builder.build(question, hits, live_facts)
        response: LLMResponse = ask_llm(
            prompt,
            system=SYSTEM_PROMPT,
            context=[h.as_dict() for h in hits],
        )

        latency = int((time.perf_counter() - started) * 1000)
        result = RAGAnswer(
            question=question,
            answer=response.text.strip() or self.builder.refusal(question),
            sources=[h.as_dict() for h in hits],
            refused=False,
            max_similarity=max_similarity,
            provider=response.provider,
            latency_ms=latency,
            used_live_data=bool(live_facts),
        )
        if persist and session_id:
            self._persist(session_id, result)
        return result

    def _persist(self, session_id: str, result: RAGAnswer) -> None:
        try:
            self.chat_repository.append(
                session_id=session_id, role="user", content=result.question
            )
            self.chat_repository.append(
                session_id=session_id, role="assistant", content=result.answer,
                sources=[s.get("source") for s in result.sources],
                latency_ms=result.latency_ms, provider=result.provider,
                refused=result.refused,
            )
        except Exception as exc:                         # pragma: no cover
            LOG.warning("Could not persist chat turn: %s", exc)

    # ── evaluation ────────────────────────────────────────────────────
    def evaluate(self, cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
        """Score the retriever and the answering path against fixed test cases.

        Reported metrics:

        * ``retrieval_accuracy`` — was the expected source document in top-k?
        * ``groundedness`` — did the answer contain the expected keywords?
        * ``refusal_accuracy`` — did out-of-corpus questions get refused?
        """
        rows: list[dict[str, Any]] = []
        for case in cases:
            question = case["question"]
            expected_source = str(case.get("expected_source", "")).lower()
            expected_keywords = [str(k).lower() for k in case.get("expected_keywords", [])]
            should_refuse = bool(case.get("should_refuse", False))

            result = self.answer(question, persist=False)
            sources = " ".join(s.get("source", "") for s in result.sources).lower()
            answer_text = result.answer.lower()
            matched = sum(1 for k in expected_keywords if k in answer_text)

            rows.append({
                "question": question[:60],
                "source_correct": (expected_source in sources) if expected_source else None,
                "keywords_found": f"{matched}/{len(expected_keywords)}"
                                  if expected_keywords else "—",
                "grounded": matched > 0 if expected_keywords else not result.refused,
                "refused": result.refused,
                "refusal_correct": result.refused == should_refuse,
                "max_similarity": round(result.max_similarity, 3),
                "latency_ms": result.latency_ms,
            })

        import pandas as pd
        frame = pd.DataFrame(rows)
        retrieval = frame["source_correct"].dropna()
        return {
            "cases": frame,
            "metrics": {
                "n_cases": len(frame),
                "retrieval_accuracy": round(float(retrieval.mean() * 100), 1)
                                      if len(retrieval) else 0.0,
                "groundedness": round(float(frame["grounded"].mean() * 100), 1),
                "refusal_accuracy": round(float(frame["refusal_correct"].mean() * 100), 1),
                "avg_latency_ms": int(frame["latency_ms"].mean()),
            },
        }


_CHAIN: RAGChain | None = None


def get_rag_chain() -> RAGChain:
    """Return the process-wide RAG chain, loading the index on first use."""
    global _CHAIN
    if _CHAIN is None:
        _CHAIN = RAGChain()
        _CHAIN.retriever.load()
    return _CHAIN


__all__ = [
    "RAGAnswer", "Retriever", "PromptBuilder", "RAGChain", "get_rag_chain",
    "SYSTEM_PROMPT",
]