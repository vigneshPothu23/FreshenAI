#!/usr/bin/env python3
"""Sprint 4 knowledge base builder.

Loads the document corpus from ``data/documents``, chunks it, and builds the
vector index under ``PATHS.vectorstore_dir``.

This script owns no RAG logic of its own. Loading, chunking, embedding and
indexing all live in ``freshsense.rag``; the script's whole job is to call those
components in order, fail clearly when the corpus cannot support an index, and
report what was built.

Two things it deliberately refuses to do:

* **Instantiate ``VectorStore`` directly.** It is an abstract base class — the
  concrete implementation is chosen by ``build_vector_store()``, which prefers
  Chroma and falls back to the in-process NumPy index. Bypassing the factory
  would both fail at runtime and discard that fallback.
* **Build an index from an empty corpus.** ``DocumentLoader.load_all()`` warns
  and returns an empty list rather than raising, so a silent success here would
  leave the retriever with an index containing nothing to retrieve — and the
  chatbot would refuse every question with no indication why.

Usage::

    python scripts/build_knowledge_base.py
    python scripts/build_knowledge_base.py --vector-store numpy
    python scripts/build_knowledge_base.py --embedding tfidf
    python scripts/build_knowledge_base.py --rebuild
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

try:                                   # normal path: scripts/_bootstrap.py
    import _bootstrap  # noqa: F401    (puts src/ on sys.path before freshsense)
except ImportError:                    # pragma: no cover - stand-alone fallback
    _here = Path(__file__).resolve()
    for _candidate in (_here.parent, *_here.parents):
        if (_candidate / "src" / "freshsense").is_dir():
            sys.path.insert(0, str(_candidate / "src"))
            break

from freshsense.exceptions import FreshSenseError
from freshsense.logging_config import get_logger
from freshsense.paths import PATHS
from freshsense.rag.document_loader import Chunker, DocumentLoader
from freshsense.rag.vectorstore import build_embedding_backend, build_vector_store

LOG = get_logger("scripts.build_knowledge_base")

RULE = "=" * 78


class EmptyCorpusError(FreshSenseError):
    """Raised when the corpus cannot support an index.

    Distinct from a genuine failure: nothing is broken, the corpus is simply
    absent. The message names the directory and the accepted file types so the
    fix is obvious without reading the loader.
    """


def build_knowledge_base(
    *,
    documents_dir: Path | None = None,
    vector_store: str | None = None,
    embedding: str | None = None,
    rebuild: bool = False,
) -> dict[str, Any]:
    """Load, chunk and index the knowledge corpus.

    Args:
        documents_dir: Corpus location. ``None`` uses ``PATHS.documents_dir``,
            which is the supported case; the argument exists so tests can point
            at a fixture without touching the real corpus.
        vector_store: Store preference forwarded to ``build_vector_store``.
            ``None`` lets the factory read ``SETTINGS.vectorstore_backend``.
        embedding: Backend preference forwarded to ``build_embedding_backend``.
            ``None`` lets the factory read ``SETTINGS.embedding_backend``.
        rebuild: Clear any existing index before building. The concrete stores
            already replace their contents, so this matters mainly when
            switching backends and leaving a stale index of the other kind
            behind.

    Returns:
        Build statistics: document and chunk counts, the embedding backend and
        vector store actually selected, the index size and a per-document
        breakdown.

    Raises:
        EmptyCorpusError: If no document loads, or if chunking yields nothing.
    """
    loader = DocumentLoader(documents_dir)
    corpus_dir = loader.documents_dir

    LOG.info("Loading documents from %s", PATHS.relative(corpus_dir))
    documents = loader.load_all()

    if not documents:
        # load_all() only warns. Failing here is what stops an empty index from
        # being written and then silently refusing every question at query time.
        discovered = loader.discover()
        detail = (
            f"{len(discovered)} file(s) were found but none could be read"
            if discovered else "the directory is empty"
        )
        raise EmptyCorpusError(
            f"No documents loaded from {PATHS.relative(corpus_dir)} — {detail}. "
            f"Add corpus files with one of these extensions: "
            f"{', '.join(sorted(DocumentLoader.SUPPORTED))}. No index was built."
        )

    LOG.info("Chunking %d document(s)", len(documents))
    chunker = Chunker()
    chunks = chunker.chunk_all(documents)

    if not chunks:
        raise EmptyCorpusError(
            f"{len(documents)} document(s) loaded but produced no chunk. Every "
            f"passage fell below the minimum chunk size ({chunker.min_chars} "
            f"characters). Add longer prose to "
            f"{PATHS.relative(corpus_dir)}, or lower rag.min_chunk_chars. "
            f"No index was built."
        )

    # The factories resolve their preference from SETTINGS when given None, so
    # configuration stays the single source and nothing is hardcoded here.
    backend = build_embedding_backend(embedding)
    store = build_vector_store(vector_store)

    if rebuild:
        LOG.info("Clearing the existing %s index before rebuilding", store.name)
        store.clear()

    # build() fits the backend on the corpus itself, so an unfitted backend is
    # exactly what it expects.
    LOG.info("Building the %s index with %s embeddings", store.name, backend.name)
    indexed = store.build(chunks, backend)

    per_document = [
        {
            "title": document.title,
            "source": document.source,
            "type": document.doc_type,
            "words": document.word_count,
            "chunks": sum(1 for c in chunks if c.document_title == document.title),
        }
        for document in documents
    ]

    return {
        "documents": len(documents),
        "chunks": len(chunks),
        "indexed": indexed,
        "index_size": store.size,
        "embedding_backend": backend.name,
        "embedding_dimension": getattr(backend, "dimension", 0),
        "vector_store": store.name,
        "corpus_dir": PATHS.relative(corpus_dir),
        "index_dir": PATHS.relative(PATHS.vectorstore_dir),
        "avg_chunk_tokens": round(
            sum(c.token_estimate for c in chunks) / len(chunks), 1
        ),
        "per_document": per_document,
    }


def _report(stats: dict[str, Any]) -> None:
    """Print the build summary."""
    print()
    print(RULE)
    print("  KNOWLEDGE BASE BUILT")
    print(RULE)
    print(f"  Corpus            : {stats['corpus_dir']}")
    print(f"  Index             : {stats['index_dir']}")
    print(f"  Documents         : {stats['documents']}")
    print(f"  Chunks            : {stats['chunks']} "
          f"(avg ~{stats['avg_chunk_tokens']} tokens)")
    print(f"  Vectors indexed   : {stats['indexed']} "
          f"(store reports {stats['index_size']})")
    print(f"  Embedding backend : {stats['embedding_backend']}"
          + (f" (dim {stats['embedding_dimension']})"
             if stats["embedding_dimension"] else ""))
    print(f"  Vector store      : {stats['vector_store']}")
    print()
    print("  Per document:")
    for row in stats["per_document"]:
        print(f"    {row['title'][:44]:<46} {row['words']:>6} words  "
              f"{row['chunks']:>4} chunk(s)  [{row['type']}]")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the FreshSense RAG knowledge base."
    )
    parser.add_argument(
        "--vector-store", choices=["auto", "chroma", "numpy"], default=None,
        help="Override the configured vector store (default: from settings)",
    )
    parser.add_argument(
        "--embedding",
        choices=["auto", "sentence_transformers", "sentence-transformers", "tfidf"],
        default=None,
        help="Override the configured embedding backend (default: from settings)",
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Clear any existing index before building",
    )
    args = parser.parse_args()

    try:
        stats = build_knowledge_base(
            vector_store=args.vector_store,
            embedding=args.embedding,
            rebuild=args.rebuild,
        )
    except EmptyCorpusError as exc:
        LOG.error("%s", exc)
        return 1
    except FreshSenseError as exc:
        LOG.error("Knowledge base build failed: %s", exc)
        return 1

    _report(stats)
    print("  Next: the retriever will load this index on first use.")
    return 0


if __name__ == "__main__":
    sys.exit(main())