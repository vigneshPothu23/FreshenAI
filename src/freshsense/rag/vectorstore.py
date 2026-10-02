"""Sprint 4 — embeddings and vector storage.

Two backends sit behind one interface at each layer, selected automatically:

**Embeddings.** ``sentence-transformers`` (``all-MiniLM-L6-v2``) when installed,
TF-IDF otherwise. Dense embeddings capture semantic similarity — *"how long does
paneer keep?"* and *"paneer keeps five days refrigerated"* share almost no
vocabulary but sit close in embedding space. TF-IDF cannot do that, but at this
corpus size (a few hundred chunks) it performs competitively and it has zero
install risk.

**Vector store.** Chroma when installed, an in-process NumPy index otherwise.

The fallbacks are not decoration. An optional dependency failing to resolve must
never take out a mandatory capability, so the retrieval path is guaranteed to
work in a bare environment.

Vectors live in the vector store; chunk text and provenance live in SQLite,
joined on ``vector_id``. That separation means the index can be rebuilt without
losing lineage.
"""

from __future__ import annotations

import json
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from freshsense.config import SETTINGS
from freshsense.exceptions import VectorStoreError
from freshsense.logging_config import get_logger
from freshsense.paths import PATHS
from freshsense.rag.document_loader import Chunk

LOG = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# Embedding backends
# ══════════════════════════════════════════════════════════════════════════
class EmbeddingBackend(ABC):
    """Interface for turning text into vectors."""

    name: str = "abstract"
    dimension: int = 0

    @abstractmethod
    def fit(self, corpus: Sequence[str]) -> "EmbeddingBackend":
        """Prepare the backend against the corpus (a no-op for pretrained models)."""

    @abstractmethod
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return L2-normalised vectors, one row per input text."""

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


class SentenceTransformerBackend(EmbeddingBackend):
    """Dense semantic embeddings from a pretrained sentence encoder."""

    name = "sentence-transformers"

    def __init__(self, model_name: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        self.model_name = model_name or SETTINGS.embedding_model
        self.model = SentenceTransformer(self.model_name)
        self.dimension = int(self.model.get_sentence_embedding_dimension())
        LOG.info("Embedding backend: %s (dim=%d)", self.model_name, self.dimension)

    def fit(self, corpus: Sequence[str]) -> "SentenceTransformerBackend":
        return self                                      # pretrained

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(list(texts), normalize_embeddings=True,
                              show_progress_bar=False),
            dtype=np.float32,
        )


class TfidfBackend(EmbeddingBackend):
    """Lexical embeddings via TF-IDF. Zero external dependencies beyond sklearn."""

    name = "tfidf"

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: PLC0415

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
            max_features=20_000,
        )
        self._fitted = False

    def fit(self, corpus: Sequence[str]) -> "TfidfBackend":
        self.vectorizer.fit(list(corpus))
        self._fitted = True
        self.dimension = len(self.vectorizer.vocabulary_)
        LOG.info("Embedding backend: TF-IDF (vocabulary=%d)", self.dimension)
        return self

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not self._fitted:
            raise VectorStoreError("TF-IDF backend used before fit()")
        matrix = self.vectorizer.transform(list(texts)).toarray().astype(np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms


def build_embedding_backend(preference: str | None = None) -> EmbeddingBackend:
    """Select an embedding backend, degrading gracefully."""
    preference = (preference or SETTINGS.embedding_backend or "auto").lower()

    if preference in ("auto", "sentence_transformers", "sentence-transformers"):
        try:
            return SentenceTransformerBackend()
        except Exception as exc:
            if preference != "auto":
                LOG.warning("sentence-transformers unavailable (%s)", exc)
            LOG.info("Falling back to TF-IDF embeddings")
    return TfidfBackend()


# ══════════════════════════════════════════════════════════════════════════
# Vector stores
# ══════════════════════════════════════════════════════════════════════════
@dataclass
class SearchHit:
    """One retrieved passage with its similarity score."""

    text: str
    score: float
    title: str = ""
    source: str = ""
    section: str = ""
    chunk_id: int | None = None
    vector_id: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text, "score": round(float(self.score), 4),
            "title": self.title, "source": self.source or self.title,
            "section": self.section, "chunk_id": self.chunk_id,
            "vector_id": self.vector_id,
        }


class VectorStore(ABC):
    """Interface for vector persistence and nearest-neighbour search."""

    name: str = "abstract"

    @abstractmethod
    def build(self, chunks: Sequence[Chunk], backend: EmbeddingBackend) -> int: ...

    @abstractmethod
    def search(self, query: str, top_k: int = 4) -> list[SearchHit]: ...

    @abstractmethod
    def load(self) -> bool: ...

    @abstractmethod
    def clear(self) -> None: ...

    @property
    @abstractmethod
    def size(self) -> int: ...


class NumpyVectorStore(VectorStore):
    """In-process cosine-similarity index persisted with pickle.

    Adequate for corpora up to tens of thousands of chunks: a dense matrix
    multiply over a few hundred vectors is microseconds, and it removes an
    entire class of dependency risk.
    """

    name = "numpy"

    def __init__(self, store_dir: Path | None = None) -> None:
        self.store_dir = store_dir or PATHS.vectorstore_dir
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.store_dir / "numpy_index.pkl"
        self.vectors: np.ndarray | None = None
        self.chunks: list[Chunk] = []
        self.backend: EmbeddingBackend | None = None

    def build(self, chunks: Sequence[Chunk], backend: EmbeddingBackend) -> int:
        if not chunks:
            raise VectorStoreError("Cannot build an index from zero chunks")

        texts = [c.text for c in chunks]
        backend.fit(texts)
        self.vectors = backend.encode(texts)
        self.chunks = list(chunks)
        self.backend = backend

        for position, chunk in enumerate(self.chunks):
            chunk.vector_id = f"vec_{position}"

        with self.index_path.open("wb") as handle:
            pickle.dump(
                {"vectors": self.vectors, "chunks": self.chunks,
                 "backend": backend, "backend_name": backend.name},
                handle,
            )
        LOG.info("NumPy index built: %d vector(s), dim=%d, backend=%s",
                 len(self.chunks), self.vectors.shape[1], backend.name)
        return len(self.chunks)

    def load(self) -> bool:
        if not self.index_path.is_file():
            return False
        try:
            with self.index_path.open("rb") as handle:
                blob = pickle.load(handle)
            self.vectors = blob["vectors"]
            self.chunks = blob["chunks"]
            self.backend = blob["backend"]
            LOG.info("NumPy index loaded: %d vector(s), backend=%s",
                     len(self.chunks), blob.get("backend_name", "?"))
            return True
        except Exception as exc:                         # pragma: no cover
            LOG.error("Failed to load the NumPy index: %s", exc)
            return False

    def search(self, query: str, top_k: int = 4) -> list[SearchHit]:
        if self.vectors is None or self.backend is None or not self.chunks:
            return []
        query_vector = self.backend.encode([query])[0]
        similarities = self.vectors @ query_vector
        order = np.argsort(-similarities)[: max(top_k, 1)]
        return [
            SearchHit(
                text=self.chunks[i].text,
                score=float(similarities[i]),
                title=self.chunks[i].document_title,
                source=self.chunks[i].citation,
                section=self.chunks[i].section,
                chunk_id=self.chunks[i].chunk_id,
                vector_id=self.chunks[i].vector_id,
            )
            for i in order
        ]

    def clear(self) -> None:
        if self.index_path.exists():
            self.index_path.unlink()
        self.vectors, self.chunks, self.backend = None, [], None

    @property
    def size(self) -> int:
        return len(self.chunks)


class ChromaVectorStore(VectorStore):
    """Persistent Chroma collection.

    Chroma is preferred when installed because it is the recognisable
    "vector database" component and it scales past the point where an in-process
    matrix multiply stops being sensible.
    """

    name = "chroma"

    def __init__(self, store_dir: Path | None = None) -> None:
        import chromadb                                  # noqa: PLC0415
        from chromadb.config import Settings as ChromaSettings  # noqa: PLC0415

        self.store_dir = store_dir or PATHS.vectorstore_dir
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = str(
            SETTINGS.rag.get("collection_name", "freshsense_kb")
        )
        self.client = chromadb.PersistentClient(
            path=str(self.store_dir / "chroma"),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self.collection: Any = None
        self.backend: EmbeddingBackend | None = None
        self._backend_path = self.store_dir / "chroma_backend.pkl"

    def build(self, chunks: Sequence[Chunk], backend: EmbeddingBackend) -> int:
        if not chunks:
            raise VectorStoreError("Cannot build an index from zero chunks")

        texts = [c.text for c in chunks]
        backend.fit(texts)
        vectors = backend.encode(texts)
        self.backend = backend

        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )

        ids, metadatas = [], []
        for position, chunk in enumerate(chunks):
            vector_id = f"vec_{position}"
            chunk.vector_id = vector_id
            ids.append(vector_id)
            metadatas.append({
                "title": chunk.document_title,
                "source": chunk.citation,
                "section": chunk.section,
                "chunk_index": chunk.chunk_index,
                "chunk_id": chunk.chunk_id if chunk.chunk_id is not None else -1,
            })

        self.collection.add(
            ids=ids, documents=texts, metadatas=metadatas,
            embeddings=[v.tolist() for v in vectors],
        )
        with self._backend_path.open("wb") as handle:
            pickle.dump(backend, handle)

        LOG.info("Chroma collection '%s' built: %d vector(s), backend=%s",
                 self.collection_name, len(ids), backend.name)
        return len(ids)

    def load(self) -> bool:
        try:
            self.collection = self.client.get_collection(self.collection_name)
            if self._backend_path.is_file():
                with self._backend_path.open("rb") as handle:
                    self.backend = pickle.load(handle)
            LOG.info("Chroma collection loaded: %d vector(s)", self.size)
            return self.size > 0
        except Exception:
            return False

    def search(self, query: str, top_k: int = 4) -> list[SearchHit]:
        if self.collection is None or self.backend is None:
            return []
        query_vector = self.backend.encode([query])[0].tolist()
        result = self.collection.query(
            query_embeddings=[query_vector],
            n_results=max(top_k, 1),
            include=["documents", "metadatas", "distances"],
        )
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        hits: list[SearchHit] = []
        for text, metadata, distance in zip(documents, metadatas, distances):
            metadata = metadata or {}
            chunk_id = metadata.get("chunk_id", -1)
            hits.append(SearchHit(
                text=text,
                score=float(1.0 - distance),             # cosine distance -> similarity
                title=str(metadata.get("title", "")),
                source=str(metadata.get("source", "")),
                section=str(metadata.get("section", "")),
                chunk_id=None if chunk_id in (-1, None) else int(chunk_id),
            ))
        return hits

    def clear(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        if self._backend_path.exists():
            self._backend_path.unlink()
        self.collection = None

    @property
    def size(self) -> int:
        try:
            return int(self.collection.count()) if self.collection else 0
        except Exception:
            return 0


def build_vector_store(preference: str | None = None) -> VectorStore:
    """Select a vector store, degrading gracefully to the NumPy index."""
    preference = (preference or SETTINGS.vectorstore_backend or "auto").lower()
    if preference in ("auto", "chroma"):
        try:
            return ChromaVectorStore()
        except Exception as exc:
            if preference == "chroma":
                LOG.warning("Chroma unavailable (%s)", exc)
            LOG.info("Falling back to the in-process NumPy vector index")
    return NumpyVectorStore()


__all__ = [
    "EmbeddingBackend", "SentenceTransformerBackend", "TfidfBackend",
    "build_embedding_backend", "VectorStore", "NumpyVectorStore",
    "ChromaVectorStore", "build_vector_store", "SearchHit",
]