"""Sprint 4 — document loading and chunking.

The corpus is **external, unstructured prose**: food-safety regulation, storage
guidance and platform policy. This distinction matters. Vector-searching the
application's own database rows would be a SQL query with extra latency and
worse precision; retrieval earns its place only over text that cannot be
expressed as a table.

Chunking uses a sentence window of roughly four sentences with one sentence of
overlap. Larger chunks dilute the signal — one relevant sentence surrounded by
nine irrelevant ones — while smaller chunks sever a fact from the context that
gives it meaning. The section heading is retained inside each chunk, because
"Cold chain temperature bands: dairy requires 2–4 °C" retrieves far better than
a bare "dairy requires 2–4 °C".
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from freshsense.config import SETTINGS
from freshsense.logging_config import get_logger
from freshsense.paths import PATHS

LOG = get_logger(__name__)

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class Document:
    """A source document plus its provenance."""

    title: str
    source: str
    text: str
    uri: str = ""
    doc_type: str = "markdown"
    license: str = ""

    @property
    def checksum(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]

    @property
    def word_count(self) -> int:
        return len(self.text.split())


@dataclass
class Chunk:
    """A retrievable passage with enough metadata to cite it."""

    text: str
    document_title: str
    source: str
    section: str = ""
    chunk_index: int = 0
    document_id: int | None = None
    chunk_id: int | None = None
    vector_id: str = ""
    license: str = ""

    @property
    def token_estimate(self) -> int:
        """Approximate token count — four characters per token is close enough
        for chunk-size control and avoids a tokeniser dependency."""
        return max(1, len(self.text) // 4)

    @property
    def citation(self) -> str:
        return f"{self.document_title} › {self.section}" if self.section else self.document_title

    def as_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "title": self.document_title,
            "source": self.citation,
            "section": self.section,
            "chunk_index": self.chunk_index,
            "chunk_id": self.chunk_id,
            "vector_id": self.vector_id,
            "license": self.license,
        }


# ══════════════════════════════════════════════════════════════════════════
class DocumentLoader:
    """Reads the knowledge corpus from ``data/documents``.

    Markdown and plain text are read directly. PDFs are read through ``pypdf``
    when it is installed; when it is not, the file is skipped with a warning
    rather than failing the ingest, so one unreadable document cannot take down
    the whole index build.
    """

    SUPPORTED = {".md", ".markdown", ".txt", ".pdf"}

    def __init__(self, documents_dir: Path | None = None) -> None:
        self.documents_dir = documents_dir or PATHS.documents_dir
        self.documents_dir.mkdir(parents=True, exist_ok=True)

    def discover(self) -> list[Path]:
        return sorted(
            path for path in self.documents_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in self.SUPPORTED
        )

    def load_file(self, path: Path) -> Document | None:
        """Load one document, returning ``None`` when it cannot be read."""
        suffix = path.suffix.lower()
        try:
            if suffix == ".pdf":
                text = self._read_pdf(path)
                doc_type = "pdf"
            else:
                text = path.read_text(encoding="utf-8", errors="ignore")
                doc_type = "markdown" if suffix in {".md", ".markdown"} else "text"
        except Exception as exc:                         # pragma: no cover
            LOG.warning("Could not read %s: %s", PATHS.relative(path), exc)
            return None

        if not text or len(text.strip()) < 50:
            LOG.warning("Skipping %s — too short to be useful",
                        PATHS.relative(path))
            return None

        title, license_ = self._extract_front_matter(text, fallback=path.stem)
        return Document(
            title=title,
            source=path.stem,
            text=text,
            uri=PATHS.relative(path),
            doc_type=doc_type,
            license=license_,
        )

    @staticmethod
    def _read_pdf(path: Path) -> str:
        try:
            from pypdf import PdfReader
        except ImportError:                              # pragma: no cover
            LOG.warning("pypdf is not installed; skipping %s", path.name)
            return ""
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    @staticmethod
    def _extract_front_matter(text: str, fallback: str) -> tuple[str, str]:
        """Pull the title from the first heading and the licence from metadata."""
        title = fallback.replace("_", " ").replace("-", " ").title()
        license_ = ""
        for line in text.splitlines()[:20]:
            stripped = line.strip()
            if match := HEADING.match(stripped):
                if len(match.group(1)) == 1 and title == fallback.replace("_", " ").title():
                    title = match.group(2).strip()
            lowered = stripped.lower()
            if lowered.startswith(("license:", "licence:", "> license:", "> licence:")):
                license_ = stripped.split(":", 1)[-1].strip()
        return title, license_

    def load_all(self) -> list[Document]:
        documents: list[Document] = []
        for path in self.discover():
            document = self.load_file(path)
            if document:
                documents.append(document)
                LOG.info("Loaded document '%s' (%d words, %s)",
                         document.title, document.word_count, document.doc_type)
        if not documents:
            LOG.warning(
                "No documents found in %s. Run "
                "`python scripts/build_knowledge_base.py` to install the "
                "starter corpus.", PATHS.relative(self.documents_dir),
            )
        return documents


# ══════════════════════════════════════════════════════════════════════════
class Chunker:
    """Sentence-window chunker that preserves section context."""

    def __init__(
        self,
        window: int | None = None,
        overlap: int | None = None,
        min_chars: int | None = None,
    ) -> None:
        config = SETTINGS.rag
        self.window = int(window or config.get("chunk_size_sentences", 4))
        self.overlap = int(overlap if overlap is not None
                           else config.get("chunk_overlap_sentences", 1))
        self.min_chars = int(min_chars or config.get("min_chunk_chars", 80))
        self.step = max(self.window - self.overlap, 1)

    def split_sections(self, text: str) -> list[tuple[str, str]]:
        """Split markdown into ``(section_heading, body)`` pairs."""
        sections: list[tuple[str, str]] = []
        heading = ""
        body: list[str] = []

        for line in text.splitlines():
            if match := HEADING.match(line.strip()):
                if body:
                    sections.append((heading, " ".join(body).strip()))
                    body = []
                heading = match.group(2).strip()
            else:
                stripped = line.strip()
                if stripped and not stripped.startswith(("---", "|", ">")):
                    body.append(stripped)

        if body:
            sections.append((heading, " ".join(body).strip()))
        return [(h, b) for h, b in sections if b]

    def chunk_document(self, document: Document) -> list[Chunk]:
        """Produce overlapping sentence-window chunks for one document."""
        chunks: list[Chunk] = []
        index = 0

        for section, body in self.split_sections(document.text):
            sentences = [
                s.strip() for s in SENTENCE_SPLIT.split(body)
                if len(s.strip()) > 20
            ]
            if not sentences:
                continue

            position = 0
            while position < len(sentences):
                window = sentences[position:position + self.window]
                text = " ".join(window)
                if len(text) >= self.min_chars:
                    # Prefixing the heading materially improves retrieval: it
                    # gives the chunk the context its sentences assume.
                    prefixed = f"{section}: {text}" if section else text
                    chunks.append(Chunk(
                        text=prefixed,
                        document_title=document.title,
                        source=document.source,
                        section=section,
                        chunk_index=index,
                        license=document.license,
                    ))
                    index += 1
                position += self.step

        LOG.info("Chunked '%s' into %d passage(s)", document.title, len(chunks))
        return chunks

    def chunk_all(self, documents: Iterable[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for document in documents:
            chunks.extend(self.chunk_document(document))
        return chunks


__all__ = ["Document", "Chunk", "DocumentLoader", "Chunker"]