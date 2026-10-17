"""Application DTOs exchanged with parsers, chunkers and use cases."""

import hashlib
import json
import math
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final, Literal

from agintis.modules.knowledge.domain import (
    Citation,
    InvalidMetadataError,
    MetadataValue,
    ScoredChunk,
    check_storable_text,
    freeze_metadata,
    strip_unstorable,
)
from agintis.shared_kernel.ids import DocumentId

type DocumentScalar = str | int | float | bool | None
# A document-level fact supplied by the caller at ingest: a scalar or a flat tuple of scalars.
type DocumentValue = DocumentScalar | tuple[DocumentScalar, ...]
type DocumentMetadata = Mapping[str, DocumentValue]

MAX_DOCUMENT_METADATA_ITEMS = 64
MAX_DOCUMENT_METADATA_KEYS = 128
# A crawler row's longest legitimate values (a description) are a few KiB.
MAX_DOCUMENT_STR_CHARS = 8 * 1024


def freeze_document_metadata(metadata: Mapping[str, object]) -> DocumentMetadata:
    """Copy caller-supplied document metadata into a read-only mapping.

    This is the ingest boundary for caller data, so everything is checked here once: keys are
    non-empty strings, there are at most `MAX_DOCUMENT_METADATA_KEYS` of them, and every string
    (key or value) is storable text (no NUL, no lone surrogate) of at most
    `MAX_DOCUMENT_STR_CHARS` characters. Text later built from these values (heading prefixes,
    index cards) is therefore storable too. Values may be str, int, finite float, bool, None or
    a list/tuple of those (lists become tuples, nested lists are rejected). Whether a value is
    storable as chunk metadata is decided later, per declared field type.
    """
    if len(metadata) > MAX_DOCUMENT_METADATA_KEYS:
        raise InvalidMetadataError(
            f"Document metadata has {len(metadata)} keys (limit {MAX_DOCUMENT_METADATA_KEYS})"
        )
    frozen: dict[str, DocumentValue] = {}
    for key, value in metadata.items():
        if not isinstance(key, str) or not key:
            raise InvalidMetadataError(f"Document metadata keys must be non-empty strings: {key!r}")
        _check_document_str(f"Document metadata key {key[:40]!r}", key)
        if isinstance(value, list | tuple):
            if len(value) > MAX_DOCUMENT_METADATA_ITEMS:
                raise InvalidMetadataError(
                    f"Document metadata {key!r} has more than {MAX_DOCUMENT_METADATA_ITEMS} items"
                )
            items = tuple(_document_scalar(key, item) for item in value)
            frozen[key] = items
        else:
            frozen[key] = _document_scalar(key, value)
    return MappingProxyType(frozen)


def _check_document_str(label: str, value: str) -> None:
    if len(value) > MAX_DOCUMENT_STR_CHARS:
        raise InvalidMetadataError(f"{label} is longer than {MAX_DOCUMENT_STR_CHARS} characters")
    check_storable_text(label, value)


def _document_scalar(key: str, value: object) -> DocumentScalar:
    if isinstance(value, str):
        _check_document_str(f"Document metadata {key!r}", value)
        return value
    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise InvalidMetadataError(f"Document metadata {key!r} must be finite, got {value!r}")
        return value
    raise InvalidMetadataError(
        f"Document metadata {key!r} holds an unsupported {type(value).__name__} value"
    )


def metadata_hash(metadata: Mapping[str, DocumentValue]) -> str:
    """sha256 of the canonical JSON of the document metadata (keys sorted, tuples as lists).

    Part of the "is this ingest a no-op" test: the metadata feeds chunk text (heading prefixes)
    and chunk metadata, so a changed index row must reprocess a document whose bytes did not
    change.
    """
    canonical = json.dumps(
        {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in metadata.items()
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class SourceFile:
    """A document to ingest. `document_metadata` = facts the caller already knows about it
    (e.g. a crawler's index row: number, date, section); profiles read them through
    `document:<field>` metadata sources and heading templates."""

    uri: str
    mime_type: str
    content: bytes
    document_metadata: Mapping[str, DocumentValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        frozen = freeze_document_metadata(self.document_metadata)
        object.__setattr__(self, "document_metadata", frozen)


@dataclass(frozen=True)
class ParsedSection:
    text: str
    page: int | None = None
    headings: tuple[str, ...] = ()


SECTION_SEPARATOR: Final = "\n\n"

type CollectedValue = str | int


@dataclass(frozen=True)
class SectionSpan:
    """Where one section lives inside a `JoinedText`: `[start, end)` plus its provenance."""

    start: int
    end: int
    page: int | None
    headings: tuple[str, ...]


@dataclass(frozen=True)
class JoinedText:
    """Sections joined into one string, with an offset map back to each section.

    Lets a chunker work on continuous text (crossing page boundaries) and still tell which
    pages a chunk covers: `pages_between(start, end)`. Separators between sections belong to
    no span; an offset inside one maps to the section before it.
    """

    text: str
    spans: tuple[SectionSpan, ...]
    starts: tuple[int, ...]

    def _index_at(self, offset: int) -> int:
        return max(0, bisect_right(self.starts, offset) - 1)

    def span_at(self, offset: int) -> SectionSpan | None:
        if not self.spans or offset < 0 or offset > len(self.text):
            return None
        return self.spans[self._index_at(offset)]

    def pages_between(self, start: int, end: int) -> tuple[int, int] | None:
        """(first page, last page) of the text in `[start, end)`; None when it has no page.

        `ValueError` when `start > end`; None when the range leaves the text.
        """
        if start > end:
            raise ValueError(f"pages_between needs start <= end, got {start} > {end}")
        if not self.spans or start < 0 or end > len(self.text):
            return None
        first, last = self._index_at(start), self._index_at(max(start, end - 1))
        pages = [s.page for s in self.spans[first : last + 1] if s.page is not None]
        return (min(pages), max(pages)) if pages else None


def join_sections(
    sections: Sequence[ParsedSection], separator: str = SECTION_SEPARATOR
) -> JoinedText:
    parts: list[str] = []
    spans: list[SectionSpan] = []
    position = 0
    for section in sections:
        if spans:
            position += len(separator)
        spans.append(
            SectionSpan(position, position + len(section.text), section.page, section.headings)
        )
        parts.append(section.text)
        position += len(section.text)
    return JoinedText(
        text=separator.join(parts),
        spans=tuple(spans),
        starts=tuple(span.start for span in spans),
    )


@dataclass(frozen=True)
class ParsedDocument:
    """Parser output. `sections` keep pages/headings; `joined()` is the continuous-text view.

    `source_html` is the raw markup of an HTML source (HTML cleaners rebuild `sections` from
    it); `collected` holds document-level values cleaners extracted (e.g. keywords), keyed by
    name: what a profile's metadata reads as `parsed:<name>`. `document_metadata` is the
    caller's `SourceFile.document_metadata`, set by the ingest use case after parsing so
    chunkers (heading templates) and metadata sources (`document:<field>`) can read it.
    """

    title: str
    sections: tuple[ParsedSection, ...]
    source_html: str | None = None
    collected: Mapping[str, tuple[CollectedValue, ...]] = field(default_factory=dict)
    document_metadata: Mapping[str, DocumentValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "collected", MappingProxyType(dict(self.collected)))
        object.__setattr__(
            self, "document_metadata", MappingProxyType(dict(self.document_metadata))
        )

    def joined(self, separator: str = SECTION_SEPARATOR) -> JoinedText:
        return join_sections(self.sections, separator)


@dataclass(frozen=True)
class ChunkDraft:
    """One chunk before storage.

    `metadata` is final-form data the chunker knows (page, heading). `groups` are the regex
    named groups the chunk matched, in order, one value per merged unit (profile metadata
    `group:<name>` reads them; a scalar field takes the first value, a list field all of them).
    `body` is the text WITHOUT the heading prefix (extractors read it); None = same as `text`.
    """

    text: str
    metadata: Mapping[str, MetadataValue] = field(default_factory=dict)
    groups: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    body: str | None = None

    def __post_init__(self) -> None:
        # Text may be assembled from caller data (heading prefixes, index cards): whatever the
        # database cannot store is removed here, once, for every chunker.
        object.__setattr__(self, "text", strip_unstorable(self.text))
        if self.body is not None:
            object.__setattr__(self, "body", strip_unstorable(self.body))
        # Rejects reserved keys (InvalidMetadataError) and freezes the mapping.
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))
        object.__setattr__(self, "groups", MappingProxyType(dict(self.groups)))

    @property
    def body_text(self) -> str:
        return self.text if self.body is None else self.body


type IngestStatus = Literal["indexed", "skipped", "replaced"]


@dataclass(frozen=True)
class IngestResult:
    """`skipped`: identical content and profile already indexed (chunk_count = existing).

    `replaced`: a previous READY version of the same source/content was swapped out.
    `title_refreshed`: a skipped document got a new display title (no re-parse, no re-embedding).
    """

    document_id: DocumentId
    chunk_count: int
    status: IngestStatus
    title_refreshed: bool = False


@dataclass(frozen=True)
class RetrievedHit:
    """A scored chunk aligned with the citation built from its document.

    Scores on different scales are kept apart:
    * `similarity`: vector similarity from the store (cosine for normalized embeddings);
      `None` for a hybrid hit that only the lexical branch found.
    * `rerank_score`: cross-encoder score, `None` when no real reranker scored this hit
      (reranker "none", or retrieval degraded to the retrieval order).
    * `fused_score`, `dense_rank`, `lexical_rank`: hybrid retrieval evidence (RRF score and
      the 1-based rank in each branch, `None` where the branch did not return the chunk or the
      profile retrieves dense-only).

    ORDERING KEY: hits are ordered by `rerank_score` when present, else by `fused_score` (hybrid)
    or `similarity` (dense); `score` and `chunk.score` expose that key. Never compare scores
    across hits of different queries or across scales. In particular `score` is NOT comparable
    across retrieval modes: a hybrid RRF score (about 0.01-0.03) and a cosine similarity (0-1)
    answer different questions, so do not apply one similarity threshold to hits of profiles
    with different `retrieval_mode` or reranker.
    """

    chunk: ScoredChunk
    citation: Citation
    similarity: float | None
    rerank_score: float | None = None
    lexical_rank: int | None = None
    dense_rank: int | None = None
    fused_score: float | None = None

    @property
    def score(self) -> float:
        if self.rerank_score is not None:
            return self.rerank_score
        if self.fused_score is not None:
            return self.fused_score
        return self.similarity if self.similarity is not None else 0.0


@dataclass(frozen=True)
class RetrievalResult:
    hits: tuple[RetrievedHit, ...]
