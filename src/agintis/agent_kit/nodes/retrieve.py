"""RetrieveNode: retrieval for the task's topic with a graded, recall-first filter fallback."""

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

from langgraph.runtime import Runtime

from agintis.agent_kit.nodes.filters import clauses_to_filter, normalize_clauses, strong_clauses
from agintis.agent_kit.nodes.sources import SourceLimits, select_sources
from agintis.agent_kit.nodes.state import FilterClause, SourceItem, TopicResult, TopicState
from agintis.agent_kit.runtime import AgentContext
from agintis.modules.knowledge.api import (
    DEFAULT_MAX_QUERY_CHARS,
    Citation,
    RagProfile,
    RetrievalResult,
    RetrievedHit,
)
from agintis.shared_kernel.criteria import FilterExpr
from agintis.shared_kernel.ids import TenantId

logger = logging.getLogger(__name__)

# The filter sets of the fallback ladder, strictest first.
type FilterLevel = Literal["full", "strong", "none"]
FULL: FilterLevel = "full"
STRONG: FilterLevel = "strong"
NONE: FilterLevel = "none"


class TopicRetrieval(Protocol):
    """What the node needs from the knowledge module (`RetrieveForTopic` satisfies it)."""

    async def execute(
        self,
        tenant_id: TenantId,
        profile: RagProfile,
        query: str,
        extra_filter: FilterExpr | None = None,
        query_vector: Sequence[float] | None = None,
    ) -> RetrievalResult: ...


@dataclass(frozen=True, slots=True)
class FilteredRetrieval:
    """`level`: the filter set that produced `result` (`none` also when nothing was extracted);
    `relaxed`: a stricter set found nothing first."""

    result: RetrievalResult
    level: FilterLevel
    relaxed: bool


async def retrieve_with_fallback(
    retrieval: TopicRetrieval,
    tenant_id: TenantId,
    profile: RagProfile,
    query: str,
    clauses: list[FilterClause],
    query_vector: Sequence[float] | None = None,
) -> FilteredRetrieval:
    """Retrieve with the extracted filters, relaxing them step by step on ZERO hits only.

    Extracted filters are hints, not constraints: a wrong or over-narrow value must cost another
    query, never an empty answer (recall first). The ladder, stopping at the first step with hits:
    1. `full`: every clause (a `gte`/`lte` pair on one field with the same value is one `eq`);
    2. `strong`: only clauses on numeric (int/float) and date fields, which a question states
       precisely and a model rarely invents, unlike free-text fields; skipped when it would
       equal the previous step;
    3. `none`: no filter.
    This judges emptiness only, not quality: that is the sufficiency loop of 3.5b.

    `query_vector` (the embedding of `query`, when already known) is handed to every step, so a
    relaxed search does not embed again either.
    """
    normalized = normalize_clauses(clauses)
    ladder: list[tuple[FilterLevel, FilterExpr | None]] = []
    full = clauses_to_filter(normalized)
    if full is not None:
        ladder.append((FULL, full))
        strong = clauses_to_filter(strong_clauses(normalized, profile))
        if strong is not None and strong != full:
            ladder.append((STRONG, strong))
    ladder.append((NONE, None))
    result = RetrievalResult(hits=())
    for index, (level, expr) in enumerate(ladder):
        result = await retrieval.execute(tenant_id, profile, query, expr, query_vector=query_vector)
        if result.hits:
            if index:
                logger.info("Profile %r: filters relaxed to level %r", profile.name, level)
            return FilteredRetrieval(result, level, relaxed=index > 0)
    if len(ladder) > 1:
        logger.info("Profile %r: no hits at any filter level", profile.name)
    return FilteredRetrieval(result, NONE, relaxed=len(ladder) > 1)


def locator_of(citation: Citation) -> str | None:
    """Where in the document the passage is: heading and page, when known."""
    parts = [citation.heading, f"p. {citation.page}" if citation.page is not None else None]
    text = ", ".join(part for part in parts if part)
    return text or None


def _evidence(hit: RetrievedHit) -> dict[str, Any]:
    """The relevance signals the hit really has (absent keys = unknown)."""
    evidence: dict[str, Any] = {}
    if hit.rerank_score is not None:
        evidence["rerank_score"] = hit.rerank_score
    if hit.lexical_rank is not None:
        evidence["lexical_rank"] = hit.lexical_rank
    if hit.dense_rank is not None:
        evidence["dense_rank"] = hit.dense_rank
    return evidence


def source_items(hits: Iterable[RetrievedHit]) -> list[SourceItem]:
    """The passages of the hits, in order, as plain state data."""
    return [
        cast(
            SourceItem,
            {
                "source": hit.citation.source_uri,
                "title": hit.citation.title or None,
                "locator": locator_of(hit.citation),
                "text": hit.chunk.chunk.text,
                "chunk_id": str(hit.citation.chunk_id),
                **_evidence(hit),
            },
        )
        for hit in hits
    ]


class RetrieveNode:
    """Retrieves for the task's topic profile as the tenant of the run (topic subgraph node).

    The query is cut to `max_query_chars` (the knowledge limit): a rewritten question that
    long is garbage anyway and must not fail the run. The hits go through `select_sources`
    (the spec limits, per topic) before they are stored. It emits NOTHING: `Sources` are emitted
    once by `SynthesizeNode`, so the client's `[n]` numbering is the one the model sees.
    Writes the topic's `topic_results` entry (possibly with no sources). When the router already
    embedded this very query (`runtime.context.query_vectors`, keyed by the cut query text) the
    vector is reused instead of embedding again.
    """

    def __init__(
        self,
        retrieval: TopicRetrieval,
        profiles: Mapping[str, RagProfile],
        limits: SourceLimits = SourceLimits(),  # noqa: B008 (frozen)
        max_query_chars: int = DEFAULT_MAX_QUERY_CHARS,
    ) -> None:
        if max_query_chars < 1:
            raise ValueError("max_query_chars must be at least 1")
        self._retrieval = retrieval
        self._profiles = dict(profiles)
        self._limits = limits
        self._max_query_chars = max_query_chars

    async def __call__(self, state: TopicState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
        topic = state["topic"]
        query = state["query"][: self._max_query_chars]
        found = await retrieve_with_fallback(
            self._retrieval,
            runtime.context.tenant_id,
            self._profiles[topic],
            query,
            state.get("filters") or [],
            runtime.context.query_vectors.get(query),
        )
        sources = select_sources(source_items(found.result.hits), self._limits)
        return {
            "topic_results": [TopicResult(topic=topic, sources=sources, filter_level=found.level)]
        }
