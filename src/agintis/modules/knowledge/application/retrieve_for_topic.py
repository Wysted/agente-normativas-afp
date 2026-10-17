"""Use case: retrieval (dense or hybrid) scoped to a tenant and a RagProfile, with citations."""

import logging
from collections.abc import Mapping, Sequence

from langchain_core.embeddings import Embeddings

from agintis.modules.knowledge.application.dto import RetrievalResult, RetrievedHit
from agintis.modules.knowledge.application.errors import QueryTooLongError, UnknownStrategyError
from agintis.modules.knowledge.application.ports import Reranker
from agintis.modules.knowledge.application.rerankers import NO_RERANKER_KEY, NoopReranker
from agintis.modules.knowledge.application.retrieval_modes import (
    DENSE_RETRIEVAL_MODE,
    CandidateRequest,
    DenseRetrieval,
    RetrievalStrategy,
)
from agintis.modules.knowledge.domain import (
    ChunkRepository,
    Citation,
    DocumentRepository,
    DocumentStatus,
    RagProfile,
    strip_unstorable,
)
from agintis.shared_kernel.criteria import Criteria, Filter, FilterExpr
from agintis.shared_kernel.ids import TenantId

# Chunks of non-READY/dangling documents are dropped after the search, so ask for more than
# top_k to still return top_k hits when some are dropped.
DEFAULT_OVERFETCH_FACTOR = 2

# Upper bound of candidates handed to the reranker. Cross-encoders are expensive on CPU
# (~45 s for 16 texts), so only the best few vector hits are reranked; never fewer than top_k.
DEFAULT_RERANK_CANDIDATES = 8

# Longest accepted query, in characters (after removing unstorable characters).
DEFAULT_MAX_QUERY_CHARS = 2000

logger = logging.getLogger(__name__)


class RetrieveForTopic:
    def __init__(
        self,
        documents: DocumentRepository,
        chunks: ChunkRepository,
        embeddings: Embeddings,
        overfetch_factor: int = DEFAULT_OVERFETCH_FACTOR,
        rerankers: Mapping[str, Reranker] | None = None,
        rerank_candidates: int = DEFAULT_RERANK_CANDIDATES,
        retrieval_modes: Mapping[str, RetrievalStrategy] | None = None,
        max_query_chars: int = DEFAULT_MAX_QUERY_CHARS,
    ) -> None:
        if overfetch_factor < 1:
            raise ValueError("overfetch_factor must be at least 1")
        if rerank_candidates < 1:
            raise ValueError("rerank_candidates must be at least 1")
        if max_query_chars < 1:
            raise ValueError("max_query_chars must be at least 1")
        self._max_query_chars = max_query_chars
        self._rerankers: Mapping[str, Reranker] = (
            rerankers if rerankers is not None else {NO_RERANKER_KEY: NoopReranker()}
        )
        self._retrieval_modes: Mapping[str, RetrievalStrategy] = (
            retrieval_modes
            if retrieval_modes is not None
            else {DENSE_RETRIEVAL_MODE: DenseRetrieval()}
        )
        self._rerank_candidates = rerank_candidates
        self._overfetch_factor = overfetch_factor
        self._documents = documents
        self._chunks = chunks
        self._embeddings = embeddings

    async def execute(
        self,
        tenant_id: TenantId,
        profile: RagProfile,
        query: str,
        extra_filter: FilterExpr | None = None,
        query_vector: Sequence[float] | None = None,
    ) -> RetrievalResult:
        """Return aligned (chunk, citation) hits.

        Chunks whose document is missing or not READY (dangling, in-flight or superseded) are
        dropped. `top_k * overfetch_factor` candidates are fetched (at least `rerank_candidates`
        when a real reranker is configured, so the reranker has enough READY candidates after
        the drop), the READY ones are reranked (profile.reranker_key; the first
        `max(rerank_candidates, top_k)` only) and the result is trimmed to top_k. Candidates come
        from the profile's `retrieval_mode` strategy (dense, or dense + full-text fused); each
        hit keeps its vector `similarity` and, for hybrid, the fusion evidence; `rerank_score` is
        set only when the reranker rescored it. Rerankers run AFTER the fusion.

        The query is sanitized ONCE here (NUL and lone surrogates removed, which Postgres cannot
        store or compare) and the sanitized text feeds the embedding, both search branches and
        the reranker. A query longer than `max_query_chars` raises `QueryTooLongError`; one with
        nothing left to search returns no hits. Search operators are inert: the lexical branch
        parses the text as plain terms.

        `query_vector` is an embedding the caller already computed for this very query (the
        profile probe of an evidence router returns it): it must come from the same embedding
        model and from the SANITIZED text (what `ProbeProfiles` embeds), and it replaces the
        embedding call. An empty vector is a bug and raises ValueError.
        """
        if query_vector is not None and len(query_vector) == 0:
            raise ValueError("query_vector must not be empty")
        try:
            reranker = self._rerankers[profile.reranker_key]
        except KeyError:
            raise UnknownStrategyError(
                f"No reranker registered for key {profile.reranker_key!r}"
            ) from None
        try:
            retrieval = self._retrieval_modes[profile.retrieval_mode]
        except KeyError:
            raise UnknownStrategyError(
                f"No retrieval mode registered for key {profile.retrieval_mode!r}"
            ) from None
        query = strip_unstorable(query)
        if len(query) > self._max_query_chars:
            raise QueryTooLongError(
                f"Query has {len(query)} characters (limit {self._max_query_chars})"
            )
        if not query.strip():
            return RetrievalResult(hits=())
        vector = (
            query_vector if query_vector is not None else await self._embeddings.aembed_query(query)
        )
        parts: list[FilterExpr] = [Filter.eq("profile", profile.name)]
        parts.extend(f for f in (profile.default_filter, extra_filter) if f is not None)
        fetch = profile.top_k * self._overfetch_factor
        if profile.reranker_key != NO_RERANKER_KEY:
            fetch = max(fetch, self._rerank_candidates)
        found = await retrieval.candidates(
            self._chunks,
            CandidateRequest(
                tenant_id=tenant_id,
                query=query,
                vector=vector,
                filter=Criteria.all_of(*parts).filter,
                k=fetch,
                metadata_fields=profile.metadata_fields,
            ),
        )
        if not found:
            return RetrievalResult(hits=())

        doc_ids = sorted({c.scored.chunk.document_id for c in found})
        documents = await self._documents.matching(
            tenant_id, Criteria.all_of(Filter.in_("id", doc_ids))
        )
        ready = {d.id: d for d in documents if d.status is DocumentStatus.READY}
        candidates = [c for c in found if c.scored.chunk.document_id in ready]
        if not candidates:
            return RetrievalResult(hits=())
        limit = max(self._rerank_candidates, profile.top_k)
        pool = candidates[:limit]
        by_id = {c.scored.chunk.id: c for c in pool}
        ranked = await reranker.rerank(query, [c.scored for c in pool], profile.top_k)
        hits: list[RetrievedHit] = []
        for item in ranked:
            original = by_id.get(item.chunk.id)
            if original is None:
                logger.warning("Reranker returned a chunk outside its candidates; dropped")
                continue
            hits.append(
                RetrievedHit(
                    chunk=item,
                    citation=Citation.from_scored(item, ready[item.chunk.document_id]),
                    similarity=original.similarity,
                    # A reranker that hands the candidate back untouched (no-op, degraded
                    # fallback) did not score it.
                    rerank_score=None if item is original.scored else item.score,
                    lexical_rank=original.lexical_rank,
                    dense_rank=original.dense_rank,
                    fused_score=original.fused_score,
                )
            )
            if len(hits) == profile.top_k:
                break
        return RetrievalResult(hits=tuple(hits))
