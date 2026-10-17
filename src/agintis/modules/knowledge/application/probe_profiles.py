"""Use case: a cheap, dense-only probe of several profiles with ONE query embedding.

A router that wants evidence from the corpus (instead of asking a model) needs, per candidate
profile, "how close is the best content of this profile to the query". `ProbeProfiles` embeds
the query once and runs one small dense search per profile, tenant-scoped and restricted to the
profile and its `default_filter` (the same scope retrieval uses), and returns each profile's top
similarities. The query vector is returned too, so the retrieval that follows does not embed
again (`RetrieveForTopic.execute(query_vector=...)`: the embedding is the expensive part on CPU).

Shared embedding space: similarities are comparable across profiles ONLY because every profile
is embedded with the same model. That holds by construction here: the module has ONE `Embeddings`
port, every chunk of every profile is written with it, the `chunks.embedding` column has one
width (checked at boot) and `RagProfile` has no embedding setting of its own. A future per-profile
embedding model must break this use case on purpose (give profiles an embedding key and refuse a
probe over mixed keys) before any cross-profile comparison is made.

Why N searches and not one SQL with a window function: a `ROW_NUMBER() OVER (PARTITION BY profile
ORDER BY distance)` cannot use the HNSW index (it would rank every chunk of the tenant), while
`ORDER BY distance LIMIT k` per profile does. N is the number of topics of one agent (a handful),
each search is a small index scan, and the searches of one probe run concurrently.

Concurrency is bounded PROCESS-WIDE, not per call: one semaphore owned by the instance (the
container builds one) caps the searches in flight across every concurrent `execute`, so N
simultaneous runs cannot take the whole connection pool away from thread leases and retrieval
(`KNOWLEDGE_PROBE_CONCURRENCY`, validated <= `DATABASE_POOL_SIZE - 1`). It is created lazily on
first use, so building the use case binds nothing to an event loop.

Failure: the first failing search cancels its siblings (they would only burn connections for a
result nobody reads) and the error re-raised is the one of the FIRST failing profile in input
order, with its own type (the error classifier maps raw exceptions; an `ExceptionGroup` would be
`internal`). Cancellation of the caller cancels every search and propagates.
"""

import asyncio
import math
from collections.abc import Sequence
from dataclasses import dataclass

from langchain_core.embeddings import Embeddings

from agintis.modules.knowledge.application.errors import QueryTooLongError
from agintis.modules.knowledge.application.retrieve_for_topic import DEFAULT_MAX_QUERY_CHARS
from agintis.modules.knowledge.domain import ChunkRepository, RagProfile, strip_unstorable
from agintis.shared_kernel.criteria import Criteria, Filter, FilterExpr
from agintis.shared_kernel.ids import TenantId

DEFAULT_PROBE_K = 3
DEFAULT_PROBE_CONCURRENCY = 2


@dataclass(frozen=True)
class ProfileEvidence:
    """What one profile holds for the query: its best cosine similarities, best first (empty =
    no READY content matched in the tenant)."""

    profile: str
    similarities: tuple[float, ...]


@dataclass(frozen=True)
class ProbeResult:
    """`evidence` follows the order of the probed profiles. `query_vector` is the embedding of
    the sanitized query (reuse it for retrieval), None when there was nothing to embed (blank
    query: every profile has empty evidence)."""

    query_vector: tuple[float, ...] | None
    evidence: tuple[ProfileEvidence, ...]


class ProbeProfiles:
    def __init__(
        self,
        chunks: ChunkRepository,
        embeddings: Embeddings,
        max_query_chars: int = DEFAULT_MAX_QUERY_CHARS,
        max_concurrency: int = DEFAULT_PROBE_CONCURRENCY,
    ) -> None:
        if max_query_chars < 1:
            raise ValueError("max_query_chars must be at least 1")
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        self._chunks = chunks
        self._embeddings = embeddings
        self._max_query_chars = max_query_chars
        self._max_concurrency = max_concurrency
        self._gate: asyncio.Semaphore | None = None  # created on first use (see module doc)

    async def execute(
        self,
        tenant_id: TenantId,
        profiles: Sequence[RagProfile],
        query: str,
        k: int = DEFAULT_PROBE_K,
        query_vector: Sequence[float] | None = None,
    ) -> ProbeResult:
        """Top-`k` similarities of each profile for `query`, as the tenant.

        The query is sanitized exactly like retrieval does (`strip_unstorable`, length limit
        `QueryTooLongError`). `query_vector`, when given, must be the embedding of that
        sanitized text and skips the embedding call (a retry of the same attempt). Only READY
        documents are searched (the repository contract), so a profile that has nothing ready
        reports empty evidence, never an error. Profile names must be unique.
        """
        if k < 1:
            raise ValueError("k must be at least 1")
        if query_vector is not None and len(query_vector) == 0:
            raise ValueError("query_vector must not be empty")
        names = [p.name for p in profiles]
        if len(set(names)) != len(names):
            raise ValueError("Probed profiles must have unique names")
        text = strip_unstorable(query)
        if len(text) > self._max_query_chars:
            raise QueryTooLongError(
                f"Query has {len(text)} characters (limit {self._max_query_chars})"
            )
        if not text.strip() or not profiles:
            return ProbeResult(None, tuple(ProfileEvidence(n, ()) for n in names))
        vector = (
            tuple(query_vector)
            if query_vector is not None
            else tuple(await self._embeddings.aembed_query(text))
        )
        if self._gate is None:
            self._gate = asyncio.Semaphore(self._max_concurrency)
        gate = self._gate

        async def probe(profile: RagProfile) -> ProfileEvidence:
            async with gate:
                found = await self._chunks.search(
                    tenant_id,
                    vector,
                    _scope(profile),
                    k,
                    metadata_fields=profile.metadata_fields,
                )
            # float(): adapters may hand back numpy scalars, which are not plain checkpoint data;
            # a non-finite score is no evidence.
            scores = (float(s.score) for s in found)
            ordered = sorted((x for x in scores if math.isfinite(x)), reverse=True)
            return ProfileEvidence(profile.name, tuple(ordered[:k]))

        tasks = [asyncio.ensure_future(probe(p)) for p in profiles]
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        finally:
            # Normal end: nothing pending. First failure or caller cancellation: stop the rest.
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        for task in tasks:  # input order: the same failure whatever the timing
            if not task.cancelled() and (error := task.exception()) is not None:
                raise error
        return ProbeResult(vector, tuple(task.result() for task in tasks))


def _scope(profile: RagProfile) -> FilterExpr | None:
    """The profile and its default filter: the same scope `RetrieveForTopic` searches."""
    parts: list[FilterExpr] = [Filter.eq("profile", profile.name)]
    if profile.default_filter is not None:
        parts.append(profile.default_filter)
    return Criteria.all_of(*parts).filter
