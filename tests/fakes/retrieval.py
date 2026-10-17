"""Fake `TopicRetrieval`, fake `TopicProbe` plus builders of knowledge hits."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from agintis.modules.knowledge.api import (
    Citation,
    ProbeResult,
    ProfileEvidence,
    RagProfile,
    RetrievalResult,
    RetrievedHit,
    ScoredChunk,
)
from agintis.modules.knowledge.domain import Chunk
from agintis.shared_kernel.criteria import FilterExpr
from agintis.shared_kernel.ids import ChunkId, DocumentId, TenantId


def make_hit(
    text: str,
    *,
    uri: str = "file:///doc.pdf",
    title: str = "Doc",
    page: int | None = 1,
    heading: str | None = None,
    ordinal: int = 0,
) -> RetrievedHit:
    chunk = Chunk(
        ChunkId(f"chunk-{uri}-{ordinal}"),
        DocumentId("doc"),
        TenantId("t"),
        "p",
        ordinal,
        text,
    )
    return RetrievedHit(
        chunk=ScoredChunk(chunk, 0.9),
        citation=Citation(
            DocumentId("doc"), chunk.id, uri, title, page, text[:300], heading=heading
        ),
        similarity=0.9,
    )


@dataclass(frozen=True)
class RetrievalCall:
    tenant_id: TenantId
    profile: RagProfile
    query: str
    extra_filter: FilterExpr | None
    query_vector: Sequence[float] | None = None


class FakeRetrieval:
    """Answers each call with `respond(call)` (default: the same hits) and records the calls."""

    def __init__(
        self,
        hits: tuple[RetrievedHit, ...] = (),
        respond: Callable[[RetrievalCall], tuple[RetrievedHit, ...]] | None = None,
    ) -> None:
        self.calls: list[RetrievalCall] = []
        self._respond = respond or (lambda _call: hits)

    async def execute(
        self,
        tenant_id: TenantId,
        profile: RagProfile,
        query: str,
        extra_filter: FilterExpr | None = None,
        query_vector: Sequence[float] | None = None,
    ) -> RetrievalResult:
        call = RetrievalCall(tenant_id, profile, query, extra_filter, query_vector)
        self.calls.append(call)
        return RetrievalResult(hits=self._respond(call))


class FakeTopicProbe:
    """`TopicProbe` answering the scripted similarities per query TEXT and PROFILE name
    (a profile missing from the script has no hit); records the probed queries. With `raises`
    every call records the query and then raises that exception."""

    def __init__(
        self, table: dict[str, dict[str, Sequence[float]]], raises: Exception | None = None
    ) -> None:
        self._table = table
        self._raises = raises
        self.queries: list[str] = []

    @property
    def calls(self) -> int:
        return len(self.queries)

    async def execute(
        self,
        tenant_id: TenantId,
        profiles: Sequence[RagProfile],
        query: str,
        k: int = 3,
        query_vector: Sequence[float] | None = None,
    ) -> ProbeResult:
        self.queries.append(query)
        if self._raises is not None:
            raise self._raises
        scores = self._table[query]
        return ProbeResult(
            (0.1,), tuple(ProfileEvidence(p.name, tuple(scores.get(p.name, ()))) for p in profiles)
        )
