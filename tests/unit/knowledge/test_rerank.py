from collections.abc import Sequence
from dataclasses import dataclass

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from agintis.modules.knowledge.api import (
    IngestDocument,
    RagProfile,
    Reranker,
    RetrieveForTopic,
    ScoredChunk,
    SourceFile,
    UnknownStrategyError,
)
from agintis.modules.knowledge.application import NoopReranker
from agintis.modules.knowledge.domain import Chunk, MetadataField
from agintis.modules.knowledge.infrastructure.tei import FallbackReranker, TeiError
from agintis.shared_kernel.criteria import FilterExpr
from agintis.shared_kernel.ids import ChunkId, DocumentId, TenantId
from tests.fakes.clock import FixedClock
from tests.fakes.knowledge import (
    FakeChunker,
    FakeParser,
    InMemoryChunkRepository,
    InMemoryDocumentRepository,
    InMemoryIngestLock,
)

TENANT = TenantId("t")
TEXT = "\n\n".join(f"paragraph {i}" for i in range(10))


def _scored(n: int) -> list[ScoredChunk]:
    return [
        ScoredChunk(
            Chunk(ChunkId(f"c{i}"), DocumentId("d"), TENANT, "p", i, f"text {i}"), 1.0 - i / 100
        )
        for i in range(n)
    ]


class ReversingReranker(Reranker):
    """Reverses the candidates it receives and records what it saw."""

    def __init__(self) -> None:
        self.seen: list[int] = []
        self.top_n: int | None = None

    async def rerank(
        self, query: str, chunks: Sequence[ScoredChunk], top_n: int
    ) -> list[ScoredChunk]:
        self.seen.append(len(chunks))
        self.top_n = top_n
        return [ScoredChunk(c.chunk, 100.0 + i) for i, c in enumerate(reversed(chunks))][:top_n]


class RecordingChunks(InMemoryChunkRepository):
    """Remembers the `k` of every search."""

    def __init__(self) -> None:
        super().__init__()
        self.ks: list[int] = []

    async def search(
        self,
        tenant_id: TenantId,
        query_vector: Sequence[float],
        filter: FilterExpr | None,
        k: int,
        *,
        metadata_fields: Sequence[MetadataField] = (),
    ) -> list[ScoredChunk]:
        self.ks.append(k)
        return await super().search(
            tenant_id, query_vector, filter, k, metadata_fields=metadata_fields
        )


class ForeignChunkReranker(Reranker):
    """Misbehaves: answers with a chunk that was never a candidate."""

    async def rerank(
        self, query: str, chunks: Sequence[ScoredChunk], top_n: int
    ) -> list[ScoredChunk]:
        stranger = Chunk(ChunkId("stranger"), DocumentId("d"), TENANT, "p", 99, "x")
        return [ScoredChunk(stranger, 5.0), ScoredChunk(chunks[0].chunk, 4.0)]


class TestNoopReranker:
    async def test_keeps_order_and_truncates(self) -> None:
        chunks = _scored(5)
        assert await NoopReranker().rerank("q", chunks, 3) == chunks[:3]

    async def test_non_positive_top_n_returns_nothing(self) -> None:
        assert await NoopReranker().rerank("q", _scored(2), 0) == []


@dataclass
class Env:
    retrieve: RetrieveForTopic
    reranker: ReversingReranker
    profile_none: RagProfile
    profile_rev: RagProfile
    documents: InMemoryDocumentRepository
    chunks: InMemoryChunkRepository
    embeddings: DeterministicFakeEmbedding


async def _env(rerank_candidates: int, top_k: int = 2) -> Env:
    documents, chunks = InMemoryDocumentRepository(), InMemoryChunkRepository()
    embeddings = DeterministicFakeEmbedding(size=16)
    profile_none = RagProfile("p", top_k, 100, "fake", "fake")
    profile_rev = RagProfile("p", top_k, 100, "fake", "fake", reranker_key="rev")
    await IngestDocument(
        documents,
        chunks,
        {"fake": FakeParser()},
        {"fake": FakeChunker()},
        embeddings,
        FixedClock(),
        InMemoryIngestLock(),
    ).execute(TENANT, profile_none, SourceFile("file:///d.txt", "text/plain", TEXT.encode()))
    reranker = ReversingReranker()
    retrieve = RetrieveForTopic(
        documents,
        chunks,
        embeddings,
        rerankers={"none": NoopReranker(), "rev": reranker},
        rerank_candidates=rerank_candidates,
    )
    return Env(retrieve, reranker, profile_none, profile_rev, documents, chunks, embeddings)


class TestRetrieveWithReranker:
    async def test_default_reranker_is_noop(self) -> None:
        env = await _env(rerank_candidates=8)
        default = RetrieveForTopic(env.documents, env.chunks, env.embeddings)
        result = await default.execute(TENANT, env.profile_none, "paragraph 1")
        assert len(result.hits) == 2

    async def test_reranker_gets_bounded_candidates_and_decides_order(self) -> None:
        env = await _env(rerank_candidates=4, top_k=2)
        result = await env.retrieve.execute(TENANT, env.profile_rev, "paragraph 3")
        assert env.reranker.seen == [4]
        assert env.reranker.top_n == 2
        assert [h.chunk.score for h in result.hits] == [100.0, 101.0]

    async def test_candidates_never_below_top_k(self) -> None:
        env = await _env(rerank_candidates=1, top_k=3)
        result = await env.retrieve.execute(TENANT, env.profile_rev, "q")
        assert env.reranker.seen == [3]
        assert len(result.hits) == 3

    async def test_unknown_reranker_key_fails_loudly(self) -> None:
        env = await _env(rerank_candidates=4)
        bad = RagProfile("p", 2, 100, "fake", "fake", reranker_key="nope")
        with pytest.raises(UnknownStrategyError, match="nope"):
            await env.retrieve.execute(TENANT, bad, "q")

    async def test_empty_result_skips_reranker(self) -> None:
        env = await _env(rerank_candidates=4)
        result = await env.retrieve.execute(TenantId("other"), env.profile_rev, "q")
        assert result.hits == ()
        assert env.reranker.seen == []

    def test_rerank_candidates_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="rerank_candidates"):
            RetrieveForTopic(
                InMemoryDocumentRepository(),
                InMemoryChunkRepository(),
                DeterministicFakeEmbedding(size=4),
                rerank_candidates=0,
            )


class TestHitScores:
    """S3: similarity and rerank score are kept apart; `score` is the documented ordering key."""

    async def test_real_reranker_sets_rerank_score_and_keeps_similarity(self) -> None:
        env = await _env(rerank_candidates=4, top_k=2)
        result = await env.retrieve.execute(TENANT, env.profile_rev, "paragraph 3")
        for hit in result.hits:
            assert hit.rerank_score is not None
            assert hit.score == hit.rerank_score == hit.chunk.score
            assert hit.similarity is not None
            assert -1.0 <= hit.similarity <= 1.0  # cosine scale, not the reranker's

    async def test_without_a_reranker_the_order_key_is_the_similarity(self) -> None:
        env = await _env(rerank_candidates=4, top_k=3)
        result = await env.retrieve.execute(TENANT, env.profile_none, "paragraph 3")
        assert all(h.rerank_score is None for h in result.hits)
        assert [h.score for h in result.hits] == [h.similarity for h in result.hits]
        assert [h.score for h in result.hits] == sorted(
            (h.score for h in result.hits), reverse=True
        )

    async def test_degraded_reranker_leaves_no_rerank_score(self) -> None:
        env = await _env(rerank_candidates=4, top_k=2)
        failing = FallbackReranker(FailingReranker(), deadline_seconds=5)
        retrieve = RetrieveForTopic(
            env.documents, env.chunks, env.embeddings, rerankers={"rev": failing}
        )
        result = await retrieve.execute(TENANT, env.profile_rev, "paragraph 3")
        assert len(result.hits) == 2
        assert all(h.rerank_score is None for h in result.hits)

    async def test_chunks_outside_the_candidates_are_dropped(self) -> None:
        env = await _env(rerank_candidates=4, top_k=3)
        retrieve = RetrieveForTopic(
            env.documents, env.chunks, env.embeddings, rerankers={"rev": ForeignChunkReranker()}
        )
        result = await retrieve.execute(TENANT, env.profile_rev, "paragraph 3")
        assert len(result.hits) == 1
        assert all(h.chunk.chunk.id != "stranger" for h in result.hits)


class FailingReranker(Reranker):
    async def rerank(
        self, query: str, chunks: Sequence[ScoredChunk], top_n: int
    ) -> list[ScoredChunk]:
        raise TeiError("down")


class TestOverfetch:
    """S4: with a real reranker the store is asked for at least `rerank_candidates` hits."""

    @staticmethod
    async def _retrieve(rerank_candidates: int, key: str, top_k: int = 2) -> list[int]:
        documents, chunks = InMemoryDocumentRepository(), RecordingChunks()
        embeddings = DeterministicFakeEmbedding(size=16)
        ingest_profile = RagProfile("p", top_k, 100, "fake", "fake")
        await IngestDocument(
            documents,
            chunks,
            {"fake": FakeParser()},
            {"fake": FakeChunker()},
            embeddings,
            FixedClock(),
            InMemoryIngestLock(),
        ).execute(TENANT, ingest_profile, SourceFile("file:///d.txt", "text/plain", TEXT.encode()))
        retrieve = RetrieveForTopic(
            documents,
            chunks,
            embeddings,
            rerankers={"none": NoopReranker(), "rev": ReversingReranker()},
            rerank_candidates=rerank_candidates,
        )
        profile = RagProfile("p", top_k, 100, "fake", "fake", reranker_key=key)
        await retrieve.execute(TENANT, profile, "q")
        return chunks.ks

    async def test_real_reranker_fetches_at_least_the_candidate_count(self) -> None:
        assert await self._retrieve(rerank_candidates=8, key="rev") == [8]  # max(2*2, 8)

    async def test_overfetch_still_wins_when_larger(self) -> None:
        assert await self._retrieve(rerank_candidates=2, key="rev", top_k=3) == [6]

    async def test_noop_reranker_keeps_the_plain_overfetch(self) -> None:
        assert await self._retrieve(rerank_candidates=8, key="none") == [4]
