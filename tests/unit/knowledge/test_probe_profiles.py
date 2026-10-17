"""`ProbeProfiles` (one embedding, one dense search per profile) and the precomputed query
vector of `RetrieveForTopic`."""

import asyncio
from collections.abc import Sequence
from dataclasses import replace

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings

from agintis.modules.knowledge.api import (
    HYBRID_RETRIEVAL_MODE,
    DocumentStatus,
    IngestDocument,
    ProbeProfiles,
    QueryTooLongError,
    RagProfile,
    RetrieveForTopic,
    ScoredChunk,
    SourceFile,
)
from agintis.modules.knowledge.domain import Chunk, MetadataField
from agintis.modules.knowledge.infrastructure.registry import build_retrieval_modes
from agintis.shared_kernel.criteria import Criteria, Filter, FilterExpr
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
OTHER_TENANT = TenantId("other")
ALPHA = RagProfile("alpha", 3, 100, "fake", "fake")
BETA = RagProfile("beta", 3, 100, "fake", "fake")
EMPTY = RagProfile("empty", 3, 100, "fake", "fake")


class CountingEmbeddings(Embeddings):
    """Deterministic vectors that remember every embedded query."""

    def __init__(self) -> None:
        self._inner = DeterministicFakeEmbedding(size=16)
        self.queries: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._inner.embed_query(text)

    async def aembed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return self._inner.embed_query(text)


class RecordingChunks(InMemoryChunkRepository):
    def __init__(self, documents: InMemoryDocumentRepository) -> None:
        super().__init__(documents)
        self.searches: list[tuple[TenantId, FilterExpr | None, int]] = []

    async def search(
        self,
        tenant_id: TenantId,
        query_vector: Sequence[float],
        filter: FilterExpr | None,
        k: int,
        *,
        metadata_fields: Sequence[MetadataField] = (),
    ) -> list[ScoredChunk]:
        self.searches.append((tenant_id, filter, k))
        return await super().search(
            tenant_id, query_vector, filter, k, metadata_fields=metadata_fields
        )


async def _env() -> tuple[ProbeProfiles, RetrieveForTopic, CountingEmbeddings, RecordingChunks]:
    documents = InMemoryDocumentRepository()
    chunks = RecordingChunks(documents)
    embeddings = CountingEmbeddings()
    ingest = IngestDocument(
        documents,
        chunks,
        {"fake": FakeParser()},
        {"fake": FakeChunker()},
        embeddings,
        FixedClock(),
        InMemoryIngestLock(),
    )
    for tenant, profile, uri, text in (
        (TENANT, ALPHA, "file:///a.txt", "alpha one\n\nalpha two\n\nalpha three\n\nalpha four"),
        (TENANT, BETA, "file:///b.txt", "beta one\n\nbeta two"),
        (OTHER_TENANT, EMPTY, "file:///e.txt", "empty one"),
    ):
        await ingest.execute(tenant, profile, SourceFile(uri, "text/plain", text.encode()))
    embeddings.queries.clear()
    probe = ProbeProfiles(chunks, embeddings)
    retrieve = RetrieveForTopic(documents, chunks, embeddings)
    return probe, retrieve, embeddings, chunks


class TestProbeProfiles:
    async def test_embeds_once_and_reports_each_profile_in_input_order(self) -> None:
        probe, _, embeddings, chunks = await _env()

        result = await probe.execute(TENANT, [BETA, ALPHA], "alpha one", k=2)

        assert embeddings.queries == ["alpha one"]  # ONE embedding for two profiles
        assert len(chunks.searches) == 2
        assert [e.profile for e in result.evidence] == ["beta", "alpha"]
        beta, alpha = result.evidence
        assert alpha.similarities[0] == pytest.approx(1.0)  # the chunk with that very text
        assert len(alpha.similarities) == 2  # capped at k
        assert list(alpha.similarities) == sorted(alpha.similarities, reverse=True)
        assert len(beta.similarities) == 2
        assert beta.similarities[0] < alpha.similarities[0]
        assert result.query_vector is not None
        assert result.query_vector == tuple(await embeddings.aembed_query("alpha one"))

    async def test_every_search_is_tenant_and_profile_scoped(self) -> None:
        probe, _, _, chunks = await _env()
        profile = RagProfile("alpha", 3, 100, "fake", "fake", default_filter=None)

        result = await probe.execute(TENANT, [profile, EMPTY], "alpha one")

        assert {t for t, _, _ in chunks.searches} == {TENANT}
        scopes = sorted((f for _, f, _ in chunks.searches), key=repr)
        expected = [
            Criteria.all_of(Filter.eq("profile", name)).filter for name in ("alpha", "empty")
        ]
        assert scopes == sorted(expected, key=repr)
        # `empty` only has content under ANOTHER tenant: nothing leaks, evidence is empty.
        assert result.evidence[1].similarities == ()

    async def test_the_profile_default_filter_is_part_of_the_scope(self) -> None:
        probe, _, _, chunks = await _env()
        only_page = Filter.eq("page", 99)
        never = RagProfile("alpha", 3, 100, "fake", "fake", default_filter=only_page)

        result = await probe.execute(TENANT, [never], "alpha one")

        assert result.evidence[0].similarities == ()  # no chunk on page 99
        (search,) = chunks.searches
        assert search[1] == Criteria.all_of(Filter.eq("profile", "alpha"), only_page).filter

    async def test_a_precomputed_vector_skips_the_embedding(self) -> None:
        probe, _, embeddings, _ = await _env()
        vector = await embeddings.aembed_query("alpha one")
        embeddings.queries.clear()

        result = await probe.execute(TENANT, [ALPHA], "alpha one", query_vector=vector)

        assert embeddings.queries == []
        assert result.evidence[0].similarities[0] == pytest.approx(1.0)

    async def test_a_blank_query_probes_nothing(self) -> None:
        probe, _, embeddings, chunks = await _env()

        result = await probe.execute(TENANT, [ALPHA, BETA], "  \x00 ")

        assert result.query_vector is None
        assert [e.similarities for e in result.evidence] == [(), ()]
        assert embeddings.queries == []
        assert chunks.searches == []

    async def test_the_query_is_sanitized_and_length_limited_like_retrieval(self) -> None:
        probe, _, embeddings, chunks = await _env()
        limited = ProbeProfiles(chunks, embeddings, max_query_chars=5)

        await probe.execute(TENANT, [ALPHA], "al\x00pha one")
        assert embeddings.queries == ["alpha one"]
        with pytest.raises(QueryTooLongError):
            await limited.execute(TENANT, [ALPHA], "alpha one")

    async def test_validation(self) -> None:
        probe, _, _, chunks = await _env()
        with pytest.raises(ValueError, match="unique"):
            await probe.execute(TENANT, [ALPHA, ALPHA], "q")
        with pytest.raises(ValueError, match="k must"):
            await probe.execute(TENANT, [ALPHA], "q", k=0)
        with pytest.raises(ValueError, match="max_query_chars"):
            ProbeProfiles(chunks, CountingEmbeddings(), max_query_chars=0)
        with pytest.raises(ValueError, match="max_concurrency"):
            ProbeProfiles(chunks, CountingEmbeddings(), max_concurrency=0)

    async def test_a_failing_search_propagates_its_own_exception(self) -> None:
        probe, _, _, chunks = await _env()

        async def explode(*args: object, **kwargs: object) -> list[ScoredChunk]:
            raise ConnectionError("database down")

        chunks.search = explode  # type: ignore[method-assign]

        with pytest.raises(ConnectionError, match="database down"):  # not an ExceptionGroup
            await probe.execute(TENANT, [ALPHA, BETA], "alpha one")

    async def test_searches_are_bounded_by_the_concurrency_limit(self) -> None:
        _, _, embeddings, chunks = await _env()
        bounded = ProbeProfiles(chunks, embeddings, max_concurrency=1)

        result = await bounded.execute(TENANT, [ALPHA, BETA], "alpha one")

        assert [e.profile for e in result.evidence] == ["alpha", "beta"]


class TestRetrieveWithAPrecomputedVector:
    async def test_the_vector_replaces_the_embedding_call(self) -> None:
        _, retrieve, embeddings, _ = await _env()
        vector = await embeddings.aembed_query("alpha one")
        embeddings.queries.clear()

        reused = await retrieve.execute(TENANT, ALPHA, "alpha one", query_vector=vector)
        plain = await retrieve.execute(TENANT, ALPHA, "alpha one")

        assert embeddings.queries == ["alpha one"]  # only the plain call embedded
        assert [h.chunk.chunk.text for h in reused.hits] == [h.chunk.chunk.text for h in plain.hits]
        assert reused.hits[0].similarity == pytest.approx(1.0)

    async def test_a_blank_query_still_returns_no_hits_and_an_empty_vector_is_a_bug(self) -> None:
        _, retrieve, _, _ = await _env()

        assert (await retrieve.execute(TENANT, ALPHA, " ", query_vector=[0.1])).hits == ()
        with pytest.raises(ValueError, match="query_vector"):
            await retrieve.execute(TENANT, ALPHA, "alpha one", query_vector=[])


class _Gated(InMemoryChunkRepository):
    """Every search waits `delay`, counting how many run at once and who got cancelled."""

    def __init__(self, delay: float = 0.02, fail: dict[str, Exception] | None = None) -> None:
        super().__init__()
        self.delay = delay
        self.fail = fail or {}
        self.in_flight = 0
        self.max_in_flight = 0
        self.two_running = asyncio.Event()
        self.cancelled: list[str] = []
        self.finished: list[str] = []

    async def search(
        self,
        tenant_id: TenantId,
        query_vector: Sequence[float],
        filter: FilterExpr | None,
        k: int,
        *,
        metadata_fields: Sequence[MetadataField] = (),
    ) -> list[ScoredChunk]:
        name = next(n for n in ("p0", "p1", "p2", "p3") if n in repr(filter))
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        if self.in_flight >= 2:
            self.two_running.set()
        try:
            await asyncio.sleep(0 if name in self.fail else self.delay)
            if name in self.fail:
                raise self.fail[name]
        except asyncio.CancelledError:
            self.cancelled.append(name)
            raise
        finally:
            self.in_flight -= 1
        self.finished.append(name)
        return []


POOL = [RagProfile(f"p{i}", 3, 100, "fake", "fake") for i in range(4)]


class TestConcurrencyAndFailure:
    async def test_the_bound_is_process_wide_across_concurrent_executions(self) -> None:
        chunks = _Gated()
        probe = ProbeProfiles(chunks, CountingEmbeddings(), max_concurrency=2)

        await asyncio.gather(
            probe.execute(TENANT, POOL[:3], "one"), probe.execute(TENANT, POOL, "two")
        )

        assert chunks.max_in_flight == 2  # 7 searches from 2 executions, never more than 2 at once
        assert len(chunks.finished) == 7

    async def test_building_it_binds_no_event_loop(self) -> None:
        probe = ProbeProfiles(_Gated(), CountingEmbeddings())

        assert probe._gate is None  # created on the first execute, inside a running loop
        await probe.execute(TENANT, POOL[:1], "q")
        assert probe._gate is not None

    async def test_the_first_failing_profile_in_input_order_wins_with_its_own_type(self) -> None:
        chunks = _Gated(fail={"p1": KeyError("p1"), "p2": ValueError("p2")})
        probe = ProbeProfiles(chunks, CountingEmbeddings(), max_concurrency=4)

        for _ in range(3):  # the same error whatever the scheduling
            with pytest.raises(KeyError, match="p1"):
                await probe.execute(TENANT, POOL[:3], "q")

    async def test_a_failure_cancels_the_siblings(self) -> None:
        chunks = _Gated(delay=5.0, fail={"p0": ConnectionError("down")})
        probe = ProbeProfiles(chunks, CountingEmbeddings(), max_concurrency=4)

        with pytest.raises(ConnectionError, match="down"):
            await probe.execute(TENANT, POOL[:3], "q")

        assert sorted(chunks.cancelled) == ["p1", "p2"]
        assert chunks.in_flight == 0

    async def test_cancelling_the_caller_cancels_every_search_and_propagates(self) -> None:
        chunks = _Gated(delay=5.0)
        probe = ProbeProfiles(chunks, CountingEmbeddings(), max_concurrency=2)
        task = asyncio.ensure_future(probe.execute(TENANT, POOL[:3], "q"))
        await chunks.two_running.wait()

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert sorted(chunks.cancelled) == ["p0", "p1"]  # the third never started
        assert chunks.in_flight == 0

    async def test_an_empty_precomputed_vector_is_a_bug(self) -> None:
        probe = ProbeProfiles(_Gated(), CountingEmbeddings())

        with pytest.raises(ValueError, match="query_vector"):
            await probe.execute(TENANT, POOL[:1], "q", query_vector=[])

    async def test_non_finite_scores_are_no_evidence(self) -> None:
        class _Weird(InMemoryChunkRepository):
            async def search(self, *args: object, **kwargs: object) -> list[ScoredChunk]:
                chunk = Chunk(ChunkId("c"), DocumentId("d"), TENANT, "p0", 0, "t")
                return [ScoredChunk(chunk, float("nan")), ScoredChunk(chunk, 0.5)]

        probe = ProbeProfiles(_Weird(), CountingEmbeddings())

        result = await probe.execute(TENANT, POOL[:1], "q")

        assert result.evidence[0].similarities == (0.5,)


class TestOnlyReadyDocumentsGiveEvidence:
    """The READY rule lives in `ChunkRepository.search` (contract suite `TestServedVersions`
    runs it against in-memory and Postgres); this checks the probe inherits it."""

    @pytest.mark.parametrize("status", [DocumentStatus.PENDING, DocumentStatus.SUPERSEDED])
    async def test_a_pending_or_superseded_document_is_invisible_to_the_probe(
        self, status: DocumentStatus
    ) -> None:
        probe, _, _, chunks = await _env()
        documents = chunks._documents
        assert documents is not None
        for doc_id, document in list(documents.items.items()):
            if document.profile == "alpha":
                documents.items[doc_id] = replace(document, status=status)

        result = await probe.execute(TENANT, [ALPHA, BETA], "alpha one")

        assert result.evidence[0].similarities == ()
        assert result.evidence[1].similarities  # beta is still READY


class TestHybridRetrievalWithAVector:
    async def test_the_lexical_branch_still_gets_the_text_not_the_vector(self) -> None:
        documents = InMemoryDocumentRepository()
        seen: list[tuple[Sequence[float], str]] = []

        class _Spy(InMemoryChunkRepository):
            async def search_hybrid(  # type: ignore[no-untyped-def]
                self, tenant_id, query_vector, query_text, *args, **kwargs
            ):
                seen.append((query_vector, query_text))
                return await super().search_hybrid(
                    tenant_id, query_vector, query_text, *args, **kwargs
                )

        chunks = _Spy(documents)
        embeddings = CountingEmbeddings()
        await IngestDocument(
            documents,
            chunks,
            {"fake": FakeParser()},
            {"fake": FakeChunker()},
            embeddings,
            FixedClock(),
            InMemoryIngestLock(),
        ).execute(TENANT, ALPHA, SourceFile("file:///a", "text/plain", b"alpha one\n\nalpha two"))
        embeddings.queries.clear()
        retrieve = RetrieveForTopic(
            documents,
            chunks,
            embeddings,
            retrieval_modes=build_retrieval_modes(branch_depth=20, rrf_k=60),
        )
        hybrid = replace(ALPHA, retrieval_mode=HYBRID_RETRIEVAL_MODE)
        vector = embeddings.embed_query("completely different text")

        result = await retrieve.execute(TENANT, hybrid, "alpha one", query_vector=vector)

        assert embeddings.queries == []  # no embedding call
        assert seen == [(vector, "alpha one")]  # the vector for the dense branch, the TEXT for FTS
        assert result.hits[0].lexical_rank == 1
        assert result.hits[0].chunk.chunk.text == "alpha one"
