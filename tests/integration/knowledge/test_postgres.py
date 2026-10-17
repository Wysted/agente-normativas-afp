"""Postgres-specific behavior: `docker compose up -d --wait db` (and `embeddings` for e2e).

Runs against the disposable `agintis_test` database (see tests/support/postgres.py); skipped
when Postgres is unreachable. The generic repository behavior lives in the contract suite.
"""

import asyncio
import hashlib
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
from langchain_core.embeddings import Embeddings
from sqlalchemy import event, text
from sqlalchemy.engine import URL
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from agintis.bootstrap.container import SCHEMA_HEAD_REVISION, Container
from agintis.modules.knowledge.api import (
    DEFAULT_PROFILE,
    ConcurrentIngestError,
    IngestDocument,
    RagProfile,
    RetrieveForTopic,
    SourceFile,
)
from agintis.modules.knowledge.application import metadata_hash
from agintis.modules.knowledge.domain import (
    Chunk,
    Document,
    DocumentNotFoundError,
    DocumentStatus,
    MetadataField,
)
from agintis.modules.knowledge.infrastructure.postgres import (
    PgChunkRepository,
    PgDocumentRepository,
    PgIngestLock,
    metadata_index_problems,
)
from agintis.platform.clock import SystemClock
from agintis.platform.db import check_database, create_session_factory
from agintis.shared_kernel.criteria import Criteria
from agintis.shared_kernel.ids import ChunkId, DocumentId, TenantId
from tests.conftest import skip_or_fail
from tests.fakes.knowledge import FakeChunker, FakeParser
from tests.support.postgres import postgres_engines, truncate_all
from tests.support.settings import make_settings

pytestmark = pytest.mark.integration

TENANT = TenantId("tenant-a")
PROFILE = RagProfile("p", 5, 100, "fake", "fake")


class SlowEmbeddings(Embeddings):
    """Deterministic 2-dimensional vectors; the delay makes concurrent ingests overlap."""

    def __init__(self, delay: float = 0.05) -> None:
        self._delay = delay

    @staticmethod
    def _vector(text_: str) -> list[float]:
        digest = hashlib.sha256(text_.encode()).digest()
        return [1.0 + digest[0] / 255, 1.0 + digest[1] / 255]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        await asyncio.sleep(self._delay)
        return self.embed_documents(texts)


@dataclass
class Db:
    engine: AsyncEngine
    lock_engine: AsyncEngine
    factory: async_sessionmaker[AsyncSession]
    documents: PgDocumentRepository
    chunks: PgChunkRepository
    lock: PgIngestLock

    def ingest(self) -> IngestDocument:
        return IngestDocument(
            self.documents,
            self.chunks,
            {"fake": FakeParser()},
            {"fake": FakeChunker()},
            SlowEmbeddings(),
            SystemClock(),
            self.lock,
        )

    async def scalar_exec(self, sql: str) -> None:
        async with self.engine.begin() as conn:
            await conn.execute(text(sql))

    async def scalar(self, sql: str) -> object:
        async with self.engine.connect() as conn:
            return (await conn.execute(text(sql))).scalar_one()


@pytest.fixture
async def db(pg_test_url: URL) -> AsyncIterator[Db]:
    async with postgres_engines(pg_test_url) as (engine, lock_engine):
        factory = create_session_factory(engine)
        await truncate_all(factory)
        yield Db(
            engine,
            lock_engine,
            factory,
            PgDocumentRepository(factory),
            PgChunkRepository(factory),
            PgIngestLock(lock_engine),
        )


def _document(doc_id: str, *, status: DocumentStatus, source: str = "s") -> Document:
    return Document(
        id=DocumentId(doc_id),
        tenant_id=TENANT,
        profile="p",
        source_uri=source,
        title=doc_id,
        mime_type="text/plain",
        content_hash=f"h-{doc_id}",
        profile_fingerprint="fp",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        status=status,
    )


def _chunk(doc_id: str, ordinal: int, body: str) -> Chunk:
    return Chunk(ChunkId(f"{doc_id}-{ordinal}"), DocumentId(doc_id), TENANT, "p", ordinal, body)


def _source(content: str, uri: str = "file:///a.txt") -> SourceFile:
    return SourceFile(uri, "text/plain", content.encode())


class TestServer:
    async def test_pgvector_is_recent_and_unaccent_is_installed(self, db: Db) -> None:
        assert await check_database(db.engine, min_vector_version=(0, 8, 4)) >= "0.8.4"
        assert await db.scalar("SELECT extversion FROM pg_extension WHERE extname = 'unaccent'")

    async def test_iterative_scan_setting_does_not_leak_out_of_the_search(self, db: Db) -> None:
        await db.documents.save(_document("d", status=DocumentStatus.READY))
        await db.chunks.add_many([_chunk("d", 0, "hola")], [[1.0, 0.0]])
        await db.chunks.search(TENANT, [1.0, 0.0], None, k=1)
        assert await db.scalar("SHOW hnsw.iterative_scan") == "off"
        assert await db.scalar("SHOW hnsw.ef_search") == "40"
        assert await db.scalar("SHOW hnsw.max_scan_tuples") == "20000"

    async def test_search_applies_the_configured_hnsw_knobs_inside_its_transaction(
        self, db: Db
    ) -> None:
        seen: list[tuple[str, ...]] = []
        repo = PgChunkRepository(db.factory, ef_search=123, max_scan_tuples=4567)
        await db.documents.save(_document("d", status=DocumentStatus.READY))
        await repo.add_many([_chunk("d", 0, "hola")], [[1.0, 0.0]])

        @event.listens_for(db.engine.sync_engine, "before_cursor_execute")
        def spy(conn: Any, cursor: Any, statement: str, parameters: Any, *_: Any) -> None:
            if "set_config" in statement:
                seen.append(tuple(str(v) for v in parameters.values()))

        try:
            await repo.search(TENANT, [1.0, 0.0], None, k=1)
        finally:
            event.remove(db.engine.sync_engine, "before_cursor_execute", spy)
        assert seen == [("123", "4567", "10000")]  # ef_search (k=1 < 123), scan cap, timeout

        seen.clear()
        event.listen(db.engine.sync_engine, "before_cursor_execute", spy)
        try:
            await repo.search(TENANT, [1.0, 0.0], None, k=500)
        finally:
            event.remove(db.engine.sync_engine, "before_cursor_execute", spy)
        assert seen == [("500", "4567", "10000")]  # ef_search follows k

    async def test_hnsw_knobs_are_validated(self, db: Db) -> None:
        kwargs_list: list[dict[str, Any]] = [
            {"ef_search": 0},
            {"ef_search": 1001},
            {"max_scan_tuples": 0},
        ]
        for kwargs in kwargs_list:
            with pytest.raises(ValueError, match=r"ef_search|max_scan_tuples"):
                PgChunkRepository(db.factory, **kwargs)


class TestHybridReadyTsv:
    """`tsv` is GENERATED with the es_unaccent config: accent- and stem-insensitive."""

    async def _matches(self, db: Db, query: str) -> list[str]:
        await db.documents.save(_document("d", status=DocumentStatus.READY))
        await db.chunks.add_many(
            [
                _chunk("d", 0, "El artículo 12 regula las obligaciones del empleador."),
                _chunk("d", 1, "Las sociedades anónimas tributan en otro régimen."),
            ],
            [[1.0, 0.0]] * 2,
        )
        async with db.engine.connect() as conn:
            rows = await conn.execute(
                text(
                    "SELECT ordinal FROM chunks "
                    "WHERE tsv @@ websearch_to_tsquery('public.es_unaccent', :q) ORDER BY ordinal"
                ),
                {"q": query},
            )
            return [str(r[0]) for r in rows]

    async def test_accents_are_ignored_in_both_directions(self, db: Db) -> None:
        assert await self._matches(db, "articulo") == ["0"]  # query without accent
        await truncate_all(db.factory)
        assert await self._matches(db, "ARTÍCULO") == ["0"]  # query with accent and caps
        await truncate_all(db.factory)
        assert await self._matches(db, "anonimas") == ["1"]

    async def test_spanish_stemming_applies(self, db: Db) -> None:
        assert await self._matches(db, "empleadores") == ["0"]  # plural -> same stem
        await truncate_all(db.factory)
        assert await self._matches(db, "artículos") == ["0"]

    async def test_tsv_is_filled_by_the_database_not_the_application(self, db: Db) -> None:
        await self._matches(db, "x")
        assert await db.scalar("SELECT count(*) FROM chunks WHERE tsv IS NOT NULL") == 2


class TestIntegrity:
    async def test_chunk_cannot_point_to_a_document_of_another_tenant(self, db: Db) -> None:
        await db.documents.save(_document("d", status=DocumentStatus.READY))
        foreign = Chunk(ChunkId("x"), DocumentId("d"), TenantId("tenant-b"), "p", 0, "leak")
        with pytest.raises(DocumentNotFoundError):
            await db.chunks.add_many([foreign], [[1.0, 0.0]])
        assert await db.scalar("SELECT count(*) FROM chunks") == 0

    async def test_chunk_cannot_point_to_a_document_of_another_profile(self, db: Db) -> None:
        await db.documents.save(_document("d", status=DocumentStatus.READY))  # profile "p"
        other = Chunk(ChunkId("x"), DocumentId("d"), TENANT, "q", 0, "wrong profile")
        with pytest.raises(DocumentNotFoundError):
            await db.chunks.add_many([other], [[1.0, 0.0]])

    async def test_other_database_errors_are_not_masked_as_missing_documents(self, db: Db) -> None:
        await db.documents.save(_document("d", status=DocumentStatus.READY))
        await db.chunks.add_many([_chunk("d", 0, "a")], [[1.0, 0.0]])
        with pytest.raises(IntegrityError):  # duplicate primary key: a different violation
            await db.chunks.add_many([_chunk("d", 0, "a")], [[1.0, 0.0]])

    async def test_document_identity_is_enforced_by_the_database_too(self, db: Db) -> None:
        await db.documents.save(_document("d", status=DocumentStatus.READY))
        await db.chunks.add_many([_chunk("d", 0, "a")], [[1.0, 0.0]])
        async with db.engine.begin() as conn:
            with pytest.raises(IntegrityError):  # the composite FK pins (id, tenant, profile)
                await conn.execute(text("UPDATE documents SET profile = 'moved' WHERE id = 'd'"))

    async def test_deleting_a_document_cascades_to_its_chunks(self, db: Db) -> None:
        await db.documents.save(_document("d", status=DocumentStatus.READY))
        await db.chunks.add_many([_chunk("d", 0, "a"), _chunk("d", 1, "b")], [[1.0, 0.0]] * 2)
        await db.documents.delete(TENANT, DocumentId("d"))
        assert await db.scalar("SELECT count(*) FROM chunks") == 0

    async def test_status_is_constrained(self, db: Db) -> None:
        async with db.engine.begin() as conn:
            with pytest.raises(IntegrityError):
                await conn.execute(
                    text(
                        "INSERT INTO documents (id, tenant_id, profile, source_uri, title, "
                        "mime_type, content_hash, profile_fingerprint, status, created_at) "
                        "VALUES ('x','t','p','s','t','m','h','f','bogus', now())"
                    )
                )


class TestMarkReadyConcurrency:
    async def test_racing_promotions_of_one_source_leave_a_single_ready_version(
        self, db: Db
    ) -> None:
        await db.documents.save(_document("old", status=DocumentStatus.READY))
        for name in ("n1", "n2", "n3", "n4"):
            await db.documents.save(_document(name, status=DocumentStatus.PENDING))
        results = await asyncio.gather(
            *(db.documents.mark_ready(TENANT, DocumentId(n), 1) for n in ("n1", "n2", "n3", "n4"))
        )
        assert all(results)  # serialized by the source lock: every promotion applies, last wins
        docs = await db.documents.matching(TENANT, Criteria())
        ready = [d for d in docs if d.status is DocumentStatus.READY]
        assert len(ready) == 1
        assert {d.status for d in docs if d.id not in {ready[0].id}} == {DocumentStatus.SUPERSEDED}


class TestIngestLock:
    async def _advisory_locks(self, db: Db) -> object:
        # Advisory locks are per database: other databases on the server must not interfere.
        return await db.scalar(
            "SELECT count(*) FROM pg_locks WHERE locktype = 'advisory' AND granted "
            "AND database = (SELECT oid FROM pg_database WHERE datname = current_database())"
        )

    async def test_same_source_is_exclusive_and_other_sources_are_not(self, db: Db) -> None:
        entered = asyncio.Event()
        release = asyncio.Event()

        async def holder() -> None:
            async with db.lock.hold(TENANT, "p", "s"):
                entered.set()
                await release.wait()

        task = asyncio.create_task(holder())
        await asyncio.wait_for(entered.wait(), 5)
        async with db.lock.hold(TENANT, "p", "other-source"):  # different key: no wait
            pass
        async with db.lock.hold(TenantId("tenant-b"), "p", "s"):  # different tenant: no wait
            pass
        with pytest.raises(TimeoutError):  # same key: blocked (and the attempt is cancelled)
            async with asyncio.timeout(0.3), db.lock.hold(TENANT, "p", "s"):
                pytest.fail("the lock must be held by another session")
        release.set()
        await task
        async with asyncio.timeout(5), db.lock.hold(TENANT, "p", "s"):
            pass  # free again: neither the cancelled waiter nor the holder leaked it
        assert await self._advisory_locks(db) == 0

    async def test_cancellation_while_holding_releases_the_lock(self, db: Db) -> None:
        entered = asyncio.Event()

        async def holder() -> None:
            async with db.lock.hold(TENANT, "p", "s"):
                entered.set()
                await asyncio.sleep(60)

        task = asyncio.create_task(holder())
        await asyncio.wait_for(entered.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        async with asyncio.timeout(5), db.lock.hold(TENANT, "p", "s"):
            pass
        assert await self._advisory_locks(db) == 0

    async def test_error_inside_the_critical_section_releases_the_lock(self, db: Db) -> None:
        with pytest.raises(RuntimeError, match="boom"):
            async with db.lock.hold(TENANT, "p", "s"):
                raise RuntimeError("boom")
        assert await self._advisory_locks(db) == 0


class TestMetadataIndexes:
    async def test_indexes_that_are_missing_or_not_ours_are_reported(self, db: Db) -> None:
        indexed = MetadataField("article", "str", filterable=True, indexed=True)
        plain = MetadataField("chapter", "str", filterable=True)

        async def problems(*fields: MetadataField) -> list[tuple[str, str]]:
            return [
                (e.name, e.status.value) for e in await metadata_index_problems(db.engine, fields)
            ]

        assert await problems(plain) == []
        assert await problems(indexed, plain) == [("ix_chunks_meta_article", "missing")]
        async with db.engine.begin() as conn:
            await conn.execute(
                text("CREATE INDEX ix_chunks_meta_article ON chunks ((metadata->>'article'))")
            )
        try:
            # Built by hand: it works, but this tool did not create it (no fingerprint).
            assert await problems(indexed) == [("ix_chunks_meta_article", "unmanaged")]
        finally:
            async with db.engine.begin() as conn:
                await conn.execute(text("DROP INDEX ix_chunks_meta_article"))


class TestIngestLockResources:
    """Bounded connections and a bounded wait: contention never piles up server backends."""

    @staticmethod
    def _count_connections(lock_engine: AsyncEngine) -> list[int]:
        opened: list[int] = []

        @event.listens_for(lock_engine.sync_engine, "connect")
        def count(dbapi_connection: Any, record: Any) -> None:
            opened.append(1)

        return opened

    async def test_waiting_for_a_held_source_times_out_without_blocking_a_backend(
        self, db: Db
    ) -> None:
        lock = PgIngestLock(db.lock_engine, acquire_timeout_seconds=0.6, poll_interval_seconds=0.05)
        entered, release = asyncio.Event(), asyncio.Event()

        async def holder() -> None:
            async with db.lock.hold(TENANT, "p", "s"):
                entered.set()
                await release.wait()

        task = asyncio.create_task(holder())
        await asyncio.wait_for(entered.wait(), 5)
        with pytest.raises(ConcurrentIngestError, match="another ingest"):
            async with lock.hold(TENANT, "p", "s"):
                pytest.fail("the source is held by another session")
        # try-lock polling: nothing sits blocked on the server waiting for the lock
        blocked = await db.scalar(
            "SELECT count(*) FROM pg_stat_activity "
            "WHERE datname = current_database() AND wait_event_type = 'Lock'"
        )
        assert blocked == 0
        release.set()
        await task
        async with asyncio.timeout(5), lock.hold(TENANT, "p", "s"):
            pass  # the timed-out attempt left nothing behind

    async def test_concurrent_lock_connections_are_capped_by_the_semaphore(self, db: Db) -> None:
        lock = PgIngestLock(db.lock_engine, max_concurrent=2)
        opened = self._count_connections(db.lock_engine)
        running = peak = 0

        async def ingest(n: int) -> None:
            nonlocal running, peak
            async with lock.hold(TENANT, "p", f"source-{n}"):  # different keys: no contention
                running += 1
                peak = max(peak, running)
                await asyncio.sleep(0.1)
                running -= 1

        await asyncio.wait_for(asyncio.gather(*(ingest(n) for n in range(6))), 20)
        assert peak == 2
        assert len(opened) == 6  # one connection per ingest, never more than 2 at a time
        assert await self._open_locks(db) == 0

    async def test_waiting_for_a_slot_opens_no_connection_and_times_out(self, db: Db) -> None:
        lock = PgIngestLock(db.lock_engine, max_concurrent=1, acquire_timeout_seconds=0.4)
        opened = self._count_connections(db.lock_engine)
        entered, release = asyncio.Event(), asyncio.Event()

        async def holder() -> None:
            async with lock.hold(TENANT, "p", "a"):
                entered.set()
                await release.wait()

        task = asyncio.create_task(holder())
        await asyncio.wait_for(entered.wait(), 5)
        with pytest.raises(ConcurrentIngestError, match="slot"):
            async with lock.hold(TENANT, "p", "b"):
                pytest.fail("no slot is free")
        assert len(opened) == 1  # only the holder's connection exists
        release.set()
        await task
        async with asyncio.timeout(5), lock.hold(TENANT, "p", "b"):
            pass  # the slot came back

    async def test_a_cancelled_waiter_returns_its_slot(self, db: Db) -> None:
        lock = PgIngestLock(db.lock_engine, max_concurrent=1)
        entered, release = asyncio.Event(), asyncio.Event()

        async def holder() -> None:
            async with lock.hold(TENANT, "p", "a"):
                entered.set()
                await release.wait()

        task = asyncio.create_task(holder())
        await asyncio.wait_for(entered.wait(), 5)
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.2), lock.hold(TENANT, "p", "b"):
                pytest.fail("no slot is free")
        release.set()
        await task
        async with asyncio.timeout(5), lock.hold(TENANT, "p", "b"):
            pass

    async def _open_locks(self, db: Db) -> object:
        return await db.scalar(
            "SELECT count(*) FROM pg_locks WHERE locktype = 'advisory' AND granted "
            "AND database = (SELECT oid FROM pg_database WHERE datname = current_database())"
        )


class TestConcurrentIngest:
    async def test_same_content_is_indexed_once_and_the_rest_skip(self, db: Db) -> None:
        source = _source("uno\n\ndos\n\ntres")
        results = await asyncio.gather(
            *(db.ingest().execute(TENANT, PROFILE, source) for _ in range(4))
        )
        assert sorted(r.status for r in results) == ["indexed", "skipped", "skipped", "skipped"]
        docs = await db.documents.find_by_source(TENANT, "p", source.uri)
        assert [d.status for d in docs] == [DocumentStatus.READY]
        assert await db.scalar("SELECT count(*) FROM chunks") == 3

    async def test_different_content_replaces_serially_and_leaves_one_version(self, db: Db) -> None:
        uri = "file:///a.txt"
        sources = [_source(f"version {n}\n\nbody {n}", uri) for n in range(3)]
        results = await asyncio.gather(*(db.ingest().execute(TENANT, PROFILE, s) for s in sources))
        assert sorted(r.status for r in results) == ["indexed", "replaced", "replaced"]
        docs = await db.documents.find_by_source(TENANT, "p", uri)
        assert [d.status for d in docs] == [DocumentStatus.READY]
        assert await db.scalar("SELECT count(*) FROM chunks") == 2

    async def test_changed_document_metadata_reprocesses_then_skips(self, db: Db) -> None:
        uri = "file:///a.txt"

        def source(**meta: object) -> SourceFile:
            return SourceFile(uri, "text/plain", b"alfa", document_metadata=meta)  # type: ignore[arg-type]

        first = await db.ingest().execute(TENANT, PROFILE, source(number=1))
        again = await db.ingest().execute(TENANT, PROFILE, source(number=1))
        changed = await db.ingest().execute(TENANT, PROFILE, source(number=2))
        settled = await db.ingest().execute(TENANT, PROFILE, source(number=2))
        assert [r.status for r in (first, again, changed, settled)] == [
            "indexed",
            "skipped",
            "replaced",
            "skipped",
        ]
        (stored,) = await db.documents.find_by_source(TENANT, "p", uri)
        assert stored.metadata_hash == metadata_hash({"number": 2})

    async def test_a_row_without_a_metadata_hash_is_reprocessed_once(self, db: Db) -> None:
        uri = "file:///a.txt"
        source = SourceFile(uri, "text/plain", b"alfa", document_metadata={"n": 1})
        await db.ingest().execute(TENANT, PROFILE, source)
        await db.scalar_exec("UPDATE documents SET metadata_hash = NULL")  # a pre-0002 row
        statuses = [(await db.ingest().execute(TENANT, PROFILE, source)).status for _ in range(2)]
        assert statuses == ["replaced", "skipped"]

    async def test_reingest_swaps_the_served_version(self, db: Db) -> None:
        uri = "file:///a.txt"
        await db.ingest().execute(TENANT, PROFILE, _source("alfa", uri))
        await db.ingest().execute(TENANT, PROFILE, _source("beta", uri))
        hits = await db.chunks.search(TENANT, SlowEmbeddings._vector("beta"), None, k=10)
        assert [h.chunk.text for h in hits] == ["beta"]


def _tei_up(url: str) -> bool:
    try:
        return httpx.get(f"{url}/health", timeout=2).is_success
    except httpx.HTTPError:
        return False


SPANISH_LAW = """# Ley de ejemplo

## Artículo 1

La jornada ordinaria de trabajo no podrá exceder de cuarenta y cuatro horas semanales.

## Artículo 2

Las sociedades anónimas deberán llevar un libro de actas de las juntas de accionistas.

## Artículo 3

El arrendatario pagará la renta dentro de los primeros cinco días de cada mes.
"""


async def test_real_container_ingests_and_retrieves_with_bge_m3(pg_e2e_url: URL) -> None:
    """Container -> TEI bge-m3 -> Postgres (1024 dims) -> retrieval with citations."""
    settings = make_settings(
        database_url=pg_e2e_url.render_as_string(hide_password=False),
        embeddings={"expected_dim": 1024},
    )
    if not _tei_up(settings.embeddings.url):
        skip_or_fail(f"TEI embeddings not reachable at {settings.embeddings.url}")
    async with Container.build(settings) as container:
        try:
            await asyncio.to_thread(container.token_counter.warm_up)
        except Exception as exc:  # hub unreachable and tokenizer not cached
            skip_or_fail(f"tokenizer unavailable: {exc}")
        assert await check_database(
            container.engine,
            min_vector_version=(0, 8, 4),
            expected_embedding_dim=1024,
            expected_revision=SCHEMA_HEAD_REVISION,
        )
        profile = RagProfile(
            "e2e", 2, 128, "simple", "structured", chunker_params={"include_heading": True}
        )
        source = SourceFile("file:///ley.md", "text/markdown", SPANISH_LAW.encode())

        first = await container.ingest_document.execute(TENANT, profile, source)
        assert first.status == "indexed"
        assert (
            await container.ingest_document.execute(TENANT, profile, source)
        ).status == "skipped"

        result = await RetrieveForTopic(
            container.documents, container.chunks, container.embeddings
        ).execute(TENANT, profile, "¿cuántas horas semanales se puede trabajar?")
        assert result.hits
        assert "jornada" in result.hits[0].chunk.chunk.text
        assert result.hits[0].citation.source_uri == "file:///ley.md"
        other = await RetrieveForTopic(
            container.documents, container.chunks, container.embeddings
        ).execute(TenantId("tenant-b"), profile, "jornada de trabajo")
        assert other.hits == ()  # another tenant sees nothing
    assert DEFAULT_PROFILE  # the default profile still imports through the public api
