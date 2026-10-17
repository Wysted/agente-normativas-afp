"""pgvector `relaxed_order` + filters: results must come back exactly ordered.

Needs a database with a wide enough embedding (64 dimensions) and enough rows for the planner to
walk the HNSW graph with an iterative scan (`agintis_test_hnsw`, see tests/support/postgres.py).
On Postgres 17+ an outer `ORDER BY distance` over the materialized candidates is dropped as
"redundant" (the CTE is ordered by distance) and relaxed_order's slightly shuffled rows leak out;
the repository defends with `ORDER BY distance + 0` and a final Python sort.
"""

import random
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import pytest
from sqlalchemy import event, insert, text
from sqlalchemy.engine import URL
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from agintis.modules.knowledge.domain import DocumentStatus
from agintis.modules.knowledge.infrastructure.postgres import PgChunkRepository
from agintis.modules.knowledge.infrastructure.postgres.chunks import search_settings_statement
from agintis.modules.knowledge.infrastructure.postgres.tables import chunks
from agintis.platform.db import create_session_factory
from agintis.shared_kernel.criteria import Filter
from agintis.shared_kernel.ids import TenantId
from tests.support.postgres import HNSW_EMBEDDING_DIM, postgres_engines

pytestmark = pytest.mark.integration

TENANTS = 10
ROWS = 20_000
BATCH = 5_000
K = 200
EF_SEARCH = 10  # far below K: the scan has to iterate
MAX_SCAN_TUPLES = 20_000
INDEX = "ix_chunks_embedding_hnsw"


def _vector(rng: random.Random) -> list[float]:
    return [rng.gauss(0, 1) for _ in range(HNSW_EMBEDDING_DIM)]


@dataclass
class Corpus:
    engine: AsyncEngine
    factory: async_sessionmaker[AsyncSession]
    repo: PgChunkRepository
    query: list[float]


@pytest.fixture(scope="module")
async def corpus(pg_hnsw_url: URL) -> AsyncIterator[Corpus]:
    rng = random.Random(7)
    async with postgres_engines(pg_hnsw_url) as (engine, _):
        factory = create_session_factory(engine)
        async with factory.begin() as session:
            await session.execute(text("DROP INDEX " + INDEX))  # bulk load, then build once
            await session.execute(
                text(
                    "INSERT INTO documents (id, tenant_id, profile, source_uri, title, mime_type, "
                    "content_hash, profile_fingerprint, status, created_at) "
                    "SELECT 'd' || t, 't' || t, 'p', 's', 't', 'm', 'h', 'f', :ready, now() "
                    "FROM generate_series(0, :last) AS t"
                ),
                {"ready": DocumentStatus.READY.value, "last": TENANTS - 1},
            )
            for start in range(0, ROWS, BATCH):
                rows = [
                    {
                        "id": f"c{i:06d}",
                        "document_id": f"d{i % TENANTS}",
                        "tenant_id": f"t{i % TENANTS}",
                        "profile": "p",
                        "ordinal": i,
                        "text": "x",
                        "metadata": {"page": i % 7},
                        "embedding": _vector(rng),
                    }
                    for i in range(start, start + BATCH)
                ]
                await session.execute(insert(chunks), rows)
            await session.execute(
                text(f"CREATE INDEX {INDEX} ON chunks USING hnsw (embedding vector_cosine_ops)")
            )
            await session.execute(text("ANALYZE documents"))
            await session.execute(text("ANALYZE chunks"))
        repo = PgChunkRepository(factory, ef_search=EF_SEARCH, max_scan_tuples=MAX_SCAN_TUPLES)
        yield Corpus(engine, factory, repo, _vector(rng))


async def _captured_search(corpus: Corpus, expr: Any) -> tuple[list[float], str, Any]:
    """Run the real search; return its scores plus the SQL/params of the main statement."""
    captured: dict[str, Any] = {}

    @event.listens_for(corpus.engine.sync_engine, "before_cursor_execute")
    def grab(conn: Any, cursor: Any, statement: str, parameters: Any, *_: Any) -> None:
        if "candidates" in statement:
            captured["sql"], captured["params"] = statement, parameters

    try:
        hits = await corpus.repo.search(TenantId("t3"), corpus.query, expr, K)
    finally:
        event.remove(corpus.engine.sync_engine, "before_cursor_execute", grab)
    return [h.score for h in hits], captured["sql"], captured["params"]


@pytest.mark.parametrize(
    "expr", [None, Filter.eq("page", 3)], ids=["tenant-only", "tenant-and-metadata"]
)
async def test_results_are_ordered_by_distance_with_a_filtered_iterative_scan(
    corpus: Corpus, expr: Any
) -> None:
    scores, _, _ = await _captured_search(corpus, expr)
    assert len(scores) == K if expr is None else len(scores) > 0
    assert scores == sorted(scores, reverse=True)  # similarity = 1 - distance, best first


async def test_the_plan_sorts_the_relaxed_candidates_at_the_top(corpus: Corpus) -> None:
    _, sql, params = await _captured_search(corpus, None)
    async with corpus.engine.connect() as conn:
        tx = await conn.begin()
        # Force the ANN path: on a table this small the planner prefers the tenant btree plus an
        # exact sort, which would make the ordering assertions trivially true. With these three
        # switches off the only way to order by distance cheaply is the HNSW index.
        for switch in ("enable_seqscan", "enable_bitmapscan", "enable_sort"):
            await conn.execute(text(f"SET LOCAL {switch} = off"))
        await conn.execute(
            search_settings_statement(),
            {
                "ef_search": str(EF_SEARCH),
                "max_scan_tuples": str(MAX_SCAN_TUPLES),
                "statement_timeout": "10000",
            },
        )
        plan = list(
            (await conn.exec_driver_sql("EXPLAIN (COSTS OFF) " + sql, params)).scalars().all()
        )
        distances = [
            row.distance for row in (await conn.exec_driver_sql(sql, params)).mappings().all()
        ]
        await tx.rollback()

    assert plan[0].startswith("Sort"), "\n".join(plan)  # not 'CTE Scan': the sort was not elided
    assert any(INDEX in line for line in plan), "\n".join(plan)  # the HNSW index drives the scan
    assert len(distances) == K
    assert distances == sorted(distances)


async def test_the_search_runs_under_its_own_statement_timeout(corpus: Corpus) -> None:
    """`SET LOCAL` per search transaction: a 1 ms cap cancels the ANN scan, and it does not leak
    into the pooled connection that serves the next statement."""
    strict = PgChunkRepository(
        corpus.factory, ef_search=EF_SEARCH, max_scan_tuples=MAX_SCAN_TUPLES, search_timeout_ms=1
    )
    with pytest.raises(DBAPIError, match="statement timeout"):
        await strict.search(TenantId("t3"), corpus.query, None, K)
    async with corpus.engine.connect() as conn:
        assert await conn.scalar(text("SHOW statement_timeout")) == "0"
    assert len(await corpus.repo.search(TenantId("t3"), corpus.query, None, K)) == K


async def test_ef_search_follows_k_inside_the_search_transaction(corpus: Corpus) -> None:
    async with corpus.engine.connect() as conn:
        await conn.execute(search_settings_statement(), corpus.repo._search_settings(K))
        assert await conn.scalar(text("SHOW hnsw.ef_search")) == str(K)
        assert await conn.scalar(text("SHOW statement_timeout")) == "10s"
        await conn.rollback()
