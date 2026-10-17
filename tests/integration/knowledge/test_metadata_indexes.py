"""Metadata index sync against a real Postgres: DDL, idempotency, prune and planner usage."""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import pytest
import sqlalchemy as sa
from sqlalchemy import event, insert, text
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.base import Executable
from sqlalchemy.sql.expression import ClauseElement

from agintis.modules.knowledge.domain import DocumentStatus, MetadataField
from agintis.modules.knowledge.infrastructure.postgres import (
    IndexStatus,
    IndexSyncError,
    MetadataIndexSpec,
    MetadataIndexSync,
    PgChunkRepository,
    metadata_index_problems,
)
from agintis.modules.knowledge.infrastructure.postgres.fields import chunk_fields
from agintis.modules.knowledge.infrastructure.postgres.locks import index_sync_lock_keys
from agintis.modules.knowledge.infrastructure.postgres.tables import chunks
from agintis.platform.db import compile_filter, create_session_factory
from agintis.shared_kernel.criteria import Filter
from agintis.shared_kernel.ids import TenantId
from tests.support.postgres import postgres_engines, truncate_all

pytestmark = pytest.mark.integration

ROWS = 4000
WEIGHT = MetadataField("weight", "int", filterable=True, indexed=True)
ARTICLE = MetadataField("article", "str", filterable=True, indexed=True)
DAY = MetadataField("day", "date", filterable=True, indexed=True)
LAWS = MetadataField("laws", "list[str]", filterable=True, indexed=True)
FIELDS = (WEIGHT, ARTICLE, DAY, LAWS)
MANAGED = ("ix_chunks_meta_weight", "ix_chunks_meta_article", "ix_chunks_meta_day")


@dataclass
class Db:
    engine: AsyncEngine
    factory: async_sessionmaker[AsyncSession]
    sync: MetadataIndexSync


async def _drop_managed(engine: AsyncEngine) -> None:
    async with engine.connect() as conn:
        conn = await conn.execution_options(isolation_level="AUTOCOMMIT")
        rows = await conn.execute(
            text("SELECT indexname FROM pg_indexes WHERE indexname LIKE 'ix\\_chunks\\_meta\\_%'")
        )
        for (name,) in rows.all():
            await conn.execute(text(f'DROP INDEX IF EXISTS "{name}"'))


@pytest.fixture
async def db(pg_test_url: URL) -> AsyncIterator[Db]:
    async with postgres_engines(pg_test_url) as (engine, lock_engine):
        factory = create_session_factory(engine)
        await truncate_all(factory)
        await _drop_managed(engine)
        yield Db(engine, factory, MetadataIndexSync(engine, lock_engine))
        await _drop_managed(engine)
        await truncate_all(factory)


async def _seed(db: Db) -> None:
    async with db.factory.begin() as session:
        await session.execute(
            text(
                "INSERT INTO documents (id, tenant_id, profile, source_uri, title, mime_type, "
                "content_hash, profile_fingerprint, status, created_at) "
                "VALUES ('d', 't', 'p', 's', 't', 'm', 'h', 'f', :ready, now())"
            ),
            {"ready": DocumentStatus.READY.value},
        )
        await session.execute(
            insert(chunks),
            [
                {
                    "id": f"c{n:05d}",
                    "document_id": "d",
                    "tenant_id": "t",
                    "profile": "p",
                    "ordinal": n,
                    "text": "x",
                    "metadata": {
                        "weight": n,
                        "article": f"a{n:05d}",
                        "day": f"2020-01-{n % 28 + 1:02d}",
                        "laws": [f"L{n % 5}"],
                    },
                    "embedding": [1.0, 0.0],
                }
                for n in range(ROWS)
            ],
        )
        await session.execute(text("ANALYZE chunks"))


class Explain(Executable, ClauseElement):
    """`EXPLAIN <select>` that keeps the statement's own bind processing (JSONB parameters)."""

    inherit_cache = False

    def __init__(self, statement: sa.Select[Any]) -> None:
        self.statement = statement


@compiles(Explain)
def _compile_explain(element: Explain, compiler: Any, **kw: Any) -> str:
    return f"EXPLAIN {compiler.process(element.statement, **kw)}"


async def _explain(db: Db, expr: Any) -> str:
    """Plan of the filter exactly as the converter emits it (seq scans discouraged)."""
    stmt = sa.select(chunks.c.id).where(compile_filter(expr, chunk_fields(FIELDS)))
    async with db.engine.connect() as conn:
        await conn.execute(text("SET enable_seqscan = off"))
        try:
            rows = await conn.execute(Explain(stmt))
            return "\n".join(str(r[0]) for r in rows)
        finally:
            await conn.execute(text("RESET enable_seqscan"))


async def _problem_names(db: Db) -> list[str]:
    return [e.name for e in await metadata_index_problems(db.engine, FIELDS)]


async def _specs(db: Db) -> list[MetadataIndexSpec]:
    return [e.spec for e in (await db.sync.plan(FIELDS)).entries if e.spec is not None]


async def _managed_names(db: Db) -> list[str]:
    return [
        e.name for e in (await db.sync.plan(FIELDS)).entries if e.status is not IndexStatus.MISSING
    ]


class TestSync:
    async def test_dry_run_plans_and_apply_creates_idempotently(self, db: Db) -> None:
        plan = await db.sync.plan(FIELDS)
        assert {e.name: e.status for e in plan.entries} == dict.fromkeys(
            MANAGED, IndexStatus.MISSING
        )  # the list field needs no per-field index
        assert await _managed_names(db) == []  # planning changed nothing
        assert await _problem_names(db) == sorted(MANAGED)

        result = await db.sync.apply(plan)
        assert sorted(result.created) == sorted(MANAGED)
        assert result.dropped == ()
        again = await db.sync.plan(FIELDS)
        assert again.in_sync
        assert await _problem_names(db) == []

        second = await db.sync.apply(again)  # idempotent
        assert (second.created, second.dropped) == ((), ())

    async def test_apply_never_drops_without_prune(self, db: Db) -> None:
        await db.sync.apply(await db.sync.plan(FIELDS))
        async with db.engine.begin() as conn:
            await conn.execute(text("CREATE INDEX ix_chunks_meta_old ON chunks ((metadata->>'x'))"))
        plan = await db.sync.plan(FIELDS)
        assert {e.name for e in plan.with_status(IndexStatus.ORPHAN)} == {"ix_chunks_meta_old"}
        result = await db.sync.apply(plan)
        assert result.dropped == ()
        assert "ix_chunks_meta_old" in await _managed_names(db)

        pruned = await db.sync.apply(plan, prune=True)
        assert pruned.dropped == ("ix_chunks_meta_old",)
        assert "ix_chunks_meta_old" not in await _managed_names(db)

    async def test_a_stale_index_is_rebuilt_only_under_prune(self, db: Db) -> None:
        await db.sync.apply(await db.sync.plan(FIELDS))
        async with db.engine.begin() as conn:
            await conn.execute(
                text("COMMENT ON INDEX ix_chunks_meta_weight IS 'agintis:expr:000000000000'")
            )
        plan = await db.sync.plan(FIELDS)
        assert {e.name for e in plan.with_status(IndexStatus.STALE)} == {"ix_chunks_meta_weight"}
        assert (await db.sync.apply(plan)).created == ()
        assert (await db.sync.plan(FIELDS)).with_status(IndexStatus.STALE)

        rebuilt = await db.sync.apply(plan, prune=True)
        assert rebuilt.dropped == rebuilt.created == ("ix_chunks_meta_weight",)
        assert (await db.sync.plan(FIELDS)).in_sync

    async def test_indexes_are_partial_and_carry_a_verifiable_comment(self, db: Db) -> None:
        await db.sync.apply(await db.sync.plan(FIELDS))
        async with db.engine.connect() as conn:
            rows = (
                await conn.execute(
                    text(
                        "SELECT indexdef, obj_description(indexrelid, 'pg_class') "
                        "FROM pg_indexes JOIN pg_class ON relname = indexname "
                        "JOIN pg_index ON indexrelid = pg_class.oid "
                        "WHERE indexname LIKE 'ix\\_chunks\\_meta\\_%'"
                    )
                )
            ).all()
        assert len(rows) == len(MANAGED)
        for indexdef, comment in rows:
            assert " WHERE " in indexdef
            assert indexdef.endswith("IS NOT NULL)")
            assert comment.startswith("agintis:expr:")
            assert ":def:" in comment

    async def test_a_hand_made_index_is_unmanaged_and_only_prune_rebuilds_it(self, db: Db) -> None:
        async with db.engine.begin() as conn:
            await conn.execute(
                text("CREATE INDEX ix_chunks_meta_weight ON chunks ((metadata->>'weight'))")
            )
        plan = await db.sync.plan(FIELDS)
        assert {e.name for e in plan.with_status(IndexStatus.UNMANAGED)} == {
            "ix_chunks_meta_weight"
        }
        assert await _problem_names(db) == sorted(MANAGED)  # reported at boot, like a missing one
        created = (await db.sync.apply(plan)).created
        assert "ix_chunks_meta_weight" not in created  # never touched without prune

        rebuilt = await db.sync.apply(plan, prune=True)
        assert "ix_chunks_meta_weight" in rebuilt.dropped
        assert "ix_chunks_meta_weight" in rebuilt.created
        assert (await db.sync.plan(FIELDS)).in_sync

    async def test_an_index_altered_behind_our_back_is_unmanaged(self, db: Db) -> None:
        await db.sync.apply(await db.sync.plan(FIELDS))
        async with db.engine.begin() as conn:
            comment = await conn.scalar(
                text("SELECT obj_description('ix_chunks_meta_weight'::regclass, 'pg_class')")
            )
            await conn.execute(text("DROP INDEX ix_chunks_meta_weight"))
            await conn.execute(
                text("CREATE INDEX ix_chunks_meta_weight ON chunks ((metadata->>'weight'))")
            )
            await conn.execute(text(f"COMMENT ON INDEX ix_chunks_meta_weight IS '{comment}'"))
        plan = await db.sync.plan(FIELDS)
        assert {e.name for e in plan.with_status(IndexStatus.UNMANAGED)} == {
            "ix_chunks_meta_weight"
        }

    async def test_a_failing_comment_is_reported_and_marks_the_index_unmanaged(
        self, db: Db, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(MetadataIndexSpec, "comment_for", lambda self, indexdef: "it's broken")
        result = await db.sync.apply(await db.sync.plan(FIELDS))
        assert sorted(result.created) == sorted(MANAGED)  # the indexes exist and are valid
        assert sorted(result.unmarked) == sorted(MANAGED)
        monkeypatch.undo()
        plan = await db.sync.plan(FIELDS)
        assert {e.status for e in plan.entries} == {IndexStatus.UNMANAGED}
        fixed = await db.sync.apply(plan, prune=True)
        assert fixed.unmarked == ()
        assert (await db.sync.plan(FIELDS)).in_sync

    async def test_a_failure_midway_reports_what_was_already_created(
        self, db: Db, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real = MetadataIndexSpec.create_sql.fget  # type: ignore[attr-defined]

        def create_sql(self: MetadataIndexSpec) -> str:
            return (
                "CREATE INDEX CONCURRENTLY broken syntax"
                if self.name.endswith("day")
                else real(self)
            )

        monkeypatch.setattr(MetadataIndexSpec, "create_sql", property(create_sql))
        plan = await db.sync.plan(FIELDS)
        with pytest.raises(IndexSyncError) as failure:
            await db.sync.apply(plan)
        assert failure.value.result.created == ("ix_chunks_meta_article",)  # sorted: article < day
        monkeypatch.undo()
        assert await _managed_names(db) == ["ix_chunks_meta_article"]
        assert (await db.sync.apply(await db.sync.plan(FIELDS))).created == (
            "ix_chunks_meta_day",
            "ix_chunks_meta_weight",
        )  # re-running continues where it stopped

    async def test_a_name_used_by_another_relation_is_an_error_before_anything_is_created(
        self, db: Db
    ) -> None:
        async with db.engine.begin() as conn:
            await conn.execute(text("CREATE TABLE ix_chunks_meta_day (x int)"))
        try:
            with pytest.raises(IndexSyncError, match="another relation") as failure:
                await db.sync.apply(await db.sync.plan(FIELDS))
            assert failure.value.result.created == ()
            assert await _managed_names(db) == []
        finally:
            async with db.engine.begin() as conn:
                await conn.execute(text("DROP TABLE ix_chunks_meta_day"))

    async def test_two_syncs_never_run_at_once(self, db: Db) -> None:
        high, low = index_sync_lock_keys()
        async with db.engine.connect() as holder:
            await holder.execute(text("SELECT pg_advisory_lock(:h, :l)"), {"h": high, "l": low})
            try:
                with pytest.raises(IndexSyncError, match="Another index sync"):
                    await db.sync.apply(await db.sync.plan(FIELDS), lock_timeout_seconds=0.2)
            finally:
                await holder.execute(
                    text("SELECT pg_advisory_unlock(:h, :l)"), {"h": high, "l": low}
                )
        assert await _managed_names(db) == []  # the blocked run did nothing
        assert len((await db.sync.apply(await db.sync.plan(FIELDS))).created) == len(MANAGED)

    async def test_a_plan_computed_before_a_concurrent_run_is_not_applied_stale(
        self, db: Db
    ) -> None:
        stale = await db.sync.plan(FIELDS)  # everything MISSING
        await db.sync.apply(await db.sync.plan(FIELDS))
        result = await db.sync.apply(stale, prune=True)  # would drop and rebuild if trusted
        assert (result.created, result.dropped) == ((), ())

    async def test_an_index_that_is_still_building_is_never_dropped(self, db: Db) -> None:
        """A CREATE INDEX CONCURRENTLY blocked by an open writer is BUILDING, not INVALID."""
        (spec,) = (s for s in await _specs(db) if s.name == "ix_chunks_meta_weight")
        holder = await db.engine.connect()
        await holder.execute(text("DELETE FROM chunks WHERE false"))  # RowExclusiveLock, kept open

        async def build() -> None:
            async with db.engine.connect() as conn:
                conn = await conn.execution_options(isolation_level="AUTOCOMMIT")
                await conn.execute(text(spec.create_sql))

        builder = asyncio.create_task(build())
        try:
            async with asyncio.timeout(10):
                while True:
                    entry = next(
                        e for e in (await db.sync.plan([WEIGHT])).entries if e.name == spec.name
                    )
                    if entry.status is IndexStatus.BUILDING:
                        break
                    await asyncio.sleep(0.05)
            result = await db.sync.apply(await db.sync.plan([WEIGHT]), prune=True)
            assert spec.name in result.skipped
            assert spec.name not in result.dropped + result.created
        finally:
            await holder.rollback()
            await holder.close()
            await builder
        # Finished by someone else, without our comment: no longer building, not ours.
        entry = next(e for e in (await db.sync.plan([WEIGHT])).entries if e.name == spec.name)
        assert entry.status is IndexStatus.UNMANAGED

    async def test_indexes_carry_the_converter_expression(self, db: Db) -> None:
        await db.sync.apply(await db.sync.plan(FIELDS))
        async with db.engine.connect() as conn:
            definition = await conn.scalar(
                text("SELECT indexdef FROM pg_indexes WHERE indexname = 'ix_chunks_meta_article'")
            )
        assert definition is not None
        assert "jsonb_typeof" in definition
        assert "COLLATE" in definition


class TestPlannerUsesTheIndexes:
    async def test_range_filters_use_the_field_index(self, db: Db) -> None:
        await _seed(db)
        await db.sync.apply(await db.sync.plan(FIELDS))
        await _check_uses(db, Filter.gte("weight", 3990), "ix_chunks_meta_weight")
        await _check_uses(db, Filter.lt("weight", 5), "ix_chunks_meta_weight")
        await _check_uses(db, Filter.gt("article", "a03990"), "ix_chunks_meta_article")
        await _check_uses(db, Filter.gte("day", "2020-01-28"), "ix_chunks_meta_day")

    async def test_the_real_search_statement_with_its_join_uses_the_field_index(
        self, db: Db
    ) -> None:
        """The range predicate must imply the partial index predicate THROUGH the real search
        statement (documents JOIN, tenant scope, READY filter, materialized candidates)."""
        await _seed(db)
        await db.sync.apply(await db.sync.plan(FIELDS))
        repo = PgChunkRepository(db.factory)
        captured: dict[str, Any] = {}

        @event.listens_for(db.engine.sync_engine, "before_cursor_execute")
        def grab(conn: Any, cursor: Any, statement: str, parameters: Any, *_: Any) -> None:
            if "candidates" in statement:
                captured["sql"], captured["params"] = statement, parameters

        try:
            hits = await repo.search(
                TenantId("t"), [1.0, 0.0], Filter.gte("weight", 3990), 5, metadata_fields=FIELDS
            )
        finally:
            event.remove(db.engine.sync_engine, "before_cursor_execute", grab)
        assert [str(h.chunk.id) for h in hits] == [f"c{n:05d}" for n in range(3990, 3995)]
        async with db.engine.connect() as conn:
            plan = "\n".join(
                (await conn.exec_driver_sql("EXPLAIN " + captured["sql"], captured["params"]))
                .scalars()
                .all()
            )
        assert "ix_chunks_meta_weight" in plan, plan

    async def test_without_the_index_the_range_filter_cannot_use_it(self, db: Db) -> None:
        await _seed(db)
        plan = await _explain(db, Filter.gte("weight", 3990))
        assert "ix_chunks_meta_weight" not in plan

    async def test_equality_and_list_membership_are_served_by_the_table_wide_gin(
        self, db: Db
    ) -> None:
        await _seed(db)
        await db.sync.apply(await db.sync.plan(FIELDS))
        for expr in (
            Filter.eq("weight", 7),
            Filter.eq("article", "a00007"),
            Filter.eq("laws", "L1"),
        ):
            assert "ix_chunks_metadata" in await _explain(db, expr)


async def _check_uses(db: Db, expr: Any, index: str) -> None:
    plan = await _explain(db, expr)
    assert index in plan, plan
