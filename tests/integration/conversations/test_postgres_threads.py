"""Postgres specifics of the thread repository: the lease race and the list index."""

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import psycopg
import pytest
from sqlalchemy import select, text
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from agintis.modules.conversations.domain import RunMode, Thread, ThreadBusyError
from agintis.modules.conversations.infrastructure.postgres import PgThreadRepository
from agintis.modules.conversations.infrastructure.postgres.fields import THREAD_FIELDS
from agintis.modules.conversations.infrastructure.postgres.tables import threads
from agintis.platform.db import apply_criteria, create_session_factory
from agintis.shared_kernel.criteria import Criteria, Order, OrderDirection, tenant_scoped
from agintis.shared_kernel.ids import TenantId, ThreadId
from tests.support.postgres import libpq_dsn, postgres_engines

pytestmark = pytest.mark.integration

A = TenantId("tenant-a")
LONG = timedelta(minutes=5)
BASE = datetime(2026, 1, 1, tzinfo=UTC)
BLOCKED_POLL_SECONDS = 0.02
BLOCKED_TIMEOUT_SECONDS = 5


@pytest.fixture
async def factory(pg_test_url: URL) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    async with postgres_engines(pg_test_url) as (engine, _lock_engine):
        session_factory = create_session_factory(engine)
        async with session_factory.begin() as session:
            await session.execute(text("TRUNCATE threads"))
        yield session_factory


async def _wait_until_blocked_on_a_lock(url: URL) -> None:
    """Poll pg_stat_activity until some backend waits for a row lock held by another."""
    async with await psycopg.AsyncConnection.connect(libpq_dsn(url), autocommit=True) as probe:
        async with asyncio.timeout(BLOCKED_TIMEOUT_SECONDS):
            while True:
                cursor = await probe.execute(
                    "SELECT count(*) FROM pg_stat_activity "
                    "WHERE wait_event_type = 'Lock' AND query ILIKE 'UPDATE threads%'"
                )
                row = await cursor.fetchone()
                if row is not None and row[0] > 0:
                    return
                await asyncio.sleep(BLOCKED_POLL_SECONDS)


async def test_a_blocked_acquire_rechecks_its_condition_after_the_winner_commits(
    pg_test_url: URL, factory: async_sessionmaker[AsyncSession]
) -> None:
    """Deterministic race: the winner holds the row lock; the loser's UPDATE blocks on it and,
    once the winner commits, re-evaluates its WHERE (READ COMMITTED) against the new row."""
    repo = PgThreadRepository(factory)
    await repo.add(Thread.create(ThreadId("t1"), A, "chat", None, BASE))

    async with await psycopg.AsyncConnection.connect(libpq_dsn(pg_test_url)) as winner:
        await winner.execute("SELECT 1 FROM threads WHERE id = 't1' FOR UPDATE")  # BEGIN + lock
        loser = asyncio.create_task(
            repo.acquire_run_lease(A, ThreadId("t1"), RunMode.MESSAGE, LONG)
        )
        await _wait_until_blocked_on_a_lock(pg_test_url)
        assert not loser.done()  # really waiting for the lock, not racing

        await winner.execute(
            "UPDATE threads SET status = 'busy', lease_until = now() + interval '5 minutes', "
            "lease_token = 'winner' WHERE id = 't1'"
        )
        await winner.commit()

        with pytest.raises(ThreadBusyError):
            await asyncio.wait_for(loser, BLOCKED_TIMEOUT_SECONDS)

    async with factory() as session:
        token = (
            await session.execute(text("SELECT lease_token FROM threads WHERE id = 't1'"))
        ).scalar_one()
    assert token == "winner"  # the loser changed nothing


async def test_the_list_index_serves_the_criteria_order_by_without_a_sort(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    repo = PgThreadRepository(factory)
    for number in range(600):
        tenant = TenantId(f"tenant-{number % 3}")
        await repo.add(
            Thread.create(
                ThreadId(f"t{number:04d}"), tenant, "chat", None, BASE + timedelta(seconds=number)
            )
        )
    criteria = Criteria(order=(Order("updated_at", OrderDirection.DESC),), limit=20)
    scoped = Criteria(
        filter=tenant_scoped(TenantId("tenant-1"), None), order=criteria.order, limit=20
    )
    statement = apply_criteria(select(threads), scoped, THREAD_FIELDS)
    async with factory.begin() as session:
        sql = str(
            statement.compile(
                dialect=session.get_bind().dialect, compile_kwargs={"literal_binds": True}
            )
        )
        await session.execute(text("ANALYZE threads"))
        await session.execute(text("SET LOCAL enable_sort = off"))  # a Sort node = no usable index
        plan = "\n".join(row[0] for row in (await session.execute(text(f"EXPLAIN {sql}"))).all())

    assert "ix_threads_tenant_updated" in plan, plan
    assert "Sort" not in plan, plan
