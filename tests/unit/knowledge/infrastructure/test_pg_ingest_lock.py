"""PgIngestLock failure paths that need no database (unreachable server)."""

import asyncio

import pytest
from pydantic import SecretStr
from sqlalchemy.exc import OperationalError

from agintis.modules.knowledge.api import ConcurrentIngestError
from agintis.modules.knowledge.infrastructure.postgres import PgIngestLock
from agintis.platform.db import create_unpooled_engine
from agintis.shared_kernel.ids import TenantId

TENANT = TenantId("t")
# Port 1 refuses connections immediately: the failure happens inside `connect()`.
DOWN = SecretStr("postgresql+psycopg://u:pw@127.0.0.1:1/db")


@pytest.mark.parametrize(
    "kwargs",
    [{"max_concurrent": 0}, {"acquire_timeout_seconds": 0}, {"poll_interval_seconds": 0}],
)
def test_limits_must_be_positive(kwargs: dict[str, float]) -> None:
    engine = create_unpooled_engine(DOWN)
    with pytest.raises(ValueError, match="greater than 0"):
        PgIngestLock(engine, **kwargs)  # type: ignore[arg-type]


async def test_a_failed_connect_surfaces_and_returns_the_slot() -> None:
    engine = create_unpooled_engine(DOWN, connect_timeout_seconds=1)
    lock = PgIngestLock(engine, max_concurrent=1, acquire_timeout_seconds=5)
    try:
        # With a single slot, the second attempt only reaches `connect()` again (and fails the
        # same way, not with a slot timeout) if the first one gave the slot back.
        for _ in range(2):
            with pytest.raises(OperationalError):
                async with lock.hold(TENANT, "p", "s"):
                    pytest.fail("the database is down")
    finally:
        await engine.dispose()


async def test_slot_wait_is_bounded_by_the_overall_deadline() -> None:
    engine = create_unpooled_engine(DOWN, connect_timeout_seconds=1)
    lock = PgIngestLock(engine, max_concurrent=1, acquire_timeout_seconds=0.2)
    await lock._slots.acquire()  # simulate a running ingest holding the only slot
    try:
        with pytest.raises(ConcurrentIngestError, match="slot"):
            async with lock.hold(TENANT, "p", "s"):
                pytest.fail("no slot is free")
    finally:
        lock._slots.release()
        await engine.dispose()


async def test_cancellation_while_connecting_does_not_leak_the_slot() -> None:
    engine = create_unpooled_engine(DOWN, connect_timeout_seconds=1)
    lock = PgIngestLock(engine, max_concurrent=1, acquire_timeout_seconds=5)
    try:
        task = asyncio.create_task(lock.hold(TENANT, "p", "s").__aenter__())
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises((asyncio.CancelledError, OperationalError)):
            await task
        assert not lock._slots.locked()
    finally:
        await engine.dispose()
