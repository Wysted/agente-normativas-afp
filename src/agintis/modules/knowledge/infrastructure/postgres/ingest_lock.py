"""Postgres `IngestLock`: a session-level advisory lock on a dedicated connection."""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from agintis.modules.knowledge.application import ConcurrentIngestError, IngestLock
from agintis.modules.knowledge.infrastructure.postgres.locks import session_lock_key
from agintis.shared_kernel.ids import TenantId

logger = logging.getLogger(__name__)


DEFAULT_MAX_CONCURRENT = 2
DEFAULT_ACQUIRE_TIMEOUT_SECONDS = 300.0
DEFAULT_POLL_INTERVAL_SECONDS = 0.2


class PgIngestLock(IngestLock):
    """`pg_try_advisory_lock` on a stable 64-bit hash of (tenant, profile, source_uri).

    The lock lives on its own connection, held in AUTOCOMMIT (no transaction stays open during
    the long parse/embed phase) for the whole critical section. Pass an engine whose
    connections are NOT pooled (`create_unpooled_engine`): then even a failed unlock cannot leak
    the lock, because closing the connection releases it. The release always runs, also on
    cancellation, shielded from further cancellation; if it cannot confirm the unlock the
    connection is invalidated (server-side close). TCP keepalives (see `create_unpooled_engine`)
    let the server notice a vanished client and drop its session, and with it the lock.

    Resource bounds. Every ingest holds one database connection for its whole duration, so:
    * `max_concurrent` (an `asyncio.Semaphore`) caps how many lock connections this process
      keeps open, i.e. how many ingests run at once. Waiters queue in the application, not as
      server backends.
    * Acquiring is bounded by ONE deadline (`acquire_timeout_seconds`) covering the queue for a
      slot, the connection and the polling for the lock: contention for the same source never
      blocks a server backend (the lock is taken by polling `pg_try_advisory_lock`), and past the
      deadline `ConcurrentIngestError` is raised (the caller may retry).

    Not compatible with PgBouncer in transaction (or statement) pooling: session-level advisory
    locks need a dedicated server session. Point the lock engine at Postgres directly or use
    session pooling.
    """

    def __init__(
        self,
        engine: AsyncEngine,
        *,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT,
        acquire_timeout_seconds: float = DEFAULT_ACQUIRE_TIMEOUT_SECONDS,
        poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    ) -> None:
        if max_concurrent <= 0:
            raise ValueError("max_concurrent must be greater than 0")
        if acquire_timeout_seconds <= 0 or poll_interval_seconds <= 0:
            raise ValueError("timeouts must be greater than 0")
        self._engine = engine
        self._slots = asyncio.Semaphore(max_concurrent)
        self._acquire_timeout = acquire_timeout_seconds
        self._poll_interval = poll_interval_seconds

    @asynccontextmanager
    async def hold(self, tenant_id: TenantId, profile: str, source_uri: str) -> AsyncIterator[None]:
        key = session_lock_key(tenant_id, profile, source_uri)
        deadline = asyncio.get_running_loop().time() + self._acquire_timeout
        what = f"{tenant_id}/{profile}/{source_uri}"
        await self._take_slot(deadline, what)
        try:
            conn = self._engine.connect()  # not started: no I/O, nothing to leak yet
            locked = False
            try:
                await self._acquire(conn, key, deadline, what)
                locked = True
                yield
            finally:
                await asyncio.shield(self._release(conn, key, locked=locked))
        finally:
            self._slots.release()

    async def _take_slot(self, deadline: float, what: str) -> None:
        scope = asyncio.timeout_at(deadline)
        try:
            async with scope:
                await self._slots.acquire()
        except TimeoutError:
            if scope.expired():
                raise ConcurrentIngestError(
                    f"Timed out waiting for an ingest slot for {what!r}"
                ) from None
            raise

    async def _acquire(self, conn: AsyncConnection, key: int, deadline: float, what: str) -> None:
        scope = asyncio.timeout_at(deadline)
        try:
            async with scope:
                await conn.start()
                await conn.execution_options(isolation_level="AUTOCOMMIT")
                while True:
                    granted = (
                        await conn.execute(select(func.pg_try_advisory_lock(key)))
                    ).scalar_one()
                    if granted:
                        return
                    await asyncio.sleep(self._poll_interval)
        except TimeoutError:
            if scope.expired():
                raise ConcurrentIngestError(
                    f"Timed out waiting for another ingest of {what!r} to finish"
                ) from None
            raise

    @staticmethod
    async def _release(conn: AsyncConnection, key: int, *, locked: bool) -> None:
        if conn.sync_connection is None:  # never connected: nothing to release
            return
        try:
            if locked:
                unlocked = (await conn.execute(select(func.pg_advisory_unlock(key)))).scalar_one()
                if unlocked:
                    return
                logger.error("Advisory lock %s was not held at release time", key)
            # Not locked: a cancellation may have interrupted pg_advisory_lock mid-flight, so the
            # server could still grant it (or be waiting on it): kill the session instead.
            await conn.invalidate()
        except BaseException:
            logger.exception("Could not release advisory lock %s cleanly; closing session", key)
            await conn.invalidate()
        finally:
            await conn.close()
