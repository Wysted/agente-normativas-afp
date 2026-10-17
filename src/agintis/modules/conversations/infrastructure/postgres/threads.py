"""Postgres `ThreadRepository` (SQLAlchemy Core; one short transaction per operation)."""

from dataclasses import replace
from datetime import timedelta

import sqlalchemy as sa
from sqlalchemy import RowMapping, delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from agintis.modules.conversations.application import ThreadRepository
from agintis.modules.conversations.domain import (
    RunLease,
    RunMode,
    Thread,
    ThreadBusyError,
    ThreadNotFoundError,
    ThreadStatus,
)
from agintis.modules.conversations.infrastructure.postgres.fields import THREAD_FIELDS
from agintis.modules.conversations.infrastructure.postgres.tables import threads
from agintis.platform.db import UnitOfWork, apply_criteria
from agintis.shared_kernel.criteria import Criteria, ensure_fields, tenant_scoped
from agintis.shared_kernel.ids import TenantId, ThreadId, new_id

_BUSY = ThreadStatus.BUSY.value


def _to_thread(row: RowMapping) -> Thread:
    return Thread(
        id=ThreadId(row["id"]),
        tenant_id=TenantId(row["tenant_id"]),
        agent_id=row["agent_id"],
        title=row["title"],
        status=ThreadStatus(row["status"]),
        lease_until=row["lease_until"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _startable() -> sa.ColumnElement[bool]:
    """SQL twin of `Thread.check_can_start` returning True: keep both in sync."""
    expired = sa.and_(threads.c.status == _BUSY, threads.c.lease_until < func.now())
    resting = threads.c.status.in_((ThreadStatus.IDLE.value, ThreadStatus.INTERRUPTED.value))
    return sa.or_(resting, expired)


class PgThreadRepository(ThreadRepository):
    """Clock: every timestamp the SQL operations write (`lease_until`, `updated_at` on lease
    changes) and compare uses the DATABASE clock, so instances with skewed clocks agree on
    expiry. Only `add` stores the entity's own `created_at`/`updated_at` (app clock, set once).

    Known gap: if the commit of a lease change is applied but its acknowledgement is lost, the
    caller thinks it failed and the thread stays busy until the TTL; the checkpoint
    reconciliation of the next run and the TTL recover it.
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._factory = session_factory

    async def add(self, thread: Thread) -> None:
        values = {
            "id": thread.id,
            "tenant_id": thread.tenant_id,
            "agent_id": thread.agent_id,
            "title": thread.title,
            "status": thread.status.value,
            "lease_until": thread.lease_until,
            "created_at": thread.created_at,
            "updated_at": thread.updated_at,
        }
        async with UnitOfWork(self._factory) as uow:
            await uow.session.execute(sa.insert(threads).values(values))

    async def get(self, tenant_id: TenantId, thread_id: ThreadId) -> Thread | None:
        stmt = select(threads).where(threads.c.tenant_id == tenant_id, threads.c.id == thread_id)
        async with UnitOfWork(self._factory) as uow:
            row = (await uow.session.execute(stmt)).mappings().one_or_none()
        return None if row is None else _to_thread(row)

    async def search(self, tenant_id: TenantId, criteria: Criteria) -> list[Thread]:
        ensure_fields(criteria, self.FILTERABLE_FIELDS)
        scoped = replace(criteria, filter=tenant_scoped(tenant_id, criteria.filter))
        stmt = apply_criteria(select(threads), scoped, THREAD_FIELDS)
        async with UnitOfWork(self._factory) as uow:
            rows = (await uow.session.execute(stmt)).mappings().all()
        return [_to_thread(row) for row in rows]

    async def delete(self, tenant_id: TenantId, thread_id: ThreadId) -> bool:
        leased = sa.and_(threads.c.status == _BUSY, threads.c.lease_until >= func.now())
        stmt = (
            delete(threads)
            .where(threads.c.tenant_id == tenant_id, threads.c.id == thread_id, sa.not_(leased))
            .returning(threads.c.id)
        )
        async with UnitOfWork(self._factory) as uow:
            return (await uow.session.execute(stmt)).scalar_one_or_none() is not None

    async def acquire_run_lease(
        self, tenant_id: TenantId, thread_id: ThreadId, mode: RunMode, ttl: timedelta
    ) -> RunLease:
        token = new_id()
        # ONE conditional UPDATE decides the race: of two concurrent callers, the second one's
        # WHERE no longer matches once the first commits (row lock + re-check).
        stmt = (
            update(threads)
            .where(threads.c.tenant_id == tenant_id, threads.c.id == thread_id, _startable())
            .values(
                status=_BUSY,
                lease_until=func.now() + sa.literal(ttl, sa.Interval),
                lease_token=token,
                updated_at=func.now(),
            )
            .returning(threads.c.agent_id, threads.c.lease_until)
        )
        async with UnitOfWork(self._factory) as uow:
            granted = (await uow.session.execute(stmt)).one_or_none()
            if granted is not None:
                return RunLease(
                    thread_id=thread_id,
                    tenant_id=tenant_id,
                    agent_id=granted.agent_id,
                    mode=mode,
                    token=token,
                    lease_until=granted.lease_until,
                )
            # Nothing updated: read the row (and the database clock) to say why.
            row = (
                (
                    await uow.session.execute(
                        select(threads, func.now().label("db_now")).where(
                            threads.c.tenant_id == tenant_id, threads.c.id == thread_id
                        )
                    )
                )
                .mappings()
                .one_or_none()
            )
        if row is None:
            raise ThreadNotFoundError
        _to_thread(row).check_can_start(row["db_now"])
        # It became startable between the UPDATE and the read (the other run just ended).
        raise ThreadBusyError

    async def release_run_lease(
        self, tenant_id: TenantId, thread_id: ThreadId, token: str, outcome: ThreadStatus
    ) -> bool:
        stmt = (
            update(threads)
            .where(
                threads.c.tenant_id == tenant_id,
                threads.c.id == thread_id,
                threads.c.lease_token == token,
            )
            .values(status=outcome.value, lease_until=None, lease_token=None, updated_at=func.now())
            .returning(threads.c.id)
        )
        async with UnitOfWork(self._factory) as uow:
            return (await uow.session.execute(stmt)).scalar_one_or_none() is not None
