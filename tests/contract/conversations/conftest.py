"""Backends under contract. Add an entry to BACKENDS to run a new adapter through the suite."""

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import text

from agintis.modules.conversations.application import ThreadRepository
from agintis.modules.conversations.infrastructure.postgres import PgThreadRepository
from agintis.platform.clock import SystemClock
from agintis.platform.db import create_session_factory
from agintis.shared_kernel.ids import ThreadId
from tests.fakes.conversations import InMemoryThreadRepository
from tests.support.postgres import postgres_engines


@dataclass(frozen=True)
class Backend:
    repo: ThreadRepository
    # Makes the current lease of a thread expire NOW (no sleeping, immune to clock skew).
    expire: Callable[[ThreadId], Awaitable[None]]


type BackendFactory = Callable[[pytest.FixtureRequest], AbstractAsyncContextManager[Backend]]
LONG_AGO = timedelta(hours=1)


@asynccontextmanager
async def _in_memory(request: pytest.FixtureRequest) -> AsyncIterator[Backend]:
    repo = InMemoryThreadRepository(SystemClock())

    async def expire(thread_id: ThreadId) -> None:
        thread = repo.items[thread_id]
        repo.items[thread_id] = replace(thread, lease_until=datetime.now(UTC) - LONG_AGO)

    yield Backend(repo, expire)


@asynccontextmanager
async def _postgres(request: pytest.FixtureRequest) -> AsyncIterator[Backend]:
    url = request.getfixturevalue("pg_test_url")  # skips when the database is unreachable
    async with postgres_engines(url) as (engine, _lock_engine):
        factory = create_session_factory(engine)
        async with factory.begin() as session:
            await session.execute(text("TRUNCATE threads"))

        async def expire(thread_id: ThreadId) -> None:
            async with factory.begin() as session:
                await session.execute(
                    text(
                        "UPDATE threads SET lease_until = now() - interval '1 hour' WHERE id = :i"
                    ),
                    {"i": thread_id},
                )

        yield Backend(PgThreadRepository(factory), expire)


BACKENDS: dict[str, BackendFactory] = {"in_memory": _in_memory, "postgres": _postgres}

# Backends that need real services: skipped by default, run with `-m integration`.
INTEGRATION_BACKENDS = frozenset({"postgres"})


@pytest.fixture(
    params=[
        pytest.param(name, marks=pytest.mark.integration) if name in INTEGRATION_BACKENDS else name
        for name in sorted(BACKENDS)
    ]
)
async def backend(request: pytest.FixtureRequest) -> AsyncIterator[Backend]:
    async with BACKENDS[request.param](request) as active:
        yield active


@pytest.fixture
def repo(backend: Backend) -> ThreadRepository:
    return backend.repo


@pytest.fixture
def expire(backend: Backend) -> Callable[[ThreadId], Awaitable[None]]:
    return backend.expire
