"""Backends under contract. Add an entry to BACKENDS to run a new adapter through the suite."""

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass

import pytest

from agintis.modules.knowledge.domain import ChunkRepository, DocumentRepository
from agintis.modules.knowledge.infrastructure.postgres import (
    PgChunkRepository,
    PgDocumentRepository,
)
from agintis.platform.db import create_session_factory
from tests.fakes.knowledge import InMemoryChunkRepository, InMemoryDocumentRepository
from tests.support.postgres import postgres_engines, truncate_all


@dataclass(frozen=True)
class Repos:
    documents: DocumentRepository
    chunks: ChunkRepository


type BackendFactory = Callable[[pytest.FixtureRequest], AbstractAsyncContextManager[Repos]]


@asynccontextmanager
async def _in_memory(request: pytest.FixtureRequest) -> AsyncIterator[Repos]:
    documents = InMemoryDocumentRepository()
    yield Repos(documents, InMemoryChunkRepository(documents))


@asynccontextmanager
async def _postgres(request: pytest.FixtureRequest) -> AsyncIterator[Repos]:
    url = request.getfixturevalue("pg_test_url")  # skips when the database is unreachable
    async with postgres_engines(url) as (engine, _lock_engine):
        factory = create_session_factory(engine)
        await truncate_all(factory)
        yield Repos(PgDocumentRepository(factory), PgChunkRepository(factory))


# Each factory yields fresh, empty repositories that serve only READY documents' chunks.
BACKENDS: dict[str, BackendFactory] = {"in_memory": _in_memory, "postgres": _postgres}

# Backends that need real services: skipped by default, run with `-m integration`.
INTEGRATION_BACKENDS = frozenset({"postgres"})


@pytest.fixture(
    params=[
        pytest.param(name, marks=pytest.mark.integration) if name in INTEGRATION_BACKENDS else name
        for name in sorted(BACKENDS)
    ]
)
async def repos(request: pytest.FixtureRequest) -> AsyncIterator[Repos]:
    async with BACKENDS[request.param](request) as backend:
        yield backend
