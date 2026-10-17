"""Postgres-side guarantees of the retrieval setup: text search config check, read-only engine."""

from collections.abc import AsyncIterator

import pytest
from pydantic import SecretStr
from sqlalchemy import text
from sqlalchemy.engine import URL
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncEngine

from agintis.modules.knowledge.infrastructure.postgres import (
    TextSearchConfigError,
    verify_text_search_config,
)
from agintis.platform.db import create_engine
from tests.support.postgres import postgres_engines

pytestmark = pytest.mark.integration


@pytest.fixture
async def engine(pg_test_url: URL) -> AsyncIterator[AsyncEngine]:
    async with postgres_engines(pg_test_url) as (pooled, _):
        yield pooled


@pytest.mark.parametrize("config", ["public.es_unaccent", "es_unaccent"])
async def test_the_configuration_tsv_is_built_with_passes(engine: AsyncEngine, config: str) -> None:
    await verify_text_search_config(engine, config)


async def test_another_existing_configuration_is_a_mismatch(engine: AsyncEngine) -> None:
    for other in ("simple", "pg_catalog.spanish"):
        with pytest.raises(TextSearchConfigError, match="differs"):
            await verify_text_search_config(engine, other)


async def test_a_configuration_that_does_not_exist_is_reported(engine: AsyncEngine) -> None:
    with pytest.raises(TextSearchConfigError, match="does not exist"):
        await verify_text_search_config(engine, "public.nope")
    # The failed lookup must not poison the pool: the next check still works.
    await verify_text_search_config(engine, "public.es_unaccent")


async def test_a_read_only_engine_can_search_but_the_server_refuses_writes(
    pg_test_url: URL,
) -> None:
    """What the eval CLI relies on: `DATABASE_READ_ONLY` is enforced by Postgres itself."""
    secret = SecretStr(pg_test_url.render_as_string(hide_password=False))
    engine = create_engine(
        secret,
        pool_size=1,
        max_overflow=0,
        pool_timeout_seconds=5,
        pool_recycle_seconds=60,
        read_only=True,
    )
    try:
        async with engine.connect() as conn:
            assert await conn.scalar(text("SELECT count(*) FROM chunks")) is not None
            await conn.rollback()
            with pytest.raises(DBAPIError, match="read-only transaction"):
                await conn.execute(text("DELETE FROM chunks"))
    finally:
        await engine.dispose()
