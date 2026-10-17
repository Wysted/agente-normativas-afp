"""Shared fixtures: a migrated, disposable Postgres database (skips when unreachable)."""

import os
from typing import NoReturn

import pytest
from pydantic import ValidationError
from sqlalchemy.engine import URL

from tests.support.postgres import (
    E2E_DATABASE_NAME,
    E2E_EMBEDDING_DIM,
    HNSW_DATABASE_NAME,
    HNSW_EMBEDDING_DIM,
    admin_database_url,
    recreate_test_database,
)

REQUIRE_SERVICES_ENV = "AGINTIS_REQUIRE_SERVICES"


def skip_or_fail(reason: str) -> NoReturn:
    """Skip when a backing service is missing; fail instead when CI requires the services.

    CI sets AGINTIS_REQUIRE_SERVICES=1 so a green integration job always means the tests ran.
    Optional services (the reranker) keep using plain `pytest.skip`.
    """
    if os.environ.get(REQUIRE_SERVICES_ENV) == "1":
        pytest.fail(f"{reason} ({REQUIRE_SERVICES_ENV}=1)", pytrace=False)
    pytest.skip(reason)


@pytest.fixture(scope="session")
def pg_test_url() -> URL:
    """URL of the freshly migrated `agintis_test` database.

    Skips the requesting test when the dev Postgres is not configured or not reachable
    (`docker compose up -d --wait db`).
    """
    import psycopg

    try:
        admin_url = admin_database_url()
        return recreate_test_database(admin_url)
    except (ValidationError, psycopg.OperationalError) as exc:
        skip_or_fail(f"Postgres not available: {exc.__class__.__name__}")


@pytest.fixture(scope="session")
def pg_e2e_url() -> URL:
    """Migrated `agintis_test_e2e` database with 1024-dimension embeddings (bge-m3)."""
    import psycopg

    try:
        return recreate_test_database(admin_database_url(), E2E_DATABASE_NAME, E2E_EMBEDDING_DIM)
    except (ValidationError, psycopg.OperationalError) as exc:
        skip_or_fail(f"Postgres not available: {exc.__class__.__name__}")


@pytest.fixture(scope="session")
def pg_hnsw_url() -> URL:
    """Migrated `agintis_test_hnsw` database with 64-dimension embeddings (search plan tests)."""
    import psycopg

    try:
        return recreate_test_database(admin_database_url(), HNSW_DATABASE_NAME, HNSW_EMBEDDING_DIM)
    except (ValidationError, psycopg.OperationalError) as exc:
        skip_or_fail(f"Postgres not available: {exc.__class__.__name__}")
