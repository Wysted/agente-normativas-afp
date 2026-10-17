"""Startup checks against the database."""

import asyncio
import re
from collections.abc import Collection

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine


class DatabaseCheckError(RuntimeError):
    """The database is unreachable or does not meet a requirement."""


def parse_version(raw: str) -> tuple[int, ...]:
    """'0.8.7' -> (0, 8, 7); trailing non-numeric parts (e.g. '0.8.7-dev') are ignored."""
    parts = re.findall(r"\d+", raw.split("-")[0])
    return tuple(int(p) for p in parts)


async def installed_extension_version(engine: AsyncEngine, name: str) -> str | None:
    async with engine.connect() as conn:
        result = await conn.execute(
            text("SELECT extversion FROM pg_extension WHERE extname = :name"), {"name": name}
        )
        return result.scalar_one_or_none()


# `to_regclass` yields NULL (not an error) when the table does not exist.
_EMBEDDING_TYPE_SQL = text(
    "SELECT format_type(a.atttypid, a.atttypmod) FROM pg_attribute a "
    "WHERE a.attrelid = to_regclass('chunks') AND a.attname = 'embedding' AND NOT a.attisdropped"
)
_REVISIONS_SQL = text("SELECT version_num FROM alembic_version")
DEFAULT_CHECK_TIMEOUT_SECONDS = 15.0


async def _read_schema(engine: AsyncEngine) -> tuple[str | None, list[str] | None]:
    """(declared type of chunks.embedding, applied alembic revisions); None = not there."""
    async with engine.connect() as conn:
        embedding_type = (await conn.execute(_EMBEDDING_TYPE_SQL)).scalar_one_or_none()
        revisions: list[str] | None = None
        if (await conn.execute(text("SELECT to_regclass('alembic_version')"))).scalar_one():
            revisions = [row[0] for row in await conn.execute(_REVISIONS_SQL)]
        return embedding_type, revisions


async def check_database(
    engine: AsyncEngine,
    *,
    min_vector_version: tuple[int, ...],
    expected_embedding_dim: int | None = None,
    expected_revision: str | None = None,
    timeout_seconds: float = DEFAULT_CHECK_TIMEOUT_SECONDS,
) -> str:
    """Verify connectivity, the pgvector version and (optionally) the migrated schema.

    * `expected_embedding_dim`: `chunks.embedding` must be exactly `vector(<dim>)`. A mismatch
      (model changed without a migration) would otherwise only surface at the first insert.
    * `expected_revision`: the database must be at this Alembic revision (the current head).
    The whole check is bounded by `timeout_seconds`. Returns the installed vector version.
    """
    try:
        async with asyncio.timeout(timeout_seconds):
            version = await installed_extension_version(engine, "vector")
            schema = await _read_schema(engine)
    except Exception as exc:
        raise DatabaseCheckError(f"Database is not reachable: {exc.__class__.__name__}") from exc
    if version is None:
        raise DatabaseCheckError(
            "The 'vector' extension is not installed (CREATE EXTENSION vector)"
        )
    if parse_version(version) < min_vector_version:
        needed = ".".join(map(str, min_vector_version))
        raise DatabaseCheckError(
            f"pgvector {version} is too old; need >= {needed} "
            "(HNSW fixes) - upgrade the image and run ALTER EXTENSION vector UPDATE"
        )
    _check_schema(schema, expected_embedding_dim, expected_revision)
    return version


def _check_schema(
    schema: tuple[str | None, list[str] | None],
    expected_embedding_dim: int | None,
    expected_revision: str | None,
) -> None:
    embedding_type, revisions = schema
    if expected_revision is not None:
        _require_revision(revisions, expected_revision)
    if expected_embedding_dim is not None:
        expected = f"vector({expected_embedding_dim})"
        if embedding_type is None:
            raise DatabaseCheckError(
                "Table chunks.embedding does not exist: run `alembic upgrade head`"
            )
        if embedding_type != expected:
            raise DatabaseCheckError(
                f"chunks.embedding is {embedding_type} but EMBEDDINGS_EXPECTED_DIM requires "
                f"{expected}: changing the embedding model needs a new migration (alter the "
                "column, rebuild the HNSW index) and re-embedding every chunk"
            )


def _require_revision(revisions: Collection[str] | None, expected: str) -> None:
    if revisions is None:
        raise DatabaseCheckError(
            "The database is not migrated (no alembic_version table): run `alembic upgrade head`"
        )
    if set(revisions) != {expected}:
        found = ", ".join(sorted(revisions)) or "none"
        raise DatabaseCheckError(
            f"Database schema is at revision {found} but this build expects {expected}: "
            "run `alembic upgrade head` (or deploy the matching build)"
        )
