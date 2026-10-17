"""Field maps of the Postgres adapter: allowlists, drift guards and the new settings."""

from typing import Any

import pytest
from pydantic import ValidationError

from agintis.bootstrap.settings import DatabaseSettings
from agintis.modules.knowledge.domain import (
    RESERVED_CHUNK_FIELDS,
    MetadataField,
    chunk_filterable_fields,
)
from agintis.modules.knowledge.infrastructure.postgres import fields as pg_fields
from agintis.modules.knowledge.infrastructure.postgres.fields import (
    CHUNK_CORE_FIELDS,
    chunk_fields,
)

SCHEMA = (MetadataField("article", "str", filterable=True), MetadataField("note", "str"))


def test_core_columns_are_exactly_the_reserved_metadata_names() -> None:
    assert set(CHUNK_CORE_FIELDS) == RESERVED_CHUNK_FIELDS


def test_mapped_fields_equal_the_domain_allowlist() -> None:
    assert set(chunk_fields(SCHEMA)) == chunk_filterable_fields(SCHEMA)
    assert "note" not in chunk_fields(SCHEMA)  # declared but not filterable


def test_a_declared_field_shadowing_a_core_column_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # MetadataField already rejects reserved names; this is the second line of defense.
    shadow = MetadataField("article", "str", filterable=True)
    monkeypatch.setattr(pg_fields, "chunk_metadata_schema", lambda declared: {"text": shadow})
    with pytest.raises(ValueError, match="shadow core chunk fields"):
        chunk_fields(SCHEMA)


class TestDatabaseSettings:
    def test_defaults(self) -> None:
        settings = DatabaseSettings(_env_file=None)
        assert settings.max_ingest_locks == 2
        assert settings.ingest_lock_timeout_seconds == 300
        assert settings.connect_timeout_seconds == 5
        assert settings.statement_timeout_ms is None
        assert (settings.hnsw_ef_search, settings.hnsw_max_scan_tuples) == (40, 20_000)
        assert settings.search_timeout_ms == 10_000
        assert settings.read_only is False

    def test_flat_environment_names(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DATABASE_MAX_INGEST_LOCKS", "4")
        monkeypatch.setenv("DATABASE_INGEST_LOCK_TIMEOUT_SECONDS", "60")
        monkeypatch.setenv("DATABASE_STATEMENT_TIMEOUT_MS", "30000")
        monkeypatch.setenv("DATABASE_HNSW_EF_SEARCH", "100")
        settings = DatabaseSettings(_env_file=None)
        assert settings.max_ingest_locks == 4
        assert settings.ingest_lock_timeout_seconds == 60
        assert settings.statement_timeout_ms == 30000
        assert settings.hnsw_ef_search == 100

    def test_an_empty_statement_timeout_means_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DATABASE_STATEMENT_TIMEOUT_MS", "")
        assert DatabaseSettings(_env_file=None).statement_timeout_ms is None

    @pytest.mark.parametrize(
        "overrides",
        [
            {"max_ingest_locks": 0},
            {"ingest_lock_timeout_seconds": 0},
            {"connect_timeout_seconds": 0},
            {"statement_timeout_ms": 0},
            {"hnsw_ef_search": 0},
            {"hnsw_ef_search": 1001},  # pgvector's upper bound
            {"hnsw_max_scan_tuples": 0},
            {"search_timeout_ms": 0},
        ],
    )
    def test_invalid_values_are_rejected(self, overrides: dict[str, Any]) -> None:
        with pytest.raises(ValidationError):
            DatabaseSettings(_env_file=None, **overrides)
