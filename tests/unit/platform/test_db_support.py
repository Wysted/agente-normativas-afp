"""Unit of Work, startup checks and engine factories (no database)."""

import asyncio
from typing import Any, cast

import pytest
from pydantic import SecretStr
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import NullPool

from agintis.platform.db import (
    DatabaseCheckError,
    UnitOfWork,
    check_database,
    create_engine,
    create_unpooled_engine,
    parse_version,
)

URL = SecretStr("postgresql+psycopg://u:pw@localhost:5432/db")


class FakeSession:
    def __init__(self) -> None:
        self.events: list[str] = []

    async def begin(self) -> None:
        self.events.append("begin")

    async def commit(self) -> None:
        self.events.append("commit")

    async def rollback(self) -> None:
        self.events.append("rollback")

    async def close(self) -> None:
        self.events.append("close")


def _factory(session: FakeSession) -> async_sessionmaker[AsyncSession]:
    return cast("async_sessionmaker[AsyncSession]", lambda: session)


class TestUnitOfWork:
    async def test_commits_on_success_and_always_closes(self) -> None:
        session = FakeSession()
        async with UnitOfWork(_factory(session)) as uow:
            assert cast("object", uow.session) is session
        assert session.events == ["begin", "commit", "close"]

    async def test_rolls_back_on_error_and_cancellation(self) -> None:
        for error in (RuntimeError("boom"), KeyboardInterrupt()):
            session = FakeSession()
            with pytest.raises(type(error)):
                async with UnitOfWork(_factory(session)):
                    raise error
            assert session.events == ["begin", "rollback", "close"]

    async def test_session_is_unavailable_outside_the_block(self) -> None:
        uow = UnitOfWork(_factory(FakeSession()))
        with pytest.raises(RuntimeError, match="not active"):
            _ = uow.session


class TestVersions:
    def test_parse_version(self) -> None:
        assert parse_version("0.8.7") == (0, 8, 7)
        assert parse_version("0.8.10") > parse_version("0.8.4")  # numeric, not lexicographic
        assert parse_version("0.9.0-dev") == (0, 9, 0)

    @staticmethod
    def _patch(
        monkeypatch: pytest.MonkeyPatch,
        *,
        version: str | None = "0.8.7",
        embedding: str | None = "vector(1024)",
        revisions: list[str] | None = None,
    ) -> None:
        async def installed(engine: Any, name: str) -> str | None:
            return version

        async def schema(engine: Any) -> tuple[str | None, list[str] | None]:
            return embedding, ["0001"] if revisions is None else revisions

        monkeypatch.setattr("agintis.platform.db.checks.installed_extension_version", installed)
        monkeypatch.setattr("agintis.platform.db.checks._read_schema", schema)

    async def test_check_database_returns_the_installed_version(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._patch(monkeypatch)
        assert await check_database(None, min_vector_version=(0, 8, 4)) == "0.8.7"  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("installed", "message"), [("0.8.1", "too old"), (None, "not installed")]
    )
    async def test_check_database_rejects_old_or_missing_pgvector(
        self, monkeypatch: pytest.MonkeyPatch, installed: str | None, message: str
    ) -> None:
        self._patch(monkeypatch, version=installed)
        with pytest.raises(DatabaseCheckError, match=message):
            await check_database(None, min_vector_version=(0, 8, 4))  # type: ignore[arg-type]

    async def test_check_database_reports_unreachable_without_leaking_the_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def down(engine: Any, name: str) -> str | None:
            raise ConnectionError("could not connect to postgresql://u:pw@host/db")

        monkeypatch.setattr("agintis.platform.db.checks.installed_extension_version", down)
        with pytest.raises(DatabaseCheckError, match="not reachable") as info:
            await check_database(None, min_vector_version=(0, 8, 4))  # type: ignore[arg-type]
        assert "pw" not in str(info.value)

    async def test_check_database_is_bounded_by_a_timeout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def hang(engine: Any, name: str) -> str | None:
            await asyncio.sleep(60)
            return None

        monkeypatch.setattr("agintis.platform.db.checks.installed_extension_version", hang)
        with pytest.raises(DatabaseCheckError, match="not reachable") as info:
            await check_database(None, min_vector_version=(0, 8, 4), timeout_seconds=0.05)  # type: ignore[arg-type]
        assert "TimeoutError" in str(info.value)

    async def test_schema_matching_the_build_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch(monkeypatch)
        version = await check_database(
            None,  # type: ignore[arg-type]
            min_vector_version=(0, 8, 4),
            expected_embedding_dim=1024,
            expected_revision="0001",
        )
        assert version == "0.8.7"

    @pytest.mark.parametrize(
        ("embedding", "match"),
        [
            ("vector(384)", r"vector\(384\).*vector\(1024\).*new migration"),
            ("vector", r"vector.*vector\(1024\)"),  # unconstrained width
            (None, "chunks.embedding does not exist"),
        ],
    )
    async def test_embedding_width_must_match_the_configured_dimension(
        self, monkeypatch: pytest.MonkeyPatch, embedding: str | None, match: str
    ) -> None:
        self._patch(monkeypatch, embedding=embedding)
        with pytest.raises(DatabaseCheckError, match=match):
            await check_database(
                None,  # type: ignore[arg-type]
                min_vector_version=(0, 8, 4),
                expected_embedding_dim=1024,
            )

    @pytest.mark.parametrize(
        ("revisions", "match"),
        [
            (["0000"], "revision 0000 but this build expects 0001.*alembic upgrade head"),
            ([], "revision none but this build expects 0001"),
            (["0001", "0002"], "revision 0001, 0002"),
        ],
    )
    async def test_alembic_revision_must_be_the_head_of_this_build(
        self, monkeypatch: pytest.MonkeyPatch, revisions: list[str], match: str
    ) -> None:
        self._patch(monkeypatch, revisions=revisions)
        with pytest.raises(DatabaseCheckError, match=match):
            await check_database(
                None,  # type: ignore[arg-type]
                min_vector_version=(0, 8, 4),
                expected_revision="0001",
            )

    async def test_unmigrated_database_is_reported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def installed(engine: Any, name: str) -> str | None:
            return "0.8.7"

        async def schema(engine: Any) -> tuple[str | None, list[str] | None]:
            return None, None

        monkeypatch.setattr("agintis.platform.db.checks.installed_extension_version", installed)
        monkeypatch.setattr("agintis.platform.db.checks._read_schema", schema)
        with pytest.raises(DatabaseCheckError, match=r"not migrated.*alembic upgrade head"):
            await check_database(
                None,  # type: ignore[arg-type]
                min_vector_version=(0, 8, 4),
                expected_revision="0001",
            )


class TestEngines:
    async def test_pooled_engine_uses_the_given_limits_and_is_lazy(self) -> None:
        engine = create_engine(
            URL, pool_size=2, max_overflow=1, pool_timeout_seconds=5, pool_recycle_seconds=60
        )
        try:
            assert engine.pool.size() == 2  # type: ignore[attr-defined]
            assert engine.pool.status().startswith("Pool size: 2")  # no connection was opened
            assert engine.dialect.driver == "psycopg"
        finally:
            await engine.dispose()

    async def test_unpooled_engine_never_keeps_connections(self) -> None:
        engine = create_unpooled_engine(URL)
        try:
            assert isinstance(engine.pool, NullPool)
        finally:
            await engine.dispose()

    @staticmethod
    async def _connect_params(engine: AsyncEngine) -> dict[str, Any]:
        captured: dict[str, Any] = {}

        class Stop(Exception):
            pass

        @event.listens_for(engine.sync_engine, "do_connect")
        def grab(dialect: Any, conn_rec: Any, cargs: Any, cparams: dict[str, Any]) -> None:
            captured.update(cparams)
            raise Stop  # never actually connect

        try:
            with pytest.raises(Stop):
                await engine.connect().start()
        finally:
            await engine.dispose()
        return captured

    async def test_connect_arguments_bound_connect_time_and_detect_dead_peers(self) -> None:
        pooled = create_engine(
            URL,
            pool_size=1,
            max_overflow=0,
            pool_timeout_seconds=1,
            pool_recycle_seconds=60,
            connect_timeout_seconds=2.2,
            statement_timeout_ms=1500,
        )
        unpooled = create_unpooled_engine(URL, connect_timeout_seconds=2.2)
        for engine in (pooled, unpooled):
            params = await self._connect_params(engine)
            assert params["connect_timeout"] == 3  # rounded UP to whole seconds, never 0
            assert params["keepalives"] == 1
            assert params["keepalives_idle"] > 0

    def test_statement_timeout_applies_only_to_the_pooled_engine(self) -> None:
        from agintis.platform.db.engine import _connect_args

        assert _connect_args(5, 1500)["options"] == "-c statement_timeout=1500"
        assert "options" not in _connect_args(5, None)
        assert _connect_args(0.1, None)["connect_timeout"] == 1

    def test_read_only_opens_every_connection_read_only(self) -> None:
        from agintis.platform.db.engine import _connect_args

        assert _connect_args(5, None, True)["options"] == "-c default_transaction_read_only=on"
        assert _connect_args(5, 1500, True)["options"] == (
            "-c statement_timeout=1500 -c default_transaction_read_only=on"
        )
