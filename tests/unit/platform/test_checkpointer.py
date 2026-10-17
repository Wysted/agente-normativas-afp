"""Checkpointer wrapper without a database: construction, lifecycle, the boot check's verdicts."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
from pydantic import SecretStr, ValidationError

from agintis.bootstrap.settings import (
    DEFAULT_RUN_DEADLINE_SECONDS,
    LEASE_MARGIN_SECONDS,
    CheckpointerSettings,
    ThreadsSettings,
)
from agintis.platform.checkpointer import (
    EXPECTED_SCHEMA_VERSION,
    Checkpointer,
    CheckpointerSetupError,
    PooledSaver,
)

DSN = SecretStr("postgresql://u:pw@localhost:5432/db")


def _checkpointer() -> Checkpointer:
    return Checkpointer(
        DSN, min_size=1, max_size=2, acquire_timeout_seconds=1, connect_timeout_seconds=1
    )


class _Cursor:
    def __init__(self, row: dict[str, Any] | None) -> None:
        self._row = row

    async def fetchone(self) -> dict[str, Any] | None:
        return self._row


class _Conn:
    def __init__(self, table: str | None, version: int | None) -> None:
        self._table = table
        self._version = version

    async def execute(self, sql: str) -> _Cursor:
        if "to_regclass" in sql:
            return _Cursor({"name": self._table})
        return _Cursor({"version": self._version})


class _Pool:
    def __init__(self, conn: _Conn | Exception) -> None:
        self._conn = conn

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[_Conn]:
        if isinstance(self._conn, Exception):
            raise self._conn
        yield self._conn


def _with_pool(pool: _Pool) -> Checkpointer:
    return Checkpointer(
        DSN,
        min_size=1,
        max_size=2,
        acquire_timeout_seconds=1,
        connect_timeout_seconds=1,
        pool=pool,  # type: ignore[arg-type]
    )


def test_it_can_be_built_without_a_running_event_loop() -> None:
    """Container.build is synchronous: neither the pool nor the saver may need a loop."""
    assert isinstance(_checkpointer().saver, PooledSaver)


async def test_pooled_saver_sets_every_attribute_the_upstream_constructor_sets() -> None:
    """Guards the constructor bypass: an upstream release that adds state breaks this test."""
    pool: Any = AsyncConnectionPool("postgresql://x@localhost/x", min_size=1, open=False)
    upstream = AsyncPostgresSaver(pool)

    ours = PooledSaver(pool)
    ours.bind_loop()

    assert vars(ours).keys() == vars(upstream).keys()
    assert ours.loop is upstream.loop


async def test_open_binds_the_loop_and_close_is_safe_even_if_never_opened() -> None:
    checkpointer = _checkpointer()

    await checkpointer.aclose()  # never opened: must not raise


async def test_verify_accepts_the_expected_schema() -> None:
    checkpointer = _with_pool(_Pool(_Conn("checkpoint_migrations", EXPECTED_SCHEMA_VERSION)))

    await checkpointer.verify()


async def test_verify_rejects_a_schema_newer_than_the_build() -> None:
    checkpointer = _with_pool(_Pool(_Conn("checkpoint_migrations", EXPECTED_SCHEMA_VERSION + 1)))

    with pytest.raises(CheckpointerSetupError, match="NEWER than this build") as info:
        await checkpointer.verify()

    assert "checkpointer_setup" not in str(info.value)  # running setup would not help


async def test_verify_asks_for_the_setup_command_when_the_tables_are_missing() -> None:
    checkpointer = _with_pool(_Pool(_Conn(None, None)))

    with pytest.raises(CheckpointerSetupError, match="checkpointer_setup"):
        await checkpointer.verify()


async def test_verify_rejects_an_outdated_schema() -> None:
    checkpointer = _with_pool(_Pool(_Conn("checkpoint_migrations", EXPECTED_SCHEMA_VERSION - 1)))

    with pytest.raises(CheckpointerSetupError, match=r"version \d+ but this build expects"):
        await checkpointer.verify()


async def test_verify_reports_an_unreachable_database_without_leaking_details() -> None:
    checkpointer = _with_pool(_Pool(OSError("password=secret host=db")))

    with pytest.raises(CheckpointerSetupError, match="not reachable: OSError") as info:
        await checkpointer.verify()

    assert "secret" not in str(info.value)


def test_settings_defaults_are_small() -> None:
    settings = CheckpointerSettings(_env_file=None)

    assert (settings.enabled, settings.pool_min_size, settings.pool_max_size) == (True, 1, 4)


def test_pool_min_cannot_exceed_max() -> None:
    with pytest.raises(ValidationError, match="pool_min_size"):
        CheckpointerSettings(_env_file=None, pool_min_size=5, pool_max_size=2)


def test_checkpointer_env_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHECKPOINTER_ENABLED", "false")
    monkeypatch.setenv("THREADS_LEASE_TTL_SECONDS", "300")
    monkeypatch.setenv("THREADS_RUN_DEADLINE_SECONDS", "120")

    assert CheckpointerSettings(_env_file=None).enabled is False
    threads = ThreadsSettings(_env_file=None)
    assert (threads.lease_ttl_seconds, threads.run_deadline_seconds) == (300, 120)


def test_the_default_deadline_leaves_the_lease_margin() -> None:
    threads = ThreadsSettings(_env_file=None)

    assert threads.run_deadline_seconds == DEFAULT_RUN_DEADLINE_SECONDS
    assert threads.run_deadline_seconds < threads.lease_ttl_seconds - LEASE_MARGIN_SECONDS


@pytest.mark.parametrize(
    ("ttl", "deadline"),
    [(100, 100), (100, 71), (100, 500), (900, 870)],  # the deadline must be < ttl - margin
)
def test_the_run_deadline_must_end_before_the_lease(ttl: float, deadline: float) -> None:
    with pytest.raises(ValidationError, match="run_deadline_seconds"):
        ThreadsSettings(_env_file=None, lease_ttl_seconds=ttl, run_deadline_seconds=deadline)


def test_a_deadline_just_inside_the_margin_is_accepted() -> None:
    ThreadsSettings(_env_file=None, lease_ttl_seconds=100, run_deadline_seconds=69)
