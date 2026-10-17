"""Real AsyncPostgresSaver on the compose database: setup, boot check, memory, deletion."""

import asyncio
from collections.abc import AsyncIterator, Iterator
from typing import Any

import psycopg
import pytest
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from pydantic import SecretStr, ValidationError
from sqlalchemy.engine import URL

from agintis.agent_kit.runtime import LangGraphRunner
from agintis.bootstrap.adapters import CheckpointerStateStore
from agintis.modules.agents_catalog.api import Clarification, RunCompleted, RunInput
from agintis.platform.checkpointer import (
    EXPECTED_SCHEMA_VERSION,
    SETUP_LOCK_KEY,
    Checkpointer,
    CheckpointerSetupError,
)
from agintis.shared_kernel.ids import RunId, TenantId, ThreadId
from tests.conftest import skip_or_fail
from tests.fakes.graphs import StreamingModel, asking_graph, llm_graph, threaded
from tests.support.postgres import (
    admin_database_url,
    drop_database,
    libpq_dsn,
    recreate_test_database,
)

pytestmark = pytest.mark.integration

DATABASE = "agintis_test_checkpointer"
THREAD = ThreadId("thread-1")
OTHER = ThreadId("thread-2")


@pytest.fixture(scope="module")
def fresh_url() -> Iterator[URL]:
    """A migrated database WITHOUT the checkpoint tables (what a new deployment looks like)."""
    try:
        url = recreate_test_database(admin_database_url(), DATABASE)
    except (ValidationError, psycopg.OperationalError) as exc:
        skip_or_fail(f"Postgres not available: {exc.__class__.__name__}")
    try:
        yield url
    finally:
        drop_database(admin_database_url(), DATABASE)


def _checkpointer(url: URL) -> Checkpointer:
    return Checkpointer(
        SecretStr(libpq_dsn(url)),
        min_size=1,
        max_size=3,
        acquire_timeout_seconds=5,
        connect_timeout_seconds=3,
    )


@pytest.fixture
async def checkpointer(fresh_url: URL) -> AsyncIterator[Checkpointer]:
    cp = _checkpointer(fresh_url)
    await cp.open(wait=True)
    try:
        yield cp
    finally:
        await cp.aclose()


def _run_input(
    thread: ThreadId,
    message: str | None = "hi",
    run: str = "a" * 32,
    resume: dict[str, object] | None = None,
) -> RunInput:
    return RunInput(TenantId("acme"), message, RunId(run), thread, resume)


async def _events(runner: LangGraphRunner, run_input: RunInput) -> list[Any]:
    return [event async for event in runner.astream(run_input)]


def _key(thread: ThreadId) -> RunnableConfig:
    return {"configurable": {"thread_id": thread}}


async def test_boot_check_lifecycle_missing_then_ready_then_outdated(fresh_url: URL) -> None:
    cp = _checkpointer(fresh_url)
    await cp.open(wait=True)
    try:
        with pytest.raises(CheckpointerSetupError, match="do not exist"):
            await cp.verify()

        await cp.setup()
        await cp.setup()  # idempotent
        await cp.verify()

        async with await psycopg.AsyncConnection.connect(
            libpq_dsn(fresh_url), autocommit=True
        ) as conn:
            await conn.execute(
                "DELETE FROM checkpoint_migrations WHERE v = %s", (EXPECTED_SCHEMA_VERSION,)
            )
        with pytest.raises(CheckpointerSetupError, match="expects"):
            await cp.verify()
        await cp.setup()  # setup also migrates an outdated schema
        await cp.verify()
    finally:
        await cp.aclose()


async def test_verify_reports_an_unreachable_database(fresh_url: URL) -> None:
    cp = Checkpointer(
        SecretStr(libpq_dsn(fresh_url.set(port=1))),
        min_size=1,
        max_size=1,
        acquire_timeout_seconds=1,
        connect_timeout_seconds=1,
    )
    await cp.open(wait=False)
    try:
        with pytest.raises(CheckpointerSetupError, match="not reachable"):
            await cp.verify(timeout_seconds=2)
    finally:
        await cp.aclose()


async def test_multi_turn_memory_survives_a_new_pool_and_is_isolated_per_thread(
    fresh_url: URL, checkpointer: Checkpointer
) -> None:
    await checkpointer.setup()
    model = StreamingModel(pieces=["ok"])
    toy = threaded(lambda s: llm_graph(model, checkpointer=s), "chat", checkpointer.saver)
    graph, runner = toy.graph, toy.runner

    await _events(runner, _run_input(THREAD, "one", "a" * 32))
    await _events(runner, _run_input(THREAD, "two", "b" * 32))
    await _events(runner, _run_input(OTHER, "elsewhere", "c" * 32))

    def contents(messages: list[BaseMessage]) -> list[Any]:
        return [m.content for m in messages]

    state = await graph.aget_state(_key(THREAD))
    assert contents(state.values["messages"]) == ["one", "ok", "two", "ok"]
    assert contents((await graph.aget_state(_key(OTHER))).values["messages"]) == ["elsewhere", "ok"]

    # A brand new pool and graph (a restart) sees the same conversation.
    restarted = _checkpointer(fresh_url)
    await restarted.open(wait=True)
    try:
        again = threaded(lambda s: llm_graph(model, checkpointer=s), "chat", restarted.saver)
        state = await again.graph.aget_state(_key(THREAD))
        assert contents(state.values["messages"]) == ["one", "ok", "two", "ok"]
    finally:
        await restarted.aclose()


async def test_interrupt_and_resume_across_a_restart(
    fresh_url: URL, checkpointer: Checkpointer
) -> None:
    await checkpointer.setup()
    thread = ThreadId("interrupt-1")
    runner = threaded(lambda s: asking_graph("Which one?", s), "ask", checkpointer.saver).runner

    first = await _events(runner, _run_input(thread, "start", "a" * 32))
    asked = next(e for e in first if isinstance(e, Clarification))
    assert asked.question == "Which one?"
    assert first[-1] == RunCompleted("interrupted")

    restarted = _checkpointer(fresh_url)
    await restarted.open(wait=True)
    try:
        again = threaded(lambda s: asking_graph("Which one?", s), "ask", restarted.saver)
        graph = again.graph
        resumed = await _events(
            again.runner,
            _run_input(thread, None, "b" * 32, resume={asked.interrupt_id: "blue"}),
        )
        assert resumed[-1] == RunCompleted("stop")
        state = await graph.aget_state(_key(thread))
        assert state.values["messages"][-1].content == "you chose blue"
    finally:
        await restarted.aclose()


async def test_delete_thread_removes_only_that_threads_checkpoints(
    checkpointer: Checkpointer,
) -> None:
    await checkpointer.setup()
    model = StreamingModel(pieces=["ok"])
    toy = threaded(lambda s: llm_graph(model, checkpointer=s), "chat", checkpointer.saver)
    graph, runner = toy.graph, toy.runner
    doomed, kept = ThreadId("delete-doomed"), ThreadId("delete-kept")
    await _events(runner, _run_input(doomed, "one", "a" * 32))
    await _events(runner, _run_input(kept, "two", "b" * 32))

    await CheckpointerStateStore(checkpointer).delete(doomed)

    assert (await graph.aget_state(_key(doomed))).values == {}
    assert len((await graph.aget_state(_key(kept))).values["messages"]) == 2


async def test_setup_waits_for_a_concurrent_setup(
    fresh_url: URL, checkpointer: Checkpointer
) -> None:
    """Two deploy jobs must not run the saver's migrations at the same time."""
    holder = await psycopg.AsyncConnection.connect(libpq_dsn(fresh_url), autocommit=True)
    try:
        await holder.execute("SELECT pg_advisory_lock(%s)", (SETUP_LOCK_KEY,))
        waiting = asyncio.create_task(checkpointer.setup())
        await asyncio.sleep(0.3)
        assert not waiting.done()  # blocked on the lock the "other job" holds
    finally:
        await holder.close()  # the session lock goes away with the connection

    await asyncio.wait_for(waiting, 10)
    await checkpointer.verify()


async def test_pending_interrupts_come_from_the_persisted_checkpoint(
    fresh_url: URL, checkpointer: Checkpointer
) -> None:
    await checkpointer.setup()
    thread = ThreadId("pending-1")
    runner = LangGraphRunner(
        "ask",
        "d",
        asking_graph("Which one?"),
        thread_graph=asking_graph("Which one?", checkpointer.saver),
    )
    first = await _events(runner, _run_input(thread, "start", "a" * 32))
    asked = next(e for e in first if isinstance(e, Clarification))

    restarted = _checkpointer(fresh_url)
    await restarted.open(wait=True)
    try:
        after = LangGraphRunner(
            "ask",
            "d",
            asking_graph("Which one?"),
            thread_graph=asking_graph("Which one?", restarted.saver),
        )
        assert await after.pending_interrupts(thread) == (asked,)
        await _events(after, _run_input(thread, None, "b" * 32, resume={asked.interrupt_id: "x"}))
        assert await after.pending_interrupts(thread) == ()
    finally:
        await restarted.aclose()
