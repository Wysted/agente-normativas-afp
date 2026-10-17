"""The pump: a run never outlives its deadline, whatever the consumer does."""

import asyncio
from collections.abc import AsyncGenerator
from typing import Annotated, Any, TypedDict

import pytest
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from agintis.modules.agents_catalog.api import (
    RUN_TIMEOUT_CODE,
    AgentRegistry,
    AgentRunner,
    RunCompleted,
    RunFailed,
    RunInput,
    RunnerEvent,
    RunStarted,
    ThreadOutcome,
    Token,
)
from agintis.modules.agents_catalog.application import RunAgent
from agintis.modules.agents_catalog.application import run_agent as run_agent_module
from agintis.modules.agents_catalog.application.run_agent import PUMP_QUEUE_SIZE
from agintis.platform.llm import ProviderErrorClassifier
from agintis.shared_kernel.ids import TenantId, ThreadId
from tests.fakes.agents import BlockingRunner, EndlessRunner, FakeRunner, FakeThreadAccess
from tests.fakes.graphs import Graph, threaded

TENANT = TenantId("t1")
THREAD = ThreadId("thread-1")
DEADLINE_SECONDS = 0.2
SLOW_NODE_SECONDS = 0.5
STALL_SECONDS = 1.0


def _run_agent(runner: AgentRunner, access: FakeThreadAccess, **kwargs: Any) -> RunAgent:
    return RunAgent(AgentRegistry((runner,)), ProviderErrorClassifier(), threads=access, **kwargs)


class _State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def _slow_write_graph(wrote: list[str]) -> Any:
    """`fast` finishes at once; `slow` would write to the state after SLOW_NODE_SECONDS."""

    def make(saver: Any) -> Graph:
        async def fast(state: _State) -> dict[str, Any]:
            return {}

        async def slow(state: _State) -> dict[str, Any]:
            await asyncio.sleep(SLOW_NODE_SECONDS)
            wrote.append("slow node wrote")  # a side effect after the deadline = a leak
            return {"messages": [AIMessage(content="late write")]}

        builder = StateGraph(_State)
        builder.add_node("fast", fast)
        builder.add_node("slow", slow)
        builder.add_edge(START, "fast")
        builder.add_edge(START, "slow")
        builder.add_edge("fast", END)
        builder.add_edge("slow", END)
        return builder.compile(checkpointer=saver)

    return make


async def test_a_stalled_consumer_cannot_keep_the_run_alive_past_the_deadline() -> None:
    wrote: list[str] = []
    toy = threaded(_slow_write_graph(wrote), "toy")
    access = FakeThreadAccess("toy")
    run_agent = _run_agent(toy.runner, access, run_deadline_seconds=DEADLINE_SECONDS)
    seen: list[Any] = []

    async with run_agent.start_thread(TENANT, THREAD, message="hi") as events:
        async for event in events:
            seen.append(event)
            if getattr(event, "status", "") == "finished":
                await asyncio.sleep(STALL_SECONDS)  # stalled reader: generator parked at a yield

    await asyncio.sleep(SLOW_NODE_SECONDS)  # long enough for a leaked node to write
    state = await toy.graph.aget_state({"configurable": {"thread_id": THREAD}})
    assert wrote == []  # the in-flight superstep was cancelled AT the deadline
    assert [m.content for m in state.values["messages"]] == ["hi"]
    failures = [e for e in seen if isinstance(e, RunFailed)]
    assert [(f.code, f.retryable) for f in failures] == [(RUN_TIMEOUT_CODE, True)]
    assert seen[-1] is failures[0]
    assert [outcome for _, outcome in access.released] == [ThreadOutcome.IDLE]


async def test_a_never_suspending_runner_is_stopped_by_the_deadline() -> None:
    runner = EndlessRunner()
    access = FakeThreadAccess("endless")
    run_agent = _run_agent(runner, access, run_deadline_seconds=0.05)
    seen: list[Any] = []

    async def consume() -> None:
        async with run_agent.start_thread(TENANT, THREAD, message="hi") as events:
            seen.extend([event async for event in events])

    await asyncio.wait_for(consume(), 5)

    assert isinstance(seen[-1], RunFailed)
    assert seen[-1].code == RUN_TIMEOUT_CODE
    assert sum(isinstance(e, RunFailed) for e in seen) == 1
    assert runner.finalized_event.is_set()
    assert len(access.released) == 1


async def test_the_runner_never_reads_far_ahead_of_a_slow_consumer() -> None:
    produced = 0

    class Counting(FakeRunner):
        async def astream(self, run_input: RunInput) -> AsyncGenerator[RunnerEvent]:
            nonlocal produced
            for _ in range(1000):
                produced += 1
                yield Token("x")
                await asyncio.sleep(0)

    run_agent = _run_agent(
        Counting(supports_threads=True), FakeThreadAccess(), run_deadline_seconds=5
    )

    async with run_agent.start_thread(TENANT, THREAD, message="hi") as events:
        async for event in events:
            if isinstance(event, Token):
                await asyncio.sleep(0.1)  # stalled after the first token
                break

    assert produced <= PUMP_QUEUE_SIZE + 3  # backpressure: bounded read-ahead


async def test_disconnect_with_a_parked_generator_stops_the_pump_and_closes_the_runner() -> None:
    runner = EndlessRunner()
    access = FakeThreadAccess("endless")
    run_agent = _run_agent(runner, access, run_deadline_seconds=30)
    before = len(asyncio.all_tasks())

    async def consume() -> None:
        async with run_agent.start_thread(TENANT, THREAD, message="hi") as events:
            async for event in events:
                if isinstance(event, Token):
                    await asyncio.sleep(30)  # parked at a yield; the client then goes away

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.1)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    await asyncio.wait_for(access.released_event.wait(), 2)
    await run_agent.aclose()

    assert runner.finalized_event.is_set()
    assert len(access.released) == 1
    assert len(asyncio.all_tasks()) == before  # no pump / release task leaked


async def test_a_runner_error_reaches_the_consumer_as_a_single_classified_error() -> None:
    runner = FakeRunner(
        [Token("a"), Token("b")], supports_threads=True, error=RuntimeError("secret-detail")
    )
    access = FakeThreadAccess()
    run_agent = _run_agent(runner, access, run_deadline_seconds=5)
    before = len(asyncio.all_tasks())

    async with run_agent.start_thread(TENANT, THREAD, message="hi") as events:
        seen: list[Any] = [event async for event in events]
    await run_agent.aclose()

    assert [type(e).__name__ for e in seen] == ["RunStarted", "Token", "Token", "RunFailed"]
    assert seen[-1].code == "internal"
    assert "secret-detail" not in seen[-1].message
    assert len(asyncio.all_tasks()) == before


async def test_the_pump_also_serves_stateless_runs() -> None:
    runner = FakeRunner([Token("a"), RunCompleted("length")], error=RuntimeError("late"))
    run_agent = RunAgent(AgentRegistry((runner,)), ProviderErrorClassifier())

    events = [e async for e in run_agent.start("fake", TENANT, "hi")]

    assert isinstance(events[0], RunStarted)
    assert events[1:] == [Token("a"), RunCompleted("length")]


async def test_a_deadline_run_that_finishes_in_time_is_unaffected() -> None:
    run_agent = _run_agent(
        FakeRunner([Token("a")], supports_threads=True),
        FakeThreadAccess(),
        run_deadline_seconds=5,
    )

    async with run_agent.start_thread(TENANT, THREAD, message="hi") as events:
        seen = [event async for event in events]

    assert seen[-1] == RunCompleted("stop")


async def test_shutdown_waits_for_runs_still_streaming(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = BlockingRunner(Token("a"))
    access = FakeThreadAccess("blocking")
    run_agent = _run_agent(runner, access)

    async def consume() -> None:
        async with run_agent.start_thread(TENANT, THREAD, message="hi") as events:
            async for _ in events:
                pass

    task = asyncio.create_task(consume())
    await asyncio.wait_for(runner.started.wait(), 2)
    closing = asyncio.create_task(run_agent.aclose())
    await asyncio.sleep(0.1)
    assert not closing.done()  # the run is active: no release task exists yet

    task.cancel()  # the server cancels the request at shutdown
    await asyncio.gather(task, return_exceptions=True)
    await asyncio.wait_for(closing, 2)

    assert len(access.released) == 1


async def test_shutdown_gives_up_after_its_timeout(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(run_agent_module, "RELEASE_DRAIN_TIMEOUT_SECONDS", 0.1)
    runner = BlockingRunner(Token("a"))
    access = FakeThreadAccess("blocking")
    run_agent = _run_agent(runner, access)

    async def consume() -> None:
        async with run_agent.start_thread(TENANT, THREAD, message="hi") as events:
            async for _ in events:
                pass

    task = asyncio.create_task(consume())
    await asyncio.wait_for(runner.started.wait(), 2)

    await asyncio.wait_for(run_agent.aclose(), 2)  # returns, run still active

    assert "did not finish their teardown" in caplog.text
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    await asyncio.wait_for(access.released_event.wait(), 2)
