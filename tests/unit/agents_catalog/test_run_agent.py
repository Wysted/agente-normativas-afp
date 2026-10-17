import asyncio
import logging
from collections.abc import AsyncGenerator

import pytest

from agintis.modules.agents_catalog.api import (
    AgentUnavailableError,
    RunCompleted,
    RunEvent,
    RunFailed,
    RunInput,
    RunnerEvent,
    RunStarted,
    Step,
    Token,
    TokenUsage,
)
from agintis.modules.agents_catalog.application import AgentNotFoundError, AgentRegistry, RunAgent
from agintis.platform.llm import ProviderErrorClassifier
from agintis.shared_kernel.errors import FailureInfo
from agintis.shared_kernel.ids import RunId, TenantId
from tests.fakes.agents import BlockingRunner, FakeRunner

TENANT = TenantId("t1")


def _run_agent(*runners: object) -> RunAgent:
    registry = AgentRegistry(tuple(runners))  # type: ignore[arg-type]
    return RunAgent(registry, ProviderErrorClassifier(), id_generator=lambda: "run-1")


async def _collect(run_agent: RunAgent, agent_id: str = "fake") -> list[RunEvent]:
    return [e async for e in run_agent.start(agent_id, TENANT, "hello")]


async def test_event_order_run_started_first_done_last() -> None:
    usage = TokenUsage(1, 2, 3)
    runner = FakeRunner([Step("n", "started"), Token("a"), RunCompleted("length", usage)])

    events = await _collect(_run_agent(runner))

    assert events == [
        RunStarted(RunId("run-1"), "fake"),
        Step("n", "started"),
        Token("a"),
        RunCompleted("length", usage),
    ]
    assert runner.inputs[0].message == "hello"
    assert runner.inputs[0].tenant_id == TENANT
    assert runner.inputs[0].run_id == "run-1"


async def test_default_completion_when_runner_gives_none() -> None:
    events = await _collect(_run_agent(FakeRunner([Token("a")])))

    assert events[-1] == RunCompleted("stop", None)
    assert sum(isinstance(e, RunCompleted) for e in events) == 1


async def test_events_after_runner_completion_are_not_forwarded() -> None:
    runner = FakeRunner([RunCompleted("stop"), Token("late")])

    events = await _collect(_run_agent(runner))

    assert events == [RunStarted(RunId("run-1"), "fake"), RunCompleted("stop")]


async def test_error_mid_stream_yields_single_error_and_stops(
    caplog: pytest.LogCaptureFixture,
) -> None:
    runner = FakeRunner([Token("a")], error=RuntimeError("secret-key-sk-123"))

    with caplog.at_level(logging.ERROR):
        events = await _collect(_run_agent(runner))

    assert events[:2] == [RunStarted(RunId("run-1"), "fake"), Token("a")]
    failures = [e for e in events if isinstance(e, RunFailed)]
    assert failures == [RunFailed("internal", "Internal error while running the agent.", False)]
    assert isinstance(events[-1], RunFailed)
    assert not any(isinstance(e, RunCompleted) for e in events)
    assert "secret-key-sk-123" not in failures[0].message
    assert any(r.exc_info for r in caplog.records)


async def test_cancelled_error_propagates() -> None:
    runner = FakeRunner([Token("a")], error=asyncio.CancelledError())

    with pytest.raises(asyncio.CancelledError):
        await _collect(_run_agent(runner))


def test_unknown_agent_raises_before_any_event() -> None:
    with pytest.raises(AgentNotFoundError):
        _run_agent(FakeRunner()).start("nope", TENANT, "hello")


def test_unavailable_runner_raises_before_any_event() -> None:
    with pytest.raises(AgentUnavailableError) as info:
        _run_agent(FakeRunner(unavailable=True)).start("fake", TENANT, "hello")
    assert info.value.code == "llm_not_configured"


async def test_runner_cannot_inject_run_started_or_run_failed() -> None:
    runner = FakeRunner(
        [Token("a"), RunStarted(RunId("evil"), "evil"), RunFailed("x", "y"), "junk", Token("b")]
    )

    events = await _collect(_run_agent(runner))

    assert events == [
        RunStarted(RunId("run-1"), "fake"),
        Token("a"),
        Token("b"),
        RunCompleted("stop", None),
    ]


class _FailsOnClose(FakeRunner):
    async def astream(self, run_input: RunInput) -> AsyncGenerator[RunnerEvent]:
        try:
            yield RunCompleted("stop")
            yield Token("never consumed")
        finally:
            raise RuntimeError("close failed")


async def test_failure_after_done_never_emits_an_error_event() -> None:
    events = await _collect(_run_agent(_FailsOnClose(agent_id="fake")))

    assert events == [RunStarted(RunId("run-1"), "fake"), RunCompleted("stop")]


class _BrokenClassifier:
    def classify(self, exc: BaseException) -> FailureInfo:
        raise RuntimeError("classifier bug")


async def test_classifier_failure_falls_back_to_internal() -> None:
    registry = AgentRegistry((FakeRunner(error=RuntimeError("x")),))
    run_agent = RunAgent(registry, _BrokenClassifier(), id_generator=lambda: "run-1")

    events = await _collect(run_agent)

    assert events[-1] == RunFailed("internal", "Internal error while running the agent.", False)


async def test_closing_the_stream_closes_the_runner() -> None:
    runner = BlockingRunner(Token("a"))
    stream = _run_agent(runner).start("blocking", TENANT, "hello")

    assert await anext(stream) == RunStarted(RunId("run-1"), "blocking")
    assert await anext(stream) == Token("a")
    await stream.aclose()

    assert runner.finalized


async def test_cancelling_the_consumer_cancels_the_runner() -> None:
    runner = BlockingRunner(Token("a"))
    stream = _run_agent(runner).start("blocking", TENANT, "hello")

    async def consume() -> None:
        async for _ in stream:
            pass

    task = asyncio.create_task(consume())
    await asyncio.wait_for(runner.started.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert runner.cancelled
    assert runner.finalized


def test_registry_rejects_duplicates_and_lists_in_order() -> None:
    a, b = FakeRunner(agent_id="a"), FakeRunner(agent_id="b")
    registry = AgentRegistry((a, b))

    assert [r.agent_id for r in registry.list()] == ["a", "b"]
    assert registry.get("b") is b
    with pytest.raises(ValueError, match="already registered"):
        registry.register(FakeRunner(agent_id="a"))


@pytest.mark.parametrize("bad_id", ["", "-lead", "has space", "x" * 65, "a/b"])
def test_registry_rejects_ids_a_client_could_not_send(bad_id: str) -> None:
    with pytest.raises(ValueError, match="must match"):
        AgentRegistry((FakeRunner(agent_id=bad_id),))


def test_registry_accepts_the_longest_valid_id() -> None:
    AgentRegistry((FakeRunner(agent_id="a" * 64),))
