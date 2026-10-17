"""RunAgent thread mode: the lease is taken before the stream, the thread status follows the
CHECKPOINT, and the lease is ALWAYS given back (after the runner is closed)."""

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

import pytest

from agintis.modules.agents_catalog.api import (
    RUN_TIMEOUT_CODE,
    AgentNotFoundError,
    AgentRegistry,
    AgentRunner,
    AgentUnavailableError,
    AmbiguousResumeError,
    Clarification,
    NoPendingQuestionError,
    ResumeMismatchError,
    RunCompleted,
    RunFailed,
    RunInput,
    RunnerEvent,
    RunStarted,
    ThreadAwaitingAnswerError,
    ThreadOutcome,
    ThreadRunMode,
    ThreadsNotSupportedError,
    ThreadsUnavailableError,
    Token,
)
from agintis.modules.agents_catalog.application import RunAgent
from agintis.modules.agents_catalog.application.run_agent import thread_outcome
from agintis.modules.conversations.api import ThreadBusyError, ThreadNotFoundError
from agintis.platform.llm import ProviderErrorClassifier
from agintis.shared_kernel.ids import RunId, TenantId, ThreadId
from tests.fakes.agents import BlockingRunner, FakeRunner, FakeThreadAccess

TENANT = TenantId("t1")
THREAD = ThreadId("thread-1")
INTERRUPT = [Clarification("Which one?", (), "id-1"), RunCompleted("interrupted")]
ASKED = Clarification("Which one?", (), "id-1")
ASKED_B = Clarification("And which other?", (), "id-2")


def _run_agent(runner: Any, access: FakeThreadAccess | None, **kwargs: Any) -> RunAgent:
    return RunAgent(
        AgentRegistry((runner,)),
        ProviderErrorClassifier(),
        id_generator=lambda: "run-1",
        threads=access,
        **kwargs,
    )


def _runner(events: list[Any] | None = None, **kwargs: Any) -> FakeRunner:
    return FakeRunner(events or [Token("a")], supports_threads=True, **kwargs)


async def _drain(run_agent: RunAgent, **kwargs: Any) -> list[Any]:
    async with run_agent.start_thread(TENANT, THREAD, **kwargs) as events:
        return [event async for event in events]


def _outcomes(access: FakeThreadAccess) -> list[ThreadOutcome]:
    return [outcome for _, outcome in access.released]


async def test_message_run_acquires_before_the_stream_and_releases_idle_on_done() -> None:
    access = FakeThreadAccess()
    runner = _runner()
    run_agent = _run_agent(runner, access)

    async with run_agent.start_thread(TENANT, THREAD, message="hi") as events:
        assert access.acquired == [(TENANT, THREAD, ThreadRunMode.MESSAGE)]
        assert access.released == []  # not before the stream ran
        collected = [e async for e in events]

    assert collected[0] == RunStarted(RunId("run-1"), "fake")
    assert collected[-1] == RunCompleted("stop")
    assert runner.inputs == [RunInput(TENANT, "hi", RunId("run-1"), THREAD)]
    assert _outcomes(access) == [ThreadOutcome.IDLE]


async def test_interrupted_run_releases_the_thread_as_interrupted() -> None:
    access = FakeThreadAccess()

    events = await _drain(_run_agent(_runner(INTERRUPT), access), message="hi")

    assert events[-1] == RunCompleted("interrupted")
    assert _outcomes(access) == [ThreadOutcome.INTERRUPTED]


async def test_resume_is_always_an_explicit_id_map() -> None:
    access = FakeThreadAccess()
    runner = _runner(pending=[ASKED])

    await _drain(_run_agent(runner, access), resume={"id-looking": "x" * 32})

    assert access.acquired[0][2] is ThreadRunMode.RESUME
    assert runner.inputs[0].resume == {"id-1": {"id-looking": "x" * 32}}
    assert runner.inputs[0].message is None
    assert _outcomes(access) == [ThreadOutcome.IDLE]


async def test_resumes_answer_several_pending_interrupts_by_id() -> None:
    access = FakeThreadAccess()
    runner = _runner(pending=[ASKED, ASKED_B])

    await _drain(_run_agent(runner, access), resumes={"id-2": "b", "id-1": "a"})

    assert runner.inputs[0].resume == {"id-1": "a", "id-2": "b"}


@pytest.mark.parametrize(
    "resumes", [{"id-1": "a"}, {"id-1": "a", "id-2": "b", "id-3": "c"}, {"id-1": "a", "x": "b"}]
)
async def test_resumes_must_name_exactly_the_pending_interrupts(resumes: dict[str, Any]) -> None:
    access = FakeThreadAccess()
    runner = _runner(pending=[ASKED, ASKED_B])

    with pytest.raises(ResumeMismatchError) as info:
        await _drain(_run_agent(runner, access), resumes=resumes)

    assert info.value.kind.value == "conflict"
    assert runner.inputs == []
    assert _outcomes(access) == [ThreadOutcome.INTERRUPTED]  # still waiting


async def test_a_bare_resume_with_two_pending_is_a_conflict_with_a_safe_message() -> None:
    access = FakeThreadAccess()
    runner = _runner(pending=[ASKED, ASKED_B])

    with pytest.raises(AmbiguousResumeError) as info:
        await _drain(_run_agent(runner, access), resume="a")

    assert "resumes" in info.value.message
    assert "id-" not in info.value.message
    assert runner.inputs == []
    assert _outcomes(access) == [ThreadOutcome.INTERRUPTED]


async def test_resume_without_a_pending_interrupt_is_refused_and_the_thread_set_idle() -> None:
    access = FakeThreadAccess()
    runner = _runner()  # the checkpoint holds no question (stale `interrupted` status)

    with pytest.raises(NoPendingQuestionError):
        await _drain(_run_agent(runner, access), resume="a")

    assert runner.inputs == []  # a bare Command(resume) would be consumed by a later interrupt()
    assert _outcomes(access) == [ThreadOutcome.IDLE]


async def test_message_into_a_pending_interrupt_is_refused_and_the_thread_set_interrupted() -> None:
    access = FakeThreadAccess()
    runner = _runner(pending=[ASKED])  # stale `idle` status over a pending question

    with pytest.raises(ThreadAwaitingAnswerError):
        await _drain(_run_agent(runner, access), message="hi")

    assert runner.inputs == []  # new input would silently discard the interrupt
    assert _outcomes(access) == [ThreadOutcome.INTERRUPTED]


async def test_failed_message_run_is_reconciled_with_the_checkpoint() -> None:
    access = FakeThreadAccess()

    events = await _drain(_run_agent(_runner(error=RuntimeError("boom")), access), message="hi")

    assert events[-1].code == "internal"  # RunFailed
    assert _outcomes(access) == [ThreadOutcome.IDLE]


async def test_interrupt_and_error_in_the_same_superstep_leave_the_thread_interrupted() -> None:
    access = FakeThreadAccess()

    class InterruptingThenFailing(FakeRunner):
        async def astream(self, run_input: RunInput) -> AsyncGenerator[RunnerEvent]:
            self.pending = (ASKED,)  # the checkpoint got the interrupt before the error
            async for event in super().astream(run_input):
                yield event

    runner = InterruptingThenFailing(
        [Token("a")], supports_threads=True, error=RuntimeError("boom")
    )

    events = await _drain(_run_agent(runner, access), message="hi")

    assert isinstance(events[-1], RunFailed)
    assert _outcomes(access) == [ThreadOutcome.INTERRUPTED]


@pytest.mark.parametrize(
    ("still_pending", "expected"),
    [(True, ThreadOutcome.INTERRUPTED), (False, ThreadOutcome.IDLE)],
)
async def test_failed_resume_run_follows_whether_the_answer_was_consumed(
    still_pending: bool, expected: ThreadOutcome
) -> None:
    access = FakeThreadAccess()

    class Consuming(FakeRunner):
        async def astream(self, run_input: RunInput) -> AsyncGenerator[RunnerEvent]:
            self.pending = (ASKED,) if still_pending else ()  # what the checkpoint holds after
            async for event in super().astream(run_input):
                yield event

    runner = Consuming(
        [Token("a")], supports_threads=True, error=RuntimeError("boom"), pending=[ASKED]
    )

    await _drain(_run_agent(runner, access), resume="b")

    assert _outcomes(access) == [expected]


async def test_an_unreadable_checkpoint_falls_back_to_the_mode_table() -> None:
    access = FakeThreadAccess()
    runner = _runner(error=RuntimeError("boom"), pending_error=OSError("db down"))

    with pytest.raises(OSError, match="db down"):  # no pending read possible before the run
        await _drain(_run_agent(runner, access), message="hi")

    assert _outcomes(access) == [ThreadOutcome.IDLE]


async def test_release_runs_when_the_stream_is_never_iterated() -> None:
    access = FakeThreadAccess()

    async with _run_agent(_runner(), access).start_thread(TENANT, THREAD, message="hi"):
        pass  # client gone before the first event

    assert _outcomes(access) == [ThreadOutcome.IDLE]


async def test_release_runs_and_closes_the_runner_when_the_consumer_is_cancelled() -> None:
    access = FakeThreadAccess("blocking")
    runner = BlockingRunner(Token("a"))
    run_agent = _run_agent(runner, access)

    async def consume() -> None:
        async with run_agent.start_thread(TENANT, THREAD, message="hi") as events:
            async for _ in events:
                pass

    task = asyncio.create_task(consume())
    await asyncio.wait_for(runner.started.wait(), 2)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    await asyncio.wait_for(access.released_event.wait(), 2)

    assert runner.cancelled
    assert _outcomes(access) == [ThreadOutcome.IDLE]


async def test_the_lease_is_released_strictly_after_the_runner_is_closed() -> None:
    log: list[str] = []

    class Logging(FakeThreadAccess):
        async def release(self, ticket: Any, outcome: ThreadOutcome) -> None:
            log.append("released")
            await super().release(ticket, outcome)

    class Closing(BlockingRunner):
        @property
        def supports_threads(self) -> bool:
            return True

        async def astream(self, run_input: RunInput) -> AsyncGenerator[RunnerEvent]:
            try:
                async for event in super().astream(run_input):
                    yield event
            finally:
                await asyncio.sleep(0.05)  # slow cleanup: nodes still winding down
                log.append("runner closed")

    access = Logging("blocking")
    runner = Closing(Token("a"))
    run_agent = _run_agent(runner, access)

    async def consume() -> None:
        async with run_agent.start_thread(TENANT, THREAD, message="hi") as events:
            async for _ in events:
                pass

    task = asyncio.create_task(consume())
    await asyncio.wait_for(runner.started.wait(), 2)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    await asyncio.wait_for(access.released_event.wait(), 2)

    assert log == ["runner closed", "released"]


async def test_release_survives_the_cancellation_of_its_own_caller() -> None:
    gate = asyncio.Event()

    class SlowRelease(FakeThreadAccess):
        async def release(self, ticket: Any, outcome: ThreadOutcome) -> None:
            await gate.wait()
            await super().release(ticket, outcome)

    access = SlowRelease()
    run_agent = _run_agent(_runner(), access)

    async def consume() -> None:
        async with run_agent.start_thread(TENANT, THREAD, message="hi"):
            pass

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.05)  # inside the (blocked) release
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    gate.set()
    await asyncio.wait_for(access.released_event.wait(), 2)

    assert _outcomes(access) == [ThreadOutcome.IDLE]


async def test_aclose_waits_for_in_flight_releases() -> None:
    gate = asyncio.Event()

    class SlowRelease(FakeThreadAccess):
        async def release(self, ticket: Any, outcome: ThreadOutcome) -> None:
            await gate.wait()
            await super().release(ticket, outcome)

    access = SlowRelease()
    run_agent = _run_agent(_runner(), access)

    async def consume() -> None:
        async with run_agent.start_thread(TENANT, THREAD, message="hi"):
            pass

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.05)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    closing = asyncio.create_task(run_agent.aclose())
    await asyncio.sleep(0.05)
    assert not closing.done()  # still draining

    gate.set()
    await asyncio.wait_for(closing, 2)

    assert _outcomes(access) == [ThreadOutcome.IDLE]


async def test_a_failing_release_is_logged_and_does_not_mask_the_run(
    caplog: pytest.LogCaptureFixture,
) -> None:
    access = FakeThreadAccess(release_error=RuntimeError("db down"))

    events = await _drain(_run_agent(_runner(), access), message="hi")

    assert events[-1] == RunCompleted("stop")
    assert "Releasing a thread lease failed" in caplog.text


async def test_a_run_past_its_deadline_is_stopped_with_a_retryable_error_and_released() -> None:
    access = FakeThreadAccess("blocking")
    runner = BlockingRunner(Token("a"))
    run_agent = _run_agent(runner, access, run_deadline_seconds=0.1)

    async with run_agent.start_thread(TENANT, THREAD, message="hi") as events:
        collected = [e async for e in events]

    assert collected[:2] == [RunStarted(RunId("run-1"), "blocking"), Token("a")]
    failure = collected[-1]
    assert isinstance(failure, RunFailed)
    assert (failure.code, failure.retryable) == (RUN_TIMEOUT_CODE, True)
    assert len(collected) == 3
    assert runner.cancelled  # the run was really stopped
    assert _outcomes(access) == [ThreadOutcome.IDLE]


async def test_a_run_that_finishes_before_the_deadline_is_not_affected() -> None:
    access = FakeThreadAccess()

    events = await _drain(_run_agent(_runner(), access, run_deadline_seconds=5), message="hi")

    assert events[-1] == RunCompleted("stop")


async def test_the_deadline_applies_to_thread_runs_only() -> None:
    runner = FakeRunner([Token("a")])
    run_agent = _run_agent(runner, None, run_deadline_seconds=0.0001)

    events = [e async for e in run_agent.start("fake", TENANT, "hi")]

    assert events[-1] == RunCompleted("stop")


@pytest.mark.parametrize("error", [ThreadNotFoundError(), ThreadBusyError()])
async def test_acquire_errors_surface_before_any_stream_and_nothing_is_released(
    error: Exception,
) -> None:
    access = FakeThreadAccess(acquire_error=error)
    runner = _runner()

    with pytest.raises(type(error)):
        await _drain(_run_agent(runner, access), message="hi")

    assert runner.inputs == []
    assert access.released == []


async def test_unavailable_agent_gives_the_lease_back_with_the_starting_state() -> None:
    access = FakeThreadAccess()
    runner = _runner(unavailable=True, pending=[ASKED])

    with pytest.raises(AgentUnavailableError):
        await _drain(_run_agent(runner, access), resume="b")

    assert _outcomes(access) == [ThreadOutcome.INTERRUPTED]


async def test_agent_without_thread_support_is_rejected_and_released_idle() -> None:
    access = FakeThreadAccess()
    runner: AgentRunner = FakeRunner([Token("a")])  # supports_threads False

    with pytest.raises(ThreadsNotSupportedError):
        await _drain(_run_agent(runner, access), message="hi")

    assert _outcomes(access) == [ThreadOutcome.IDLE]


async def test_thread_whose_agent_is_gone_is_a_404_and_released() -> None:
    access = FakeThreadAccess(agent_id="deleted-agent")

    with pytest.raises(AgentNotFoundError):
        await _drain(_run_agent(_runner(), access), message="hi")

    assert _outcomes(access) == [ThreadOutcome.IDLE]


async def test_without_a_thread_access_thread_runs_are_unavailable() -> None:
    with pytest.raises(ThreadsUnavailableError):
        await _drain(_run_agent(_runner(), None), message="hi")


@pytest.mark.parametrize(
    ("mode", "finish", "expected"),
    [
        (ThreadRunMode.MESSAGE, None, ThreadOutcome.IDLE),
        (ThreadRunMode.RESUME, None, ThreadOutcome.INTERRUPTED),
        (ThreadRunMode.MESSAGE, "stop", ThreadOutcome.IDLE),
        (ThreadRunMode.RESUME, "length", ThreadOutcome.IDLE),
        (ThreadRunMode.MESSAGE, "interrupted", ThreadOutcome.INTERRUPTED),
        (ThreadRunMode.RESUME, "interrupted", ThreadOutcome.INTERRUPTED),
    ],
)
def test_thread_outcome_table_is_the_last_resort(
    mode: ThreadRunMode, finish: str | None, expected: ThreadOutcome
) -> None:
    assert thread_outcome(mode, finish) is expected


def test_run_input_needs_exactly_one_of_message_and_resume() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        RunInput(TENANT, None, RunId("r"), THREAD)
    with pytest.raises(ValueError, match="exactly one"):
        RunInput(TENANT, "hi", RunId("r"), THREAD, resume={"i": "x"})
    with pytest.raises(ValueError, match="thread"):
        RunInput(TENANT, None, RunId("r"), None, resume={"i": "x"})
    with pytest.raises(ValueError, match="at least one"):
        RunInput(TENANT, None, RunId("r"), THREAD, resume={})
