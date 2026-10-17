import asyncio
import logging
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Mapping, Sequence
from contextlib import aclosing, asynccontextmanager, nullcontext, suppress
from typing import Any

from agintis.modules.agents_catalog.application.ports import (
    AgentRunner,
    AmbiguousResumeError,
    NoPendingQuestionError,
    ResumeMismatchError,
    RunInput,
    ThreadAccess,
    ThreadAwaitingAnswerError,
    ThreadOutcome,
    ThreadRunMode,
    ThreadRunTicket,
    ThreadsNotSupportedError,
    ThreadsUnavailableError,
)
from agintis.modules.agents_catalog.application.registry import AgentRegistry
from agintis.modules.agents_catalog.domain import (
    DEFAULT_FINISH_REASON,
    INTERRUPTED_FINISH_REASON,
    RUN_TIMEOUT_CODE,
    Clarification,
    RunCompleted,
    RunEvent,
    RunFailed,
    RunStarted,
    Scope,
    Sources,
    Step,
    Token,
)
from agintis.shared_kernel.errors import FailureInfo
from agintis.shared_kernel.ids import RunId, TenantId, ThreadId, new_id
from agintis.shared_kernel.ports import ErrorClassifier

logger = logging.getLogger(__name__)

_CONTENT_EVENTS = (Token, Step, Sources, Scope, Clarification)
_TIMEOUT_MESSAGE = "The run took too long and was stopped."
# Upper bound of the wait for in-flight lease releases at shutdown.
RELEASE_DRAIN_TIMEOUT_SECONDS = 5.0
# Events the pump may read ahead of a slow consumer: small on purpose, so a stalled reader stalls
# the run (backpressure) instead of buffering it, and the deadline still ends it.
PUMP_QUEUE_SIZE = 8
# Thread left in this status when a request is rejected after the lease was taken, because the
# checkpoint disagrees with it.
_REJECTION_OUTCOMES: dict[type[BaseException], ThreadOutcome] = {
    ThreadAwaitingAnswerError: ThreadOutcome.INTERRUPTED,
    AmbiguousResumeError: ThreadOutcome.INTERRUPTED,
    ResumeMismatchError: ThreadOutcome.INTERRUPTED,
    NoPendingQuestionError: ThreadOutcome.IDLE,
}
_FALLBACK_FAILURE = FailureInfo("internal", "Internal error while running the agent.", False)


class _RunOutcome:
    """What the stream observed, read by the thread release once the run is over."""

    def __init__(self) -> None:
        self.finish_reason: str | None = None


def thread_outcome(mode: ThreadRunMode, finish_reason: str | None) -> ThreadOutcome:
    """Where the thread goes when its run ends, from what the stream observed.

    - A run that reached `done` decides: finish reason "interrupted" -> interrupted (it asked a
      question), anything else -> idle.
    - A run that did NOT reach `done` is reconciled against the checkpoint by `RunAgent` (any
      pending interrupt -> interrupted, none -> idle). This table is only the LAST RESORT when
      the checkpoint cannot be read: a message run -> idle, a resume run -> interrupted.
    """
    if finish_reason is None:
        return ThreadOutcome.IDLE if mode is ThreadRunMode.MESSAGE else ThreadOutcome.INTERRUPTED
    if finish_reason == INTERRUPTED_FINISH_REASON:
        return ThreadOutcome.INTERRUPTED
    return ThreadOutcome.IDLE


def outcome_from_pending(pending: Sequence[Clarification]) -> ThreadOutcome:
    return ThreadOutcome.INTERRUPTED if pending else ThreadOutcome.IDLE


def resume_answers(
    pending: Sequence[Clarification], resume: object | None, resumes: Mapping[str, object] | None
) -> dict[str, object]:
    """The explicit `{interrupt_id: value}` map for a resume, validated against the checkpoint.

    Always an id map (never a bare value): LangGraph rejects a bare value when several interrupts
    are pending and mistakes a dict of 32-hex keys for an id map.
    """
    if not pending:
        raise NoPendingQuestionError
    ids = [question.interrupt_id for question in pending]
    if resumes is not None:
        if set(resumes) != set(ids):
            raise ResumeMismatchError
        return dict(resumes)
    if len(ids) > 1:
        raise AmbiguousResumeError
    return {ids[0]: resume}


class _PumpState:
    def __init__(self) -> None:
        self.timed_out = False


_PUMP_ENDED = object()


async def _next_item(queue: asyncio.Queue[object], pump: asyncio.Future[None]) -> object:
    """The next queued event, or `_PUMP_ENDED` once the pump finished and the queue is drained."""
    if not queue.empty():
        return queue.get_nowait()
    if pump.done():
        return _PUMP_ENDED
    getter: asyncio.Future[object] = asyncio.ensure_future(queue.get())
    waiting: set[asyncio.Future[Any]] = {getter, pump}
    try:
        await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)
    except BaseException:
        getter.cancel()
        raise
    if getter.done():
        return getter.result()
    getter.cancel()
    return queue.get_nowait() if not queue.empty() else _PUMP_ENDED


class RunAgent:
    """Runs an agent and yields its lifecycle: run_started, content..., done | error.

    `start` does every check that can fail for a known reason (unknown agent, runner not
    available) synchronously, BEFORE returning the stream, so a transport can still answer
    with a proper error status. Once the stream is returned, failures become a single
    `RunFailed` event followed by the end of the stream (never after `RunCompleted`); the
    stream never raises, except for `asyncio.CancelledError`, which propagates (a client
    disconnect cancels the run).

    The runner contract is enforced here: only content events and `RunCompleted` pass through;
    anything else a runner yields (RunStarted, RunFailed, ...) is dropped with a warning.
    """

    def __init__(
        self,
        registry: AgentRegistry,
        classifier: ErrorClassifier,
        id_generator: Callable[[], str] = new_id,
        threads: ThreadAccess | None = None,
        run_deadline_seconds: float | None = None,
    ) -> None:
        self._registry = registry
        self._classifier = classifier
        self._id_generator = id_generator
        self._threads = threads
        # Thread runs only: a run must end before its lease expires (no heartbeat).
        self._run_deadline_seconds = run_deadline_seconds
        # Strong references to in-flight lease releases (a task only held weakly may vanish).
        self._releases: set[asyncio.Task[None]] = set()
        # One future per thread run, from acquire until its teardown finished: what shutdown
        # waits for (a run still streaming has no release task yet).
        self._active: set[asyncio.Future[None]] = set()

    def start(self, agent_id: str, tenant_id: TenantId, message: str) -> AsyncGenerator[RunEvent]:
        runner = self._registry.get(agent_id)
        runner.ensure_available()
        run_id = RunId(self._id_generator())
        return self._stream(runner, RunInput(tenant_id=tenant_id, message=message, run_id=run_id))

    @asynccontextmanager
    async def start_thread(
        self,
        tenant_id: TenantId,
        thread_id: ThreadId,
        *,
        message: str | None = None,
        resume: object | None = None,
        resumes: Mapping[str, object] | None = None,
    ) -> AsyncIterator[AsyncGenerator[RunEvent]]:
        """Acquire the thread, hand over the not-yet-started stream, release the thread after.

        Pass exactly one of `message`, `resume` (one answer; only when exactly one question is
        pending) or `resumes` (`{interrupt_id: answer}`, must name the pending ones exactly).

        Entering raises BEFORE any stream exists (thread unknown for the tenant, busy, in the
        wrong state, agent gone / unavailable / without thread support, checkpoint disagreeing
        with the request), with the lease already given back. Leaving ALWAYS releases the lease
        (done, error, deadline, client disconnect, run never iterated): the caller must keep the
        block open until the response has ended and let cancellation travel through it.

        The thread status is RECONCILED with the checkpoint, the source of truth: a message is
        never fed into a pending interrupt (409, status set to interrupted), a resume never goes
        out without one (409, status set to idle), and a run that did not reach `done` settles
        idle or interrupted according to the pending interrupts (see `thread_outcome`).

        The run deadline is anchored here, BEFORE the acquire and the checkpoint read, so the
        whole thing ends before the lease does. `RunFailed(run_timeout, retryable=True)` means
        "safe to retry only after checking the thread": the supersteps committed before the
        deadline persist in the checkpoint (retryable is not idempotent for thread runs).
        """
        if self._threads is None:
            raise ThreadsUnavailableError
        deadline_at = (
            None
            if self._run_deadline_seconds is None
            else asyncio.get_running_loop().time() + self._run_deadline_seconds
        )
        mode = ThreadRunMode.MESSAGE if message is not None else ThreadRunMode.RESUME
        ticket = await self._threads.acquire(tenant_id, thread_id, mode)
        active: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._active.add(active)
        runner: AgentRunner | None = None
        try:
            runner = self._registry.get(ticket.agent_id)
            runner.ensure_available()
            if not runner.supports_threads:
                raise ThreadsNotSupportedError(runner.agent_id)
            pending = await runner.pending_interrupts(thread_id)
            answers = None
            if mode is ThreadRunMode.MESSAGE:
                if pending:
                    raise ThreadAwaitingAnswerError
            else:
                answers = resume_answers(pending, resume, resumes)
            run_input = RunInput(
                tenant_id=tenant_id,
                message=message,
                run_id=RunId(self._id_generator()),
                thread_id=thread_id,
                resume=answers,
            )
        except BaseException as exc:
            forced = _REJECTION_OUTCOMES.get(type(exc))
            await self._finish(ticket, runner, mode, active, forced=forced)
            raise
        outcome = _RunOutcome()
        events = self._stream(runner, run_input, outcome, deadline_at)
        try:
            yield events
        finally:
            await self._finish(ticket, runner, mode, active, events=events, outcome=outcome)

    async def aclose(self) -> None:
        """Wait (bounded) for every thread run to finish its teardown, releases included: call
        before the database closes. Counts runs from their acquire, not only the releases already
        in flight (a run still streaming has none yet)."""
        active = set(self._active)
        if not active:
            return
        _, still_running = await asyncio.wait(active, timeout=RELEASE_DRAIN_TIMEOUT_SECONDS)
        if still_running:
            logger.warning(
                "%d thread run(s) did not finish their teardown at shutdown", len(still_running)
            )

    async def _finish(
        self,
        ticket: ThreadRunTicket,
        runner: AgentRunner | None,
        mode: ThreadRunMode,
        active: asyncio.Future[None],
        *,
        events: AsyncGenerator[RunEvent] | None = None,
        outcome: _RunOutcome | None = None,
        forced: ThreadOutcome | None = None,
    ) -> None:
        """Close the stream, THEN settle and release, all in one task that survives the
        cancellation of the caller.

        Under a cancelled scope (client gone) every `await` re-raises CancelledError, so the work
        runs in its own task behind `shield`; a failure is logged instead of masking the run.
        The release happens strictly after the runner is closed (no node still writing).
        """
        task = asyncio.ensure_future(
            self._close_and_release(ticket, runner, mode, active, events, outcome, forced)
        )
        self._releases.add(task)
        task.add_done_callback(self._release_done)
        with suppress(Exception):  # a failure is logged by the done callback
            await asyncio.shield(task)

    async def _close_and_release(
        self,
        ticket: ThreadRunTicket,
        runner: AgentRunner | None,
        mode: ThreadRunMode,
        active: asyncio.Future[None],
        events: AsyncGenerator[RunEvent] | None,
        outcome: _RunOutcome | None,
        forced: ThreadOutcome | None,
    ) -> None:
        assert self._threads is not None
        try:
            try:
                if events is not None:
                    await events.aclose()  # also stops the pump and closes the runner
            finally:
                decided = forced or await self._settle(ticket, runner, mode, outcome)
                await self._threads.release(ticket, decided)
        finally:
            self._active.discard(active)
            if not active.done():
                active.set_result(None)

    async def _settle(
        self,
        ticket: ThreadRunTicket,
        runner: AgentRunner | None,
        mode: ThreadRunMode,
        outcome: _RunOutcome | None,
    ) -> ThreadOutcome:
        finish_reason = None if outcome is None else outcome.finish_reason
        if finish_reason is not None:
            return thread_outcome(mode, finish_reason)  # the run said how it ended
        if runner is not None and runner.supports_threads:
            try:
                return outcome_from_pending(await runner.pending_interrupts(ticket.thread_id))
            except Exception:
                logger.exception("Could not read the checkpoint of thread %s", ticket.thread_id)
        return thread_outcome(mode, None)

    def _release_done(self, task: asyncio.Task[None]) -> None:
        self._releases.discard(task)
        if not task.cancelled() and (error := task.exception()) is not None:
            logger.error("Releasing a thread lease failed", exc_info=error)

    async def _stream(
        self,
        runner: AgentRunner,
        run_input: RunInput,
        outcome: _RunOutcome | None = None,
        deadline_at: float | None = None,
    ) -> AsyncGenerator[RunEvent]:
        yield RunStarted(run_id=run_input.run_id, agent_id=runner.agent_id)
        completed = False
        # A pump task reads the runner into a small queue and the WHOLE pump (pulls and puts)
        # runs under the deadline: a consumer that stops reading cannot keep the run, and its
        # in-flight supersteps, alive past the deadline. Closing this generator (finally below)
        # cancels the pump, which closes the runner (aclosing) BEFORE the lease is released.
        queue: asyncio.Queue[object] = asyncio.Queue(PUMP_QUEUE_SIZE)
        pump_state = _PumpState()
        pump = asyncio.ensure_future(self._pump(runner, run_input, queue, deadline_at, pump_state))
        try:
            while (item := await _next_item(queue, pump)) is not _PUMP_ENDED:
                # Typed as object: the runner contract is enforced here, not trusted.
                event = item
                if isinstance(event, RunCompleted):
                    completed = True
                    if outcome is not None:
                        outcome.finish_reason = event.finish_reason
                    yield event
                    break
                if isinstance(event, _CONTENT_EVENTS):
                    yield event
                else:
                    logger.warning(
                        "Agent %r yielded a non-content event %s; dropped",
                        runner.agent_id,
                        type(event).__name__,
                    )
            else:
                await pump  # the runner ended (or failed: raised here) on its own
        except Exception as exc:
            logger.exception("Run %s of agent %r failed", run_input.run_id, runner.agent_id)
            if completed:
                return  # e.g. the runner failed while closing: `done` was already sent
            failure = self._classify(exc)
            yield RunFailed(code=failure.code, message=failure.message, retryable=failure.retryable)
            return
        finally:
            pump.cancel()
            await asyncio.gather(pump, return_exceptions=True)
        if pump_state.timed_out:
            logger.warning("Run %s of agent %r hit its deadline", run_input.run_id, runner.agent_id)
            yield RunFailed(code=RUN_TIMEOUT_CODE, message=_TIMEOUT_MESSAGE, retryable=True)
            return
        if not completed:
            if outcome is not None:
                outcome.finish_reason = DEFAULT_FINISH_REASON
            yield RunCompleted(finish_reason=DEFAULT_FINISH_REASON)

    @staticmethod
    async def _pump(
        runner: AgentRunner,
        run_input: RunInput,
        queue: asyncio.Queue[object],
        deadline_at: float | None,
        state: _PumpState,
    ) -> None:
        """Feed the runner's events into `queue`; stop at the deadline, closing the runner."""
        timer = None if deadline_at is None else asyncio.timeout_at(deadline_at)
        try:
            async with timer or nullcontext(), aclosing(runner.astream(run_input)) as events:
                async for item in events:
                    if deadline_at is not None and asyncio.get_running_loop().time() >= deadline_at:
                        state.timed_out = True  # a runner that never suspends cannot be cancelled
                        return
                    await queue.put(item)
        except TimeoutError:
            if timer is not None and timer.expired():
                state.timed_out = True
                return
            raise

    def _classify(self, exc: Exception) -> FailureInfo:
        try:
            return self._classifier.classify(exc)
        except Exception:
            logger.exception("Error classifier failed")
            return _FALLBACK_FAILURE
