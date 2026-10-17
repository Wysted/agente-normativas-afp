from datetime import UTC, datetime, timedelta

import pytest

from agintis.modules.conversations.domain import (
    TITLE_MAX_LENGTH,
    Thread,
    ThreadBusyError,
    ThreadStatus,
    normalize_title,
)
from agintis.shared_kernel.errors import ErrorKind, InvalidRequestError
from agintis.shared_kernel.ids import TenantId, ThreadId

NOW = datetime(2026, 1, 1, 12, tzinfo=UTC)
TTL = timedelta(minutes=5)


def _thread() -> Thread:
    return Thread.create(ThreadId("t1"), TenantId("acme"), "chat", "  Hello  ", NOW)


def test_create_starts_idle_and_normalizes_the_title() -> None:
    thread = _thread()

    assert thread.status is ThreadStatus.IDLE
    assert thread.lease_until is None
    assert thread.title == "Hello"
    assert thread.created_at == thread.updated_at == NOW


@pytest.mark.parametrize(("raw", "expected"), [(None, None), ("   ", None), (" a ", "a")])
def test_normalize_title(raw: str | None, expected: str | None) -> None:
    assert normalize_title(raw) == expected


def test_over_long_title_is_rejected() -> None:
    with pytest.raises(InvalidRequestError):
        normalize_title("x" * (TITLE_MAX_LENGTH + 1))


def test_a_run_makes_the_thread_busy_with_a_lease() -> None:
    started = _thread().start_run(NOW, TTL)

    assert started.status is ThreadStatus.BUSY
    assert started.lease_until == NOW + TTL
    assert started.is_leased(NOW)
    assert not started.is_leased(NOW + TTL + timedelta(seconds=1))


def test_a_valid_lease_blocks_with_conflict() -> None:
    busy = _thread().start_run(NOW, TTL)

    with pytest.raises(ThreadBusyError) as info:
        busy.check_can_start(NOW)

    assert info.value.kind is ErrorKind.CONFLICT


def test_the_lease_is_pure_mutual_exclusion() -> None:
    """Idle, interrupted and expired-busy are all acquirable: the checkpoint decides the rest."""
    idle = _thread()
    interrupted = idle.start_run(NOW, TTL).end_run(ThreadStatus.INTERRUPTED, NOW)
    expired = idle.start_run(NOW, TTL)
    later = NOW + TTL + timedelta(seconds=1)

    idle.check_can_start(NOW)
    interrupted.check_can_start(NOW)
    expired.check_can_start(later)
    assert interrupted.lease_until is None
    assert interrupted.start_run(NOW, TTL).status is ThreadStatus.BUSY


def test_end_run_clears_the_lease_and_cannot_end_busy() -> None:
    busy = _thread().start_run(NOW, TTL)

    done = busy.end_run(ThreadStatus.IDLE, NOW + timedelta(seconds=3))

    assert (done.status, done.lease_until) == (ThreadStatus.IDLE, None)
    assert done.updated_at == NOW + timedelta(seconds=3)
    with pytest.raises(ValueError, match="busy"):
        busy.end_run(ThreadStatus.BUSY, NOW)
