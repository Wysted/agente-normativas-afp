"""Contract every ThreadRepository backend must satisfy: tenant isolation first, then the lease."""

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta

import pytest

from agintis.modules.conversations.application import ListThreads, ThreadRepository
from agintis.modules.conversations.domain import (
    RunMode,
    Thread,
    ThreadBusyError,
    ThreadNotFoundError,
    ThreadStatus,
)
from agintis.shared_kernel.criteria import (
    AllOf,
    AnyOf,
    Criteria,
    Filter,
    Order,
    OrderDirection,
    UnknownFieldError,
)
from agintis.shared_kernel.ids import TenantId, ThreadId

A = TenantId("tenant-a")
B = TenantId("tenant-b")
LONG = timedelta(minutes=5)
type Expire = Callable[[ThreadId], Awaitable[None]]
BASE = datetime(2026, 1, 1, tzinfo=UTC)


def _thread(
    thread_id: str,
    tenant: TenantId = A,
    *,
    agent: str = "chat",
    title: str | None = None,
    minutes: int = 0,
) -> Thread:
    return Thread.create(
        ThreadId(thread_id), tenant, agent, title, BASE + timedelta(minutes=minutes)
    )


async def _ids(repo: ThreadRepository, tenant: TenantId, criteria: Criteria) -> list[str]:
    return [t.id for t in await repo.search(tenant, criteria)]


async def test_add_and_get_round_trip(repo: ThreadRepository) -> None:
    thread = _thread("t1", title="Plan")
    await repo.add(thread)

    assert await repo.get(A, ThreadId("t1")) == thread
    assert await repo.get(A, ThreadId("missing")) is None


async def test_get_never_crosses_tenants(repo: ThreadRepository) -> None:
    await repo.add(_thread("t1"))

    assert await repo.get(B, ThreadId("t1")) is None


async def test_search_is_tenant_scoped_even_against_an_or_filter(repo: ThreadRepository) -> None:
    await repo.add(_thread("a1"))
    await repo.add(_thread("b1", B))
    escape = AnyOf(Filter.eq("tenant_id", "tenant-b"), Filter.eq("agent_id", "chat"))

    assert await _ids(repo, A, Criteria()) == ["a1"]
    assert await _ids(repo, A, Criteria(filter=escape)) == ["a1"]
    assert await _ids(repo, A, Criteria(filter=Filter.eq("id", "b1"))) == []


async def test_search_filters_orders_and_paginates(repo: ThreadRepository) -> None:
    await repo.add(_thread("t1", minutes=1, agent="chat", title="x"))
    await repo.add(_thread("t2", minutes=3, agent="chat"))
    await repo.add(_thread("t3", minutes=2, agent="other"))
    await repo.add(_thread("t4", minutes=3, agent="chat"))
    newest_first = (Order("updated_at", OrderDirection.DESC),)

    assert await _ids(repo, A, Criteria(order=newest_first)) == ["t2", "t4", "t3", "t1"]
    assert await _ids(repo, A, Criteria(order=newest_first, limit=2, offset=1)) == ["t4", "t3"]
    both = Criteria(filter=AllOf(Filter.eq("agent_id", "chat"), Filter.eq("status", "idle")))
    assert sorted(await _ids(repo, A, both)) == ["t1", "t2", "t4"]
    assert await _ids(repo, A, Criteria(filter=Filter.eq("title", "x"))) == ["t1"]
    assert await _ids(repo, A, Criteria(filter=Filter.neq("title", "x"))) == ["t2", "t3", "t4"]
    after = Criteria(filter=Filter.gt("updated_at", BASE + timedelta(minutes=2)))
    assert await _ids(repo, A, after) == ["t2", "t4"]


async def test_search_rejects_unknown_fields(repo: ThreadRepository) -> None:
    with pytest.raises(UnknownFieldError):
        await repo.search(A, Criteria(filter=Filter.eq("lease_token", "x")))
    with pytest.raises(UnknownFieldError):
        await repo.search(A, Criteria(order=(Order("secret"),)))


async def test_keyset_pagination_walks_every_row_once(repo: ThreadRepository) -> None:
    for number in range(7):  # pairs share an instant: the id tiebreak must keep pages stable
        await repo.add(_thread(f"t{number}", minutes=number // 2))
    listing = ListThreads(repo, max_page_size=3)

    seen: list[str] = []
    cursor = None
    while True:
        page = await listing.execute(A, cursor=cursor)
        seen.extend(t.id for t in page.items)
        cursor = page.next_cursor
        if cursor is None:
            break

    assert seen == ["t6", "t4", "t5", "t2", "t3", "t0", "t1"]


async def test_delete_is_tenant_scoped(repo: ThreadRepository) -> None:
    await repo.add(_thread("t1"))

    assert await repo.delete(B, ThreadId("t1")) is False
    assert await repo.get(A, ThreadId("t1")) is not None
    assert await repo.delete(A, ThreadId("t1")) is True
    assert await repo.get(A, ThreadId("t1")) is None
    assert await repo.delete(A, ThreadId("t1")) is False


async def test_delete_refuses_a_valid_lease_but_not_an_expired_one(
    repo: ThreadRepository, expire: Expire
) -> None:
    await repo.add(_thread("t1"))
    await repo.acquire_run_lease(A, ThreadId("t1"), RunMode.MESSAGE, LONG)
    assert await repo.delete(A, ThreadId("t1")) is False

    await repo.add(_thread("t2"))
    await repo.acquire_run_lease(A, ThreadId("t2"), RunMode.MESSAGE, LONG)
    await expire(ThreadId("t2"))
    assert await repo.delete(A, ThreadId("t2")) is True


async def test_acquire_makes_the_thread_busy_and_names_the_agent(repo: ThreadRepository) -> None:
    await repo.add(_thread("t1", agent="chat"))

    lease = await repo.acquire_run_lease(A, ThreadId("t1"), RunMode.MESSAGE, LONG)

    assert (lease.thread_id, lease.agent_id, lease.mode) == ("t1", "chat", RunMode.MESSAGE)
    assert lease.token
    thread = await repo.get(A, ThreadId("t1"))
    assert thread is not None
    assert thread.status is ThreadStatus.BUSY
    assert thread.lease_until is not None
    assert thread.lease_until > datetime.now(UTC)


async def test_acquire_is_tenant_scoped_and_reports_missing_threads(
    repo: ThreadRepository,
) -> None:
    await repo.add(_thread("t1"))

    with pytest.raises(ThreadNotFoundError):
        await repo.acquire_run_lease(B, ThreadId("t1"), RunMode.MESSAGE, LONG)
    with pytest.raises(ThreadNotFoundError):
        await repo.acquire_run_lease(A, ThreadId("nope"), RunMode.MESSAGE, LONG)
    thread = await repo.get(A, ThreadId("t1"))
    assert thread is not None
    assert thread.status is ThreadStatus.IDLE


async def test_a_second_acquire_conflicts_for_both_modes(repo: ThreadRepository) -> None:
    await repo.add(_thread("t1"))
    await repo.acquire_run_lease(A, ThreadId("t1"), RunMode.MESSAGE, LONG)

    for mode in RunMode:
        with pytest.raises(ThreadBusyError):
            await repo.acquire_run_lease(A, ThreadId("t1"), mode, LONG)


async def test_the_lease_is_acquirable_for_any_mode_from_any_resting_status(
    repo: ThreadRepository,
) -> None:
    """Pure mutual exclusion: idle and interrupted accept a message and a resume alike."""
    for number, mode in enumerate(RunMode):
        await repo.add(_thread(f"idle{number}"))
        assert (await repo.acquire_run_lease(A, ThreadId(f"idle{number}"), mode, LONG)).mode is mode

        await repo.add(_thread(f"int{number}"))
        first = await repo.acquire_run_lease(A, ThreadId(f"int{number}"), RunMode.MESSAGE, LONG)
        await repo.release_run_lease(
            A, ThreadId(f"int{number}"), first.token, ThreadStatus.INTERRUPTED
        )
        assert (await repo.acquire_run_lease(A, ThreadId(f"int{number}"), mode, LONG)).mode is mode


async def test_release_sets_the_outcome_and_clears_the_lease(repo: ThreadRepository) -> None:
    await repo.add(_thread("t1"))
    lease = await repo.acquire_run_lease(A, ThreadId("t1"), RunMode.MESSAGE, LONG)

    assert await repo.release_run_lease(A, ThreadId("t1"), lease.token, ThreadStatus.IDLE)

    thread = await repo.get(A, ThreadId("t1"))
    assert thread is not None
    assert (thread.status, thread.lease_until) == (ThreadStatus.IDLE, None)
    # And the thread can run again.
    await repo.acquire_run_lease(A, ThreadId("t1"), RunMode.MESSAGE, LONG)


async def test_release_needs_the_right_token_and_tenant(repo: ThreadRepository) -> None:
    await repo.add(_thread("t1"))
    lease = await repo.acquire_run_lease(A, ThreadId("t1"), RunMode.MESSAGE, LONG)

    assert not await repo.release_run_lease(A, ThreadId("t1"), "wrong", ThreadStatus.IDLE)
    assert not await repo.release_run_lease(B, ThreadId("t1"), lease.token, ThreadStatus.IDLE)
    assert not await repo.release_run_lease(A, ThreadId("nope"), lease.token, ThreadStatus.IDLE)
    thread = await repo.get(A, ThreadId("t1"))
    assert thread is not None
    assert thread.status is ThreadStatus.BUSY
    assert await repo.release_run_lease(A, ThreadId("t1"), lease.token, ThreadStatus.IDLE)
    assert not await repo.release_run_lease(A, ThreadId("t1"), lease.token, ThreadStatus.IDLE)


async def test_an_expired_lease_can_be_taken_over_and_the_old_release_is_ignored(
    repo: ThreadRepository, expire: Expire
) -> None:
    await repo.add(_thread("t1"))
    old = await repo.acquire_run_lease(A, ThreadId("t1"), RunMode.MESSAGE, LONG)
    await expire(ThreadId("t1"))

    new = await repo.acquire_run_lease(A, ThreadId("t1"), RunMode.MESSAGE, LONG)

    assert new.token != old.token
    assert not await repo.release_run_lease(A, ThreadId("t1"), old.token, ThreadStatus.IDLE)
    thread = await repo.get(A, ThreadId("t1"))
    assert thread is not None
    assert thread.status is ThreadStatus.BUSY  # the newer run still owns it


async def test_an_expired_lease_is_acquirable_for_any_mode(
    repo: ThreadRepository, expire: Expire
) -> None:
    """A dead run (message or resume) must not strand the thread."""
    await repo.add(_thread("t1"))
    first = await repo.acquire_run_lease(A, ThreadId("t1"), RunMode.MESSAGE, LONG)
    await repo.release_run_lease(A, ThreadId("t1"), first.token, ThreadStatus.INTERRUPTED)
    await repo.acquire_run_lease(A, ThreadId("t1"), RunMode.RESUME, LONG)  # dies, never released
    with pytest.raises(ThreadBusyError):
        await repo.acquire_run_lease(A, ThreadId("t1"), RunMode.RESUME, LONG)
    await expire(ThreadId("t1"))

    retry = await repo.acquire_run_lease(A, ThreadId("t1"), RunMode.RESUME, LONG)
    await expire(ThreadId("t1"))
    message = await repo.acquire_run_lease(A, ThreadId("t1"), RunMode.MESSAGE, LONG)

    assert (retry.mode, message.mode) == (RunMode.RESUME, RunMode.MESSAGE)


async def test_concurrent_acquires_have_exactly_one_winner(repo: ThreadRepository) -> None:
    await repo.add(_thread("t1"))

    results = await asyncio.gather(
        *(repo.acquire_run_lease(A, ThreadId("t1"), RunMode.MESSAGE, LONG) for _ in range(6)),
        return_exceptions=True,
    )

    winners = [r for r in results if not isinstance(r, BaseException)]
    losers = [r for r in results if isinstance(r, BaseException)]
    assert len(winners) == 1
    assert len(losers) == 5
    assert all(isinstance(r, ThreadBusyError) for r in losers)
