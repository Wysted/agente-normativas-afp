"""Thread: a tenant-scoped conversation whose graph state lives in the checkpointer.

The thread row only tracks WHO owns it (tenant, agent) and WHAT it is doing (`status`); the
messages themselves are LangGraph checkpoints keyed by the thread id.

Status machine. The lease is PURE MUTUAL EXCLUSION (one run at a time, with an expiry); it does
not look at what kind of run asks:

    idle | interrupted | expired busy --acquire--> busy --release--> idle | interrupted

Whether a question is pending is decided by the CHECKPOINT, never by `status`: `RunAgent` reads it
after every acquire, refuses a message over a pending interrupt (status corrected to
interrupted) and a resume without one (status corrected to idle). `status` is therefore a hint
for listings. A thread stays `busy` in responses after its lease expired (it is acquirable):
clients should treat `busy` as "maybe running" and rely on the 409 `thread_busy` of the run
endpoint.
"""

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from enum import StrEnum

from agintis.modules.conversations.domain.errors import ThreadBusyError
from agintis.shared_kernel.errors import InvalidRequestError
from agintis.shared_kernel.ids import TenantId, ThreadId

TITLE_MAX_LENGTH = 200


class ThreadStatus(StrEnum):
    IDLE = "idle"
    BUSY = "busy"
    INTERRUPTED = "interrupted"


class RunMode(StrEnum):
    """What a run does with the thread: a new user message, or the answer to an interrupt."""

    MESSAGE = "message"
    RESUME = "resume"


def normalize_title(title: str | None) -> str | None:
    """Trimmed title, `None` when blank; over-long titles are rejected."""
    if title is None:
        return None
    cleaned = title.strip()
    if not cleaned:
        return None
    if len(cleaned) > TITLE_MAX_LENGTH:
        raise InvalidRequestError(f"title exceeds {TITLE_MAX_LENGTH} characters")
    return cleaned


@dataclass(frozen=True, slots=True)
class Thread:
    id: ThreadId
    tenant_id: TenantId
    agent_id: str
    title: str | None
    status: ThreadStatus
    lease_until: datetime | None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def create(
        cls,
        thread_id: ThreadId,
        tenant_id: TenantId,
        agent_id: str,
        title: str | None,
        now: datetime,
    ) -> "Thread":
        return cls(
            id=thread_id,
            tenant_id=tenant_id,
            agent_id=agent_id,
            title=normalize_title(title),
            status=ThreadStatus.IDLE,
            lease_until=None,
            created_at=now,
            updated_at=now,
        )

    def is_leased(self, now: datetime) -> bool:
        """True while a run holds a valid (non expired) lease."""
        return (
            self.status is ThreadStatus.BUSY
            and self.lease_until is not None
            and self.lease_until >= now
        )

    def check_can_start(self, now: datetime) -> None:
        """Raise `ThreadBusyError` while another run holds a valid lease."""
        if self.is_leased(now):
            raise ThreadBusyError

    def start_run(self, now: datetime, ttl: timedelta) -> "Thread":
        self.check_can_start(now)
        return replace(self, status=ThreadStatus.BUSY, lease_until=now + ttl, updated_at=now)

    def end_run(self, outcome: ThreadStatus, now: datetime) -> "Thread":
        """Release the lease with `IDLE` (finished) or `INTERRUPTED` (waiting for an answer)."""
        if outcome is ThreadStatus.BUSY:
            raise ValueError("a run cannot end in the busy status")
        return replace(self, status=outcome, lease_until=None, updated_at=now)


@dataclass(frozen=True, slots=True)
class RunLease:
    """Proof that a run owns the thread until `lease_until`; `token` is needed to release it."""

    thread_id: ThreadId
    tenant_id: TenantId
    agent_id: str
    mode: RunMode
    token: str
    lease_until: datetime
