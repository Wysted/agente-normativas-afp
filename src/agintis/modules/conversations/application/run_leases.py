import logging
from datetime import timedelta

from agintis.modules.conversations.application.ports import ThreadRepository
from agintis.modules.conversations.domain import RunLease, RunMode, ThreadStatus
from agintis.shared_kernel.ids import TenantId, ThreadId

logger = logging.getLogger(__name__)

RELEASE_OUTCOMES = frozenset({ThreadStatus.IDLE, ThreadStatus.INTERRUPTED})


class AcquireRunLease:
    def __init__(self, threads: ThreadRepository, ttl: timedelta) -> None:
        self._threads = threads
        self._ttl = ttl

    async def execute(self, tenant_id: TenantId, thread_id: ThreadId, mode: RunMode) -> RunLease:
        return await self._threads.acquire_run_lease(tenant_id, thread_id, mode, self._ttl)


class ReleaseRunLease:
    def __init__(self, threads: ThreadRepository) -> None:
        self._threads = threads

    async def execute(
        self, tenant_id: TenantId, thread_id: ThreadId, token: str, outcome: ThreadStatus
    ) -> bool:
        """False (and a warning) when the lease was no longer ours."""
        if outcome not in RELEASE_OUTCOMES:
            raise ValueError(f"A run cannot end in the {outcome.value!r} status")
        released = await self._threads.release_run_lease(tenant_id, thread_id, token, outcome)
        if not released:
            logger.warning("Lease of thread %s was lost before its run ended", thread_id)
        return released
