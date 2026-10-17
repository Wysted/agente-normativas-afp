import logging

from agintis.modules.conversations.application.ports import ThreadRepository, ThreadStateStore
from agintis.modules.conversations.domain import ThreadBusyError, ThreadNotFoundError
from agintis.shared_kernel.ids import TenantId, ThreadId

logger = logging.getLogger(__name__)


class DeleteThread:
    """Deletes the thread row (refused while a run holds the lease) and then its state.

    The row goes first, atomically: once it is gone nobody can start a run on the id, so the
    checkpoint cleanup that follows cannot race with a writer. If that cleanup fails the
    checkpoints are orphaned (unreachable, ids are random) and the failure is logged.
    """

    def __init__(self, threads: ThreadRepository, state: ThreadStateStore) -> None:
        self._threads = threads
        self._state = state

    async def execute(self, tenant_id: TenantId, thread_id: ThreadId) -> None:
        if not await self._threads.delete(tenant_id, thread_id):
            if await self._threads.get(tenant_id, thread_id) is None:
                raise ThreadNotFoundError
            raise ThreadBusyError
        try:
            await self._state.delete(thread_id)
        except Exception:
            logger.exception("Thread %s was deleted but its checkpoints could not be", thread_id)
