from dataclasses import dataclass

from agintis.modules.conversations.application.ports import (
    NoPendingQuestions,
    PendingQuestion,
    PendingQuestions,
    ThreadRepository,
)
from agintis.modules.conversations.domain import Thread, ThreadNotFoundError, ThreadStatus
from agintis.shared_kernel.ids import TenantId, ThreadId


@dataclass(frozen=True, slots=True)
class ThreadDetail:
    thread: Thread
    pending: tuple[PendingQuestion, ...]


class GetThread:
    def __init__(
        self, threads: ThreadRepository, questions: PendingQuestions | None = None
    ) -> None:
        self._threads = threads
        self._questions = questions or NoPendingQuestions()

    async def execute(self, tenant_id: TenantId, thread_id: ThreadId) -> Thread:
        thread = await self._threads.get(tenant_id, thread_id)
        if thread is None:
            raise ThreadNotFoundError
        return thread

    async def detail(self, tenant_id: TenantId, thread_id: ThreadId) -> ThreadDetail:
        """The thread plus its pending questions, so a client that lost the stream after an
        `interrupted` can recover them (with their interrupt ids). The checkpoint is only read
        for an interrupted thread; the tenant check comes first."""
        thread = await self.execute(tenant_id, thread_id)
        if thread.status is not ThreadStatus.INTERRUPTED:
            return ThreadDetail(thread, ())
        return ThreadDetail(thread, await self._questions.for_thread(thread.agent_id, thread.id))
