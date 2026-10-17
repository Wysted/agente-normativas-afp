from collections.abc import Callable

from agintis.modules.conversations.application.ports import AgentDirectory, ThreadRepository
from agintis.modules.conversations.domain import (
    Thread,
    ThreadAgentNotFoundError,
    ThreadAgentUnsupportedError,
    ThreadsDisabledError,
)
from agintis.shared_kernel.ids import TenantId, ThreadId, new_id
from agintis.shared_kernel.ports import Clock


class CreateThread:
    def __init__(
        self,
        threads: ThreadRepository,
        agents: AgentDirectory,
        clock: Clock,
        id_generator: Callable[[], str] = new_id,
        threads_enabled: bool = True,
    ) -> None:
        self._threads_enabled = threads_enabled
        self._threads = threads
        self._agents = agents
        self._clock = clock
        self._id_generator = id_generator

    async def execute(self, tenant_id: TenantId, agent_id: str, title: str | None = None) -> Thread:
        if not self._agents.exists(agent_id):
            raise ThreadAgentNotFoundError(agent_id)
        if not self._threads_enabled:
            raise ThreadsDisabledError
        if not self._agents.supports_threads(agent_id):
            raise ThreadAgentUnsupportedError(agent_id)
        thread = Thread.create(
            ThreadId(self._id_generator()), tenant_id, agent_id, title, self._clock.now()
        )
        await self._threads.add(thread)
        return thread
