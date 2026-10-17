import re

from agintis.modules.agents_catalog.application.ports import AgentRunner
from agintis.shared_kernel.errors import AppError, ErrorKind

# What a client may send as `agent_id` and the rule every registered agent id must follow.
AGENT_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"
_AGENT_ID = re.compile(AGENT_ID_PATTERN)


class AgentNotFoundError(AppError):
    kind = ErrorKind.NOT_FOUND
    code = "agent_not_found"

    def __init__(self, agent_id: str) -> None:
        super().__init__(f"Agent {agent_id!r} does not exist")
        self.agent_id = agent_id


class AgentRegistry:
    """Agents available to run, keyed by `agent_id` (insertion order preserved)."""

    def __init__(self, runners: tuple[AgentRunner, ...] = ()) -> None:
        self._runners: dict[str, AgentRunner] = {}
        for runner in runners:
            self.register(runner)

    def register(self, runner: AgentRunner) -> None:
        if not _AGENT_ID.fullmatch(runner.agent_id):
            raise ValueError(f"Agent id {runner.agent_id!r} must match {AGENT_ID_PATTERN}")
        if runner.agent_id in self._runners:
            raise ValueError(f"Agent {runner.agent_id!r} is already registered")
        self._runners[runner.agent_id] = runner

    def __contains__(self, agent_id: object) -> bool:
        return agent_id in self._runners

    def get(self, agent_id: str) -> AgentRunner:
        runner = self._runners.get(agent_id)
        if runner is None:
            raise AgentNotFoundError(agent_id)
        return runner

    def list(self) -> tuple[AgentRunner, ...]:
        return tuple(self._runners.values())
