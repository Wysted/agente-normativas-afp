"""Client-safe errors of the conversations module (messages never carry internal detail)."""

from agintis.shared_kernel.errors import AppError, ErrorKind


class ThreadNotFoundError(AppError):
    """The thread does not exist for the caller's tenant (cross-tenant ids look the same)."""

    kind = ErrorKind.NOT_FOUND
    code = "thread_not_found"

    def __init__(self) -> None:
        super().__init__("Thread does not exist")


class ThreadAgentNotFoundError(AppError):
    kind = ErrorKind.NOT_FOUND
    code = "agent_not_found"

    def __init__(self, agent_id: str) -> None:
        super().__init__(f"Agent {agent_id!r} does not exist")


class ThreadAgentUnsupportedError(AppError):
    """The agent keeps no conversation state, so a thread for it would be useless."""

    kind = ErrorKind.INVALID
    code = "agent_threads_unsupported"

    def __init__(self, agent_id: str) -> None:
        super().__init__(f"Agent {agent_id!r} does not support threads")


class ThreadsDisabledError(AppError):
    """Threads are switched off in this deployment (no checkpointer)."""

    kind = ErrorKind.UNAVAILABLE
    code = "threads_unavailable"

    def __init__(self) -> None:
        super().__init__("Conversation threads are not available")


class ThreadBusyError(AppError):
    """Another run holds the thread's lease."""

    kind = ErrorKind.CONFLICT
    code = "thread_busy"

    def __init__(self) -> None:
        super().__init__("Thread already has a run in progress")


class InvalidCursorError(AppError):
    kind = ErrorKind.INVALID
    code = "invalid_cursor"

    def __init__(self) -> None:
        super().__init__("The pagination cursor is not valid")
