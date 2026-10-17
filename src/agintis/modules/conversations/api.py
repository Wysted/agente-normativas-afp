"""Public surface of the conversations module: a pure Python facade (no web framework).

Threads are tenant-scoped conversations; the run lease (`AcquireRunLease` / `ReleaseRunLease`)
is what other modules reach, through an adapter owned by the composition root. The HTTP routes
live in `http.py`.
"""

from agintis.modules.conversations.application import (
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    AcquireRunLease,
    AgentDirectory,
    CreateThread,
    DeleteThread,
    GetThread,
    ListThreads,
    NoPendingQuestions,
    NoThreadState,
    PendingQuestion,
    PendingQuestions,
    ReleaseRunLease,
    ThreadDetail,
    ThreadPage,
    ThreadRepository,
    ThreadsServices,
    ThreadStateStore,
)
from agintis.modules.conversations.domain import (
    TITLE_MAX_LENGTH,
    InvalidCursorError,
    RunLease,
    RunMode,
    Thread,
    ThreadAgentNotFoundError,
    ThreadAgentUnsupportedError,
    ThreadBusyError,
    ThreadNotFoundError,
    ThreadsDisabledError,
    ThreadStatus,
)

__all__ = [
    "DEFAULT_PAGE_SIZE",
    "MAX_PAGE_SIZE",
    "TITLE_MAX_LENGTH",
    "AcquireRunLease",
    "AgentDirectory",
    "CreateThread",
    "DeleteThread",
    "GetThread",
    "InvalidCursorError",
    "ListThreads",
    "NoPendingQuestions",
    "NoThreadState",
    "PendingQuestion",
    "PendingQuestions",
    "ReleaseRunLease",
    "RunLease",
    "RunMode",
    "Thread",
    "ThreadAgentNotFoundError",
    "ThreadAgentUnsupportedError",
    "ThreadBusyError",
    "ThreadDetail",
    "ThreadNotFoundError",
    "ThreadPage",
    "ThreadRepository",
    "ThreadStateStore",
    "ThreadStatus",
    "ThreadsDisabledError",
    "ThreadsServices",
]
