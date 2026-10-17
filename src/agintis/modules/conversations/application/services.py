from dataclasses import dataclass

from agintis.modules.conversations.application.create_thread import CreateThread
from agintis.modules.conversations.application.delete_thread import DeleteThread
from agintis.modules.conversations.application.get_thread import GetThread
from agintis.modules.conversations.application.list_threads import ListThreads
from agintis.modules.conversations.application.run_leases import AcquireRunLease, ReleaseRunLease


@dataclass(frozen=True, slots=True)
class ThreadsServices:
    """What the HTTP routes and the cross-module adapters need, built by the composition root."""

    create: CreateThread
    get: GetThread
    list: ListThreads
    delete: DeleteThread
    acquire_lease: AcquireRunLease
    release_lease: ReleaseRunLease
