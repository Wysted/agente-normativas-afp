from dataclasses import dataclass

from agintis.modules.conversations.application.cursor import (
    after_cursor,
    decode_cursor,
    encode_cursor,
)
from agintis.modules.conversations.application.ports import ThreadRepository
from agintis.modules.conversations.domain import Thread, ThreadStatus
from agintis.shared_kernel.criteria import Criteria, Filter, FilterExpr, Order, OrderDirection
from agintis.shared_kernel.ids import TenantId

DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100


@dataclass(frozen=True, slots=True)
class ThreadPage:
    items: tuple[Thread, ...]
    next_cursor: str | None


class ListThreads:
    """Newest-updated first; keyset pagination on (`updated_at` desc, `id`).

    Not a snapshot: `updated_at` changes whenever a run acquires or releases the thread, so a
    thread updated between two page requests can move (and be skipped or seen twice). Each page
    is consistent; clients that need completeness should re-list.
    """

    def __init__(self, threads: ThreadRepository, max_page_size: int = MAX_PAGE_SIZE) -> None:
        self._threads = threads
        self._max_page_size = max_page_size

    async def execute(
        self,
        tenant_id: TenantId,
        *,
        agent_id: str | None = None,
        status: ThreadStatus | None = None,
        limit: int = DEFAULT_PAGE_SIZE,
        cursor: str | None = None,
    ) -> ThreadPage:
        limit = min(max(limit, 1), self._max_page_size)
        filters: list[FilterExpr] = []
        if agent_id is not None:
            filters.append(Filter.eq("agent_id", agent_id))
        if status is not None:
            filters.append(Filter.eq("status", status.value))
        if cursor is not None:
            filters.append(after_cursor(*decode_cursor(cursor)))
        criteria = Criteria.all_of(*filters)
        # One extra row tells whether there is a next page.
        criteria = Criteria(
            filter=criteria.filter,
            order=(Order("updated_at", OrderDirection.DESC),),
            limit=limit + 1,
        )
        rows = await self._threads.search(tenant_id, criteria)
        page = rows[:limit]
        next_cursor = (
            encode_cursor(page[-1].updated_at, page[-1].id) if len(rows) > limit and page else None
        )
        return ThreadPage(tuple(page), next_cursor)
