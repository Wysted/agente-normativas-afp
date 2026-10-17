"""SQLAlchemy Core tables of the conversations module (Alembic autogenerates from these).

`id` is the primary key ALONE (not per tenant): the same string is the LangGraph checkpoint
`thread_id`, whose namespace is global, so two tenants must never share one. Every query still
filters by tenant.
"""

from sqlalchemy import CheckConstraint, Column, DateTime, Index, Table, Text, func

from agintis.modules.conversations.domain import ThreadStatus
from agintis.platform.db import metadata

_STATUSES = ", ".join(f"'{status.value}'" for status in ThreadStatus)

threads = Table(
    "threads",
    metadata,
    Column("id", Text, primary_key=True),
    Column("tenant_id", Text, nullable=False),
    Column("agent_id", Text, nullable=False),
    Column("title", Text, nullable=True),
    Column("status", Text, nullable=False),
    # Lease of the run in progress: both set exactly while the status is busy.
    Column("lease_until", DateTime(timezone=True), nullable=True),
    Column("lease_token", Text, nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    CheckConstraint(f"status IN ({_STATUSES})", name="status_valid"),
    CheckConstraint(
        "(status = 'busy') = (lease_until IS NOT NULL AND lease_token IS NOT NULL)",
        name="lease_matches_status",
    ),
    CheckConstraint("(lease_until IS NULL) = (lease_token IS NULL)", name="lease_fields_together"),
    Index("ix_threads_tenant_agent_status", "tenant_id", "agent_id", "status"),
)

# Listing: newest updated first within a tenant. It must match the ORDER BY the Criteria converter
# emits (`updated_at DESC NULLS LAST, id COLLATE "C" ASC NULLS LAST`), otherwise the planner
# cannot use it for the sort. Alembic reflects the COLLATE expression as a plain column, so the
# drift check cannot compare this index (migrations/env.py skips it); the EXPLAIN integration test
# (`test_the_list_index_serves_the_criteria_order_by_without_a_sort`) guards it instead.
THREADS_ORDER_INDEX = "ix_threads_tenant_updated"
Index(
    THREADS_ORDER_INDEX,
    threads.c.tenant_id,
    threads.c.updated_at.desc().nulls_last(),
    threads.c.id.collate("C"),
)
