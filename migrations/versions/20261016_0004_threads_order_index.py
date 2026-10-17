"""threads: list index matching the Criteria ORDER BY (NULLS LAST, id COLLATE "C")

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-16

The Criteria converter orders by `updated_at DESC NULLS LAST, id COLLATE "C" ASC NULLS LAST`; the
index of revision 0003 (`updated_at DESC, id`) has another null ordering and the database default
collation for `id`, so the planner could not use it to avoid the sort.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_index("ix_threads_tenant_updated", table_name="threads")
    op.create_index(
        "ix_threads_tenant_updated",
        "threads",
        ["tenant_id", sa.text("updated_at DESC NULLS LAST"), sa.text('(id COLLATE "C")')],
    )


def downgrade() -> None:
    op.drop_index("ix_threads_tenant_updated", table_name="threads")
    op.create_index(
        "ix_threads_tenant_updated", "threads", ["tenant_id", sa.text("updated_at DESC"), "id"]
    )
