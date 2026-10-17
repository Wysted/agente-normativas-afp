"""threads: tenant-scoped conversation threads with a run lease

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-16

`threads.id` is also the LangGraph checkpoint `thread_id` (global namespace), hence the primary
key on `id` alone. The checkpoint tables themselves are NOT part of Alembic: LangGraph owns and
migrates them (`python -m agintis.cli.checkpointer_setup`).
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "threads",
        sa.Column("id", sa.Text, nullable=False),
        sa.Column("tenant_id", sa.Text, nullable=False),
        sa.Column("agent_id", sa.Text, nullable=False),
        sa.Column("title", sa.Text, nullable=True),
        sa.Column("status", sa.Text, nullable=False),
        sa.Column("lease_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("lease_token", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_threads")),
        sa.CheckConstraint(
            "status IN ('idle', 'busy', 'interrupted')", name=op.f("ck_threads_status_valid")
        ),
        sa.CheckConstraint(
            "(status = 'busy') = (lease_until IS NOT NULL AND lease_token IS NOT NULL)",
            name=op.f("ck_threads_lease_matches_status"),
        ),
        sa.CheckConstraint(
            "(lease_until IS NULL) = (lease_token IS NULL)",
            name=op.f("ck_threads_lease_fields_together"),
        ),
    )
    op.create_index(
        "ix_threads_tenant_updated", "threads", ["tenant_id", sa.text("updated_at DESC"), "id"]
    )
    op.create_index(
        "ix_threads_tenant_agent_status", "threads", ["tenant_id", "agent_id", "status"]
    )


def downgrade() -> None:
    op.drop_table("threads")  # indexes go with the table
