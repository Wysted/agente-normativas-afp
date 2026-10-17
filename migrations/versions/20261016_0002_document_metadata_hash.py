"""documents.metadata_hash: re-ingest a document whose index metadata changed

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-16

The chunk text (heading prefixes) and chunk metadata derive from the caller's document metadata,
so an ingest is only a no-op when the content hash, the profile fingerprint AND this hash match.
Existing rows get NULL, which never matches: each document is reprocessed once on its next ingest.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("metadata_hash", sa.Text, nullable=True))


def downgrade() -> None:
    op.drop_column("documents", "metadata_hash")
