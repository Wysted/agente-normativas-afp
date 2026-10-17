"""knowledge schema: documents, chunks, es_unaccent text search config

Revision ID: 0001
Revises:
Create Date: 2026-10-15

The embedding width comes from `EMBEDDINGS_EXPECTED_DIM` (default 1024, bge-m3), handed over
by migrations/env.py through `Config.attributes["embedding_dim"]`. Changing the embedding model
to another dimension needs a NEW migration (alter the column, rebuild the HNSW index) and a
full re-embed: once released, this revision is a snapshot and must never be edited again
(it was edited before the first release to harden the extension steps and the chunk foreign
key).

Operations: extension management needs privileges. `CREATE EXTENSION vector` / `unaccent` and
`ALTER EXTENSION vector UPDATE` only run when the migration role can (superuser, or owner of the
extension for the update); otherwise they are skipped with a NOTICE and a DBA must run, once per
database and as a privileged role:

    CREATE EXTENSION IF NOT EXISTS vector;  ALTER EXTENSION vector UPDATE;
    CREATE EXTENSION IF NOT EXISTS unaccent;

The app refuses to start when pgvector is older than 0.8.4 (startup check).

Indexes on `MetadataField.indexed` fields are NOT part of this revision: each one needs its own
migration (expression index named `ix_chunks_meta_<field>`), see `jsonb_typed_expression`.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

DEFAULT_EMBEDDING_DIM = 1024

CREATE_TS_CONFIG = """
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_ts_config c JOIN pg_namespace n ON n.oid = c.cfgnamespace
        WHERE c.cfgname = 'es_unaccent' AND n.nspname = 'public'
    ) THEN
        CREATE TEXT SEARCH CONFIGURATION public.es_unaccent (COPY = pg_catalog.spanish);
        ALTER TEXT SEARCH CONFIGURATION public.es_unaccent
            ALTER MAPPING FOR hword, hword_part, word WITH unaccent, spanish_stem;
    END IF;
END
$$
"""


def _embedding_dim() -> int:
    config = op.get_context().config
    attributes = config.attributes if config is not None else {}
    return int(attributes.get("embedding_dim", DEFAULT_EMBEDDING_DIM))


# Guarded extension management (the compose init script already creates both on a fresh volume):
# a role that cannot create/update an extension must not break the whole migration when the
# extension is already in place; a missing extension then fails later with a clear error.
ENSURE_EXTENSION = """
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = '{name}') THEN
        BEGIN
            CREATE EXTENSION {name};
        EXCEPTION WHEN insufficient_privilege THEN
            RAISE NOTICE 'role % cannot CREATE EXTENSION {name}: ask a superuser to run it',
                current_user;
        END;
    END IF;
END
$$
"""

UPDATE_EXTENSION = """
DO $$
DECLARE
    installed text;
    available text;
    can_update boolean;
BEGIN
    SELECT e.extversion, pg_has_role(current_user, e.extowner, 'USAGE')
        INTO installed, can_update
        FROM pg_extension e WHERE e.extname = '{name}';
    SELECT default_version INTO available FROM pg_available_extensions WHERE name = '{name}';
    IF installed IS NULL OR installed IS NOT DISTINCT FROM available THEN
        RETURN;
    END IF;
    IF NOT (can_update OR (SELECT rolsuper FROM pg_roles WHERE rolname = current_user)) THEN
        RAISE NOTICE 'extension {name} is %, % is available; role % is not its owner: ask a '
            'privileged role to run ALTER EXTENSION {name} UPDATE', installed, available,
            current_user;
        RETURN;
    END IF;
    ALTER EXTENSION {name} UPDATE;
END
$$
"""


def upgrade() -> None:
    op.execute(ENSURE_EXTENSION.format(name="vector"))
    op.execute(UPDATE_EXTENSION.format(name="vector"))
    op.execute(ENSURE_EXTENSION.format(name="unaccent"))
    op.execute(CREATE_TS_CONFIG)

    op.create_table(
        "documents",
        sa.Column("id", sa.Text, nullable=False),
        sa.Column("tenant_id", sa.Text, nullable=False),
        sa.Column("profile", sa.Text, nullable=False),
        sa.Column("source_uri", sa.Text, nullable=False),
        sa.Column("title", sa.Text, nullable=False),
        sa.Column("mime_type", sa.Text, nullable=False),
        sa.Column("content_hash", sa.Text, nullable=False),
        sa.Column("profile_fingerprint", sa.Text, nullable=False),
        sa.Column("status", sa.Text, nullable=False),
        sa.Column("chunk_count", sa.Integer, nullable=False, server_default=sa.text("0")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_documents")),
        sa.UniqueConstraint(
            "id", "tenant_id", "profile", name=op.f("uq_documents_id_tenant_id_profile")
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'ready', 'superseded')", name=op.f("ck_documents_status_valid")
        ),
        sa.CheckConstraint("chunk_count >= 0", name=op.f("ck_documents_chunk_count_non_negative")),
    )
    op.create_index(
        "uq_documents_ready_source",
        "documents",
        ["tenant_id", "profile", "source_uri"],
        unique=True,
        postgresql_where=sa.text("status = 'ready'"),
    )
    op.create_index(
        "ix_documents_tenant_profile_source", "documents", ["tenant_id", "profile", "source_uri"]
    )

    op.create_table(
        "chunks",
        sa.Column("id", sa.Text, nullable=False),
        sa.Column("document_id", sa.Text, nullable=False),
        sa.Column("tenant_id", sa.Text, nullable=False),
        sa.Column("profile", sa.Text, nullable=False),
        sa.Column("ordinal", sa.Integer, nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column(
            "metadata",
            postgresql.JSONB,
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("embedding", Vector(_embedding_dim()), nullable=False),
        sa.Column(
            "tsv",
            postgresql.TSVECTOR,
            sa.Computed("to_tsvector('public.es_unaccent'::regconfig, text)", persisted=True),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_chunks")),
        sa.ForeignKeyConstraint(
            ["document_id", "tenant_id", "profile"],
            ["documents.id", "documents.tenant_id", "documents.profile"],
            name="fk_chunks_document_tenant",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint("document_id", "ordinal", name=op.f("uq_chunks_document_id_ordinal")),
    )
    op.create_index(
        "ix_chunks_embedding_hnsw",
        "chunks",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
    op.create_index("ix_chunks_tsv", "chunks", ["tsv"], postgresql_using="gin")
    op.create_index(
        "ix_chunks_metadata",
        "chunks",
        ["metadata"],
        postgresql_using="gin",
        postgresql_ops={"metadata": "jsonb_path_ops"},
    )
    op.create_index(
        "ix_chunks_tenant_profile_document", "chunks", ["tenant_id", "profile", "document_id"]
    )


def downgrade() -> None:
    op.drop_table("chunks")  # indexes go with the table
    op.drop_table("documents")
    op.execute("DROP TEXT SEARCH CONFIGURATION IF EXISTS public.es_unaccent")
    # Extensions are deliberately kept: they are shared infrastructure (init script, other users).
