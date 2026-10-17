"""Postgres `DocumentRepository`."""

from dataclasses import replace
from typing import Any

from psycopg import errors as pg_errors
from sqlalchemy import RowMapping, delete, func, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from agintis.modules.knowledge.domain import (
    Document,
    DocumentIdentityError,
    DocumentRepository,
    DocumentStatus,
    DuplicateDocumentError,
    TenantMismatchError,
)
from agintis.modules.knowledge.infrastructure.postgres.fields import DOCUMENT_FIELDS
from agintis.modules.knowledge.infrastructure.postgres.locks import xact_lock_keys
from agintis.modules.knowledge.infrastructure.postgres.tables import READY_SOURCE_INDEX, documents
from agintis.platform.db import UnitOfWork, apply_criteria
from agintis.shared_kernel.criteria import Criteria, ensure_fields, tenant_scoped
from agintis.shared_kernel.ids import DocumentId, TenantId

# Columns an upsert may change. Never `id`, `created_at` nor the IDENTITY columns `tenant_id`,
# `profile`, `source_uri`: the composite foreign key of chunks, the partial unique index and the
# per-source locks all assume they never change after the first insert.
_MUTABLE_COLUMNS = (
    "title",
    "mime_type",
    "content_hash",
    "profile_fingerprint",
    "metadata_hash",
    "status",
    "chunk_count",
)


def _to_document(row: RowMapping) -> Document:
    return Document(
        id=DocumentId(row["id"]),
        tenant_id=TenantId(row["tenant_id"]),
        profile=row["profile"],
        source_uri=row["source_uri"],
        title=row["title"],
        mime_type=row["mime_type"],
        content_hash=row["content_hash"],
        profile_fingerprint=row["profile_fingerprint"],
        metadata_hash=row["metadata_hash"],
        created_at=row["created_at"],
        status=DocumentStatus(row["status"]),
        chunk_count=row["chunk_count"],
    )


def _is_ready_source_violation(exc: IntegrityError) -> bool:
    orig = exc.orig
    return (
        isinstance(orig, pg_errors.UniqueViolation)
        and orig.diag.constraint_name == READY_SOURCE_INDEX
    )


class PgDocumentRepository(DocumentRepository):
    """Every operation runs in its own short transaction and is tenant-scoped."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._factory = session_factory

    async def save(self, document: Document) -> None:
        values: dict[str, Any] = {
            "id": document.id,
            "tenant_id": document.tenant_id,
            "profile": document.profile,
            "source_uri": document.source_uri,
            "title": document.title,
            "mime_type": document.mime_type,
            "content_hash": document.content_hash,
            "profile_fingerprint": document.profile_fingerprint,
            "metadata_hash": document.metadata_hash,
            "status": document.status.value,
            "chunk_count": document.chunk_count,
            "created_at": document.created_at,
        }
        insert = pg_insert(documents).values(values)
        upsert = insert.on_conflict_do_update(
            index_elements=[documents.c.id],
            set_={
                **{name: insert.excluded[name] for name in _MUTABLE_COLUMNS},
                "updated_at": func.now(),
            },
            # Identity guard: a row owned by another tenant, or whose profile/source differs, is
            # left untouched (0 rows returned) and diagnosed below.
            where=(
                (documents.c.tenant_id == insert.excluded.tenant_id)
                & (documents.c.profile == insert.excluded.profile)
                & (documents.c.source_uri == insert.excluded.source_uri)
            ),
        ).returning(documents.c.id)
        owner: str | None = None
        try:
            async with UnitOfWork(self._factory) as uow:
                saved = (await uow.session.execute(upsert)).scalar_one_or_none()
                if saved is None:
                    owner = (
                        await uow.session.execute(
                            select(documents.c.tenant_id).where(documents.c.id == document.id)
                        )
                    ).scalar_one_or_none()
        except IntegrityError as exc:
            if _is_ready_source_violation(exc):
                raise DuplicateDocumentError(document.source_uri) from exc
            raise
        if saved is None:
            if owner is not None and owner != document.tenant_id:
                raise TenantMismatchError(document.id)
            raise DocumentIdentityError(document.id)

    async def mark_ready(
        self, tenant_id: TenantId, document_id: DocumentId, chunk_count: int
    ) -> bool:
        pending = (
            (documents.c.id == document_id)
            & (documents.c.tenant_id == tenant_id)
            & (documents.c.status == DocumentStatus.PENDING.value)
        )
        async with UnitOfWork(self._factory) as uow:
            session = uow.session
            # Unlocked peek, only to derive the source lock key (identity columns are immutable).
            peek = (
                await session.execute(
                    select(documents.c.profile, documents.c.source_uri).where(pending)
                )
            ).one_or_none()
            if peek is None:
                return False
            # Serialize concurrent promotions of the same source until commit, so the
            # supersede+promote pair below never races with another one (the partial unique
            # index would otherwise reject the loser).
            high, low = xact_lock_keys(tenant_id, peek.profile, peek.source_uri)
            await session.execute(
                text("SELECT pg_advisory_xact_lock(:high, :low)"), {"high": high, "low": low}
            )
            # Re-read under the lock AND the row lock: a concurrent cleanup may have
            # removed/promoted it, and the source used below must be the locked row's own.
            source = (
                await session.execute(
                    select(documents.c.profile, documents.c.source_uri)
                    .where(pending)
                    .with_for_update()
                )
            ).one_or_none()
            if source is None or tuple(source) != tuple(peek):
                return False
            await session.execute(
                update(documents)
                .where(
                    documents.c.tenant_id == tenant_id,
                    documents.c.profile == source.profile,
                    documents.c.source_uri == source.source_uri,
                    documents.c.status == DocumentStatus.READY.value,
                    documents.c.id != document_id,
                )
                .values(status=DocumentStatus.SUPERSEDED.value, updated_at=func.now())
            )
            promoted = await session.execute(
                update(documents)
                .where(pending)
                .values(
                    status=DocumentStatus.READY.value,
                    chunk_count=chunk_count,
                    updated_at=func.now(),
                )
                .returning(documents.c.id)
            )
            return promoted.scalar_one_or_none() is not None

    async def update_title(self, tenant_id: TenantId, document_id: DocumentId, title: str) -> bool:
        stmt = (
            update(documents)
            .where(
                documents.c.id == document_id,
                documents.c.tenant_id == tenant_id,
                documents.c.status == DocumentStatus.READY.value,
            )
            .values(title=title, updated_at=func.now())
            .returning(documents.c.id)
        )
        async with UnitOfWork(self._factory) as uow:
            return (await uow.session.execute(stmt)).scalar_one_or_none() is not None

    async def get(self, tenant_id: TenantId, document_id: DocumentId) -> Document | None:
        stmt = select(documents).where(
            documents.c.id == document_id, documents.c.tenant_id == tenant_id
        )
        async with UnitOfWork(self._factory) as uow:
            row = (await uow.session.execute(stmt)).mappings().one_or_none()
        return None if row is None else _to_document(row)

    async def find_by_source(
        self, tenant_id: TenantId, profile: str, source_uri: str
    ) -> list[Document]:
        stmt = (
            select(documents)
            .where(
                documents.c.tenant_id == tenant_id,
                documents.c.profile == profile,
                documents.c.source_uri == source_uri,
            )
            .order_by(documents.c.created_at, documents.c.id)
        )
        async with UnitOfWork(self._factory) as uow:
            rows = (await uow.session.execute(stmt)).mappings().all()
        return [_to_document(row) for row in rows]

    async def delete(self, tenant_id: TenantId, document_id: DocumentId) -> None:
        stmt = delete(documents).where(
            documents.c.id == document_id, documents.c.tenant_id == tenant_id
        )
        async with UnitOfWork(self._factory) as uow:
            await uow.session.execute(stmt)  # chunks go away through ON DELETE CASCADE

    async def matching(self, tenant_id: TenantId, criteria: Criteria) -> list[Document]:
        ensure_fields(criteria, self.FILTERABLE_FIELDS)
        scoped = replace(criteria, filter=tenant_scoped(tenant_id, criteria.filter))
        stmt = apply_criteria(select(documents), scoped, DOCUMENT_FIELDS)
        async with UnitOfWork(self._factory) as uow:
            rows = (await uow.session.execute(stmt)).mappings().all()
        return [_to_document(row) for row in rows]
