"""Criteria field map of `threads`: the allowlist of names a filter or order may use."""

from collections.abc import Mapping

from agintis.modules.conversations.application import THREAD_FILTERABLE_FIELDS
from agintis.modules.conversations.infrastructure.postgres.tables import threads
from agintis.platform.db import ColumnField, FieldSpec

THREAD_FIELDS: Mapping[str, FieldSpec] = {
    "id": ColumnField(threads.c.id, "str"),
    "tenant_id": ColumnField(threads.c.tenant_id, "str"),
    "agent_id": ColumnField(threads.c.agent_id, "str"),
    "title": ColumnField(threads.c.title, "str"),
    "status": ColumnField(threads.c.status, "str"),
    "lease_until": ColumnField(threads.c.lease_until, "datetime"),
    "created_at": ColumnField(threads.c.created_at, "datetime"),
    "updated_at": ColumnField(threads.c.updated_at, "datetime"),
}

if set(THREAD_FIELDS) != THREAD_FILTERABLE_FIELDS:  # pragma: no cover - import-time guard
    raise RuntimeError("THREAD_FIELDS drifted from THREAD_FILTERABLE_FIELDS")
