"""Criteria field maps: which names a filter may use and where they live in SQL."""

from collections.abc import Iterable, Mapping

from agintis.modules.knowledge.domain import (
    RESERVED_CHUNK_FIELDS,
    MetadataField,
    chunk_metadata_schema,
)
from agintis.modules.knowledge.infrastructure.postgres.tables import chunks, documents
from agintis.platform.db import ColumnField, FieldSpec, JsonbField

DOCUMENT_FIELDS: Mapping[str, FieldSpec] = {
    "id": ColumnField(documents.c.id, "str"),
    "tenant_id": ColumnField(documents.c.tenant_id, "str"),
    "profile": ColumnField(documents.c.profile, "str"),
    "source_uri": ColumnField(documents.c.source_uri, "str"),
    "title": ColumnField(documents.c.title, "str"),
    "mime_type": ColumnField(documents.c.mime_type, "str"),
    "content_hash": ColumnField(documents.c.content_hash, "str"),
    "profile_fingerprint": ColumnField(documents.c.profile_fingerprint, "str"),
    "created_at": ColumnField(documents.c.created_at, "datetime"),
    "status": ColumnField(documents.c.status, "str"),
    "chunk_count": ColumnField(documents.c.chunk_count, "number"),
}

CHUNK_CORE_FIELDS: Mapping[str, FieldSpec] = {
    "id": ColumnField(chunks.c.id, "str"),
    "document_id": ColumnField(chunks.c.document_id, "str"),
    "tenant_id": ColumnField(chunks.c.tenant_id, "str"),
    "profile": ColumnField(chunks.c.profile, "str"),
    "ordinal": ColumnField(chunks.c.ordinal, "number"),
    "text": ColumnField(chunks.c.text, "str"),
}

# The core columns and the names metadata may never use are one and the same set.
if set(CHUNK_CORE_FIELDS) != RESERVED_CHUNK_FIELDS:  # pragma: no cover - import-time guard
    raise RuntimeError("CHUNK_CORE_FIELDS drifted from RESERVED_CHUNK_FIELDS")

_JSONB_KIND = {
    "str": "str",
    "date": "str",
    "int": "number",
    "float": "number",
    "list[str]": "str",
    "list[int]": "number",
}


def metadata_spec(field: MetadataField) -> JsonbField:
    """JSONB-backed spec of a declared metadata field.

    `date` fields hold ISO-8601 strings and compare as text (no casts, immutable); `int` and
    `float` are one numeric kind (the oracle compares them numerically); `list[...]` fields are
    JSON arrays of their element kind, filtered as sets.
    """
    kind = _JSONB_KIND[field.type]
    return JsonbField(chunks.c.metadata, field.name, kind, array=field.is_list)  # type: ignore[arg-type]


def chunk_fields(declared: Iterable[MetadataField]) -> dict[str, FieldSpec]:
    """Core chunk columns plus the built-in and declared FILTERABLE metadata fields.

    The key set equals `chunk_filterable_fields(declared)`: it is the allowlist the converter
    maps, so a field name that is not here can never reach SQL.
    """
    mapped: dict[str, FieldSpec] = dict(CHUNK_CORE_FIELDS)
    schema = chunk_metadata_schema(declared)
    if overlap := CHUNK_CORE_FIELDS.keys() & schema.keys():
        # MetadataField forbids reserved names; this guards a drift between the two sets.
        raise ValueError(f"Metadata fields shadow core chunk fields: {sorted(overlap)}")
    for field in schema.values():
        if field.filterable:
            mapped[field.name] = metadata_spec(field)
    return mapped
