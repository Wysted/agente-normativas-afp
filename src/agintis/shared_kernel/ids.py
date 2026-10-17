"""Typed identifiers. NewType keeps them plain strings at runtime."""

from typing import NewType
from uuid import uuid4

TenantId = NewType("TenantId", str)
DocumentId = NewType("DocumentId", str)
ChunkId = NewType("ChunkId", str)
RunId = NewType("RunId", str)
ThreadId = NewType("ThreadId", str)


def new_id() -> str:
    """Return a fresh random identifier (uuid4 hex; stdlib has no uuid7 in 3.12)."""
    return uuid4().hex
