"""Stable keys for per-source locks (ingest lock and mark_ready serialization)."""

import hashlib
import struct

from agintis.shared_kernel.ids import TenantId

_SEPARATOR = "\x00"


def source_digest(tenant_id: TenantId, profile: str, source_uri: str) -> bytes:
    """sha256 of the unambiguous (NUL-separated) identity of a source."""
    return hashlib.sha256(
        _SEPARATOR.join((tenant_id, profile, source_uri)).encode("utf-8", "surrogatepass")
    ).digest()


def session_lock_key(tenant_id: TenantId, profile: str, source_uri: str) -> int:
    """Signed 64-bit key for `pg_advisory_lock(bigint)`."""
    (key,) = struct.unpack(">q", source_digest(tenant_id, profile, source_uri)[:8])
    return int(key)


def xact_lock_keys(tenant_id: TenantId, profile: str, source_uri: str) -> tuple[int, int]:
    """Two signed 32-bit keys for `pg_advisory_xact_lock(int, int)`.

    The two-int form lives in a different key space than the bigint form, so it can never
    collide with the ingest lock a caller holds on the same source at the same time.
    """
    high, low = struct.unpack(">ii", source_digest(tenant_id, profile, source_uri)[8:16])
    return int(high), int(low)


def index_sync_lock_keys() -> tuple[int, int]:
    """Two signed 32-bit keys of the metadata index sync (`pg_advisory_lock(int, int)`).

    One fixed key for the whole sync run: two concurrent syncs would otherwise race on the same
    DROP / CREATE INDEX CONCURRENTLY and leave invalid indexes behind.
    """
    high, low = struct.unpack(">ii", hashlib.sha256(b"agintis:metadata_index_sync").digest()[:8])
    return int(high), int(low)
