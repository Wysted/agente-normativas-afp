"""Opaque pagination cursor: base64url JSON of the last row's sort key (`updated_at`, `id`)."""

import base64
import binascii
import json
from datetime import datetime

from agintis.modules.conversations.domain import InvalidCursorError
from agintis.shared_kernel.criteria import AllOf, AnyOf, Filter, FilterExpr


def encode_cursor(updated_at: datetime, thread_id: str) -> str:
    raw = json.dumps({"u": updated_at.isoformat(), "i": thread_id}, separators=(",", ":"))
    return base64.urlsafe_b64encode(raw.encode()).decode().rstrip("=")


def decode_cursor(cursor: str) -> tuple[datetime, str]:
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded.encode()))
        updated_at = datetime.fromisoformat(payload["u"])
        thread_id = payload["i"]
    except (ValueError, KeyError, TypeError, binascii.Error):
        raise InvalidCursorError from None
    if not isinstance(thread_id, str) or updated_at.utcoffset() is None:
        raise InvalidCursorError
    return updated_at, thread_id


def after_cursor(updated_at: datetime, thread_id: str) -> FilterExpr:
    """Rows after the cursor in the order (`updated_at` desc, `id` asc)."""
    return AnyOf(
        Filter.lt("updated_at", updated_at),
        AllOf(Filter.eq("updated_at", updated_at), Filter.gt("id", thread_id)),
    )
