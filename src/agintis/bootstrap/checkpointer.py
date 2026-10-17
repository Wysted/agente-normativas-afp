"""Checkpointer construction shared by the Container and the checkpointer_setup CLI."""

from agintis.bootstrap.settings import Settings
from agintis.platform.checkpointer import Checkpointer


def build_checkpointer(settings: Settings) -> Checkpointer:
    """The pool is created closed (no I/O); the caller opens it."""
    cp = settings.checkpointer
    return Checkpointer(
        settings.libpq_url,
        min_size=cp.pool_min_size,
        max_size=cp.pool_max_size,
        acquire_timeout_seconds=cp.pool_timeout_seconds,
        connect_timeout_seconds=settings.database.connect_timeout_seconds,
    )
