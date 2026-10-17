"""Async SQLAlchemy engine and session factory (single driver: psycopg3).

Pure factories with primitive arguments (bootstrap reads the settings). Creating an engine does
no I/O: connections open lazily on first use. Whoever creates an engine disposes it
(`await engine.dispose()`; the Container registers it on its AsyncExitStack).
"""

import math
from typing import Any

from pydantic import SecretStr
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

DEFAULT_CONNECT_TIMEOUT_SECONDS = 5.0
# TCP keepalives (libpq): a half-open connection (client or server vanished) is detected after
# roughly idle + interval * count = 30 + 10 * 3 seconds instead of never.
_KEEPALIVES: dict[str, Any] = {
    "keepalives": 1,
    "keepalives_idle": 30,
    "keepalives_interval": 10,
    "keepalives_count": 3,
}


def _connect_args(
    connect_timeout_seconds: float, statement_timeout_ms: int | None, read_only: bool = False
) -> dict[str, Any]:
    args: dict[str, Any] = {
        # libpq takes whole seconds (0 would mean "wait forever": never go below 1).
        "connect_timeout": max(1, math.ceil(connect_timeout_seconds)),
        **_KEEPALIVES,
    }
    options: list[str] = []
    if statement_timeout_ms is not None:
        options.append(f"-c statement_timeout={int(statement_timeout_ms)}")
    if read_only:
        options.append("-c default_transaction_read_only=on")
    if options:
        args["options"] = " ".join(options)
    return args


def create_engine(
    url: SecretStr,
    *,
    pool_size: int,
    max_overflow: int,
    pool_timeout_seconds: float,
    pool_recycle_seconds: int,
    connect_timeout_seconds: float = DEFAULT_CONNECT_TIMEOUT_SECONDS,
    statement_timeout_ms: int | None = None,
    read_only: bool = False,
    echo: bool = False,
) -> AsyncEngine:
    """Pooled engine for regular queries. `pool_pre_ping` survives idle-killed connections.

    `statement_timeout_ms` (None = server default) caps every statement of this engine; the
    value is a validated int, never interpolated user text. `read_only` makes every transaction
    of this engine read-only at the server (`default_transaction_read_only=on`).
    """
    return create_async_engine(
        url.get_secret_value(),
        connect_args=_connect_args(connect_timeout_seconds, statement_timeout_ms, read_only),
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_timeout=pool_timeout_seconds,
        pool_recycle=pool_recycle_seconds,
        pool_pre_ping=True,
        echo=echo,
    )


def create_unpooled_engine(
    url: SecretStr, *, connect_timeout_seconds: float = DEFAULT_CONNECT_TIMEOUT_SECONDS
) -> AsyncEngine:
    """Engine whose connections are really closed on release (never pooled).

    For session-scoped state that must die with the connection, e.g. advisory locks: even if an
    explicit unlock fails, closing the connection releases them. No statement timeout (a lock
    session is idle for the whole critical section) but TCP keepalives, so the server drops the
    session, and its locks, when the client disappears.
    """
    return create_async_engine(
        url.get_secret_value(),
        poolclass=NullPool,
        connect_args=_connect_args(connect_timeout_seconds, None),
    )


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Session factory: no autoflush surprises, objects stay usable after commit."""
    return async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
