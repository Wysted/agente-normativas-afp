"""LangGraph's `AsyncPostgresSaver` on its own psycopg pool.

Why a separate pool (and not the SQLAlchemy engine): the saver speaks raw psycopg and needs
`autocommit=True`, `row_factory=dict_row` and `prepare_threshold=0` (no prepared statements, so it
also survives a statement-pooling proxy; PgBouncer TRANSACTION mode is still unsupported). Its
writes happen outside the app's UnitOfWork: checkpoints and the `threads` table are NOT atomic;
the thread status is reconciled when the run ends.

The pool is created closed (no I/O: `Container.build` is synchronous and network-free) and opened
by `Container.start()` only when the checkpointer is enabled. The saver's tables are NOT created
at boot: `setup()` (the `checkpointer_setup` CLI) creates/migrates them and `verify()` is the boot
check that refuses to start without them.
"""

import asyncio
import hashlib
import logging
import math
from typing import Any

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg import AsyncConnection, Capabilities
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool
from pydantic import SecretStr

logger = logging.getLogger(__name__)

# Checkpointer schema versions are the indexes of `AsyncPostgresSaver.MIGRATIONS`.
EXPECTED_SCHEMA_VERSION = len(AsyncPostgresSaver.MIGRATIONS) - 1
DEFAULT_VERIFY_TIMEOUT_SECONDS = 15.0
# Serializes concurrent `setup()` runs (two deploy jobs): the saver's migrations are not
# concurrency-safe. Session-level advisory lock on a dedicated connection.
SETUP_LOCK_KEY = int.from_bytes(hashlib.sha256(b"agintis.checkpointer.setup").digest()[:8]) >> 1
SETUP_HINT = "run `python -m agintis.cli.checkpointer_setup`"

type Pool = AsyncConnectionPool[AsyncConnection[DictRow]]


class CheckpointerSetupError(RuntimeError):
    """The checkpointer database is unreachable or its tables are missing / outdated."""


class PooledSaver(AsyncPostgresSaver):
    """`AsyncPostgresSaver` that can be built outside a running event loop.

    The upstream constructor stores `asyncio.get_running_loop()`, which only its BLOCKING sync
    wrappers (`put`, `get_tuple`, ...) use to hop back onto the loop. Container.build is
    synchronous, so the loop is bound in `bind_loop` (called when the pool opens) instead. Every
    other attribute is what upstream sets for a pool without a pipeline; the integration tests
    run a real graph through it, so an upstream change of `__init__` fails there.
    """

    def __init__(self, pool: Pool) -> None:
        super(AsyncPostgresSaver, self).__init__()  # BasePostgresSaver: serde, no loop needed
        self.conn = pool
        self.pipe = None
        self.lock = asyncio.Lock()
        self.supports_pipeline = Capabilities().has_pipeline()

    def bind_loop(self) -> None:
        self.loop = asyncio.get_running_loop()


class Checkpointer:
    """Pool + saver + lifecycle. `saver` is what graphs are compiled with."""

    def __init__(
        self,
        dsn: SecretStr,
        *,
        min_size: int,
        max_size: int,
        acquire_timeout_seconds: float,
        connect_timeout_seconds: float,
        pool: Pool | None = None,
    ) -> None:
        """`pool` replaces the real one (a test seam for the boot check; never set in the app)."""
        self._dsn = dsn
        self._connect_timeout_seconds = connect_timeout_seconds
        connection_kwargs: dict[str, Any] = {
            "autocommit": True,
            "row_factory": dict_row,
            "prepare_threshold": 0,
            "connect_timeout": max(1, math.ceil(connect_timeout_seconds)),
        }
        self._pool: Pool = pool or AsyncConnectionPool(
            dsn.get_secret_value(),
            kwargs=connection_kwargs,
            min_size=min_size,
            max_size=max_size,
            timeout=acquire_timeout_seconds,
            # Drops connections the server or a proxy closed while idle, before handing them out.
            check=AsyncConnectionPool.check_connection,
            name="agintis-checkpointer",
            open=False,
        )
        self.saver = PooledSaver(self._pool)

    async def open(self, *, wait: bool = False, wait_timeout_seconds: float = 30.0) -> None:
        """Start the pool. `wait=False` returns at once (connections come up in the background)."""
        await self._pool.open(wait=wait, timeout=wait_timeout_seconds)
        self.saver.bind_loop()

    async def aclose(self) -> None:
        await self._pool.close()

    async def setup(self) -> None:
        """Create/migrate the checkpoint tables (idempotent; one run at a time). Needs an open
        pool; the lock lives on its own connection so a tiny pool cannot deadlock on it."""
        async with await AsyncConnection.connect(
            self._dsn.get_secret_value(),
            autocommit=True,
            connect_timeout=max(1, math.ceil(self._connect_timeout_seconds)),
        ) as lock_connection:
            await lock_connection.execute("SELECT pg_advisory_lock(%s)", (SETUP_LOCK_KEY,))
            await self.saver.setup()  # the lock goes away with the connection

    async def verify(self, timeout_seconds: float = DEFAULT_VERIFY_TIMEOUT_SECONDS) -> None:
        """Boot check: reachable, and every checkpointer migration applied."""
        try:
            async with asyncio.timeout(timeout_seconds), self._pool.connection() as conn:
                table = await (
                    await conn.execute("SELECT to_regclass('checkpoint_migrations') AS name")
                ).fetchone()
                version: int | None = None
                if table is not None and table["name"] is not None:
                    row = await (
                        await conn.execute("SELECT max(v) AS version FROM checkpoint_migrations")
                    ).fetchone()
                    version = None if row is None else row["version"]
        except Exception as exc:
            raise CheckpointerSetupError(
                f"Checkpointer database is not reachable: {exc.__class__.__name__}"
            ) from exc
        if version is None:
            raise CheckpointerSetupError(f"The checkpoint tables do not exist: {SETUP_HINT}")
        if version > EXPECTED_SCHEMA_VERSION:
            raise CheckpointerSetupError(
                f"The checkpoint tables are at version {version}, NEWER than this build "
                f"expects ({EXPECTED_SCHEMA_VERSION}): deploy the matching (newer) build"
            )
        if version < EXPECTED_SCHEMA_VERSION:
            raise CheckpointerSetupError(
                f"The checkpoint tables are at version {version} but this build expects "
                f"{EXPECTED_SCHEMA_VERSION}: {SETUP_HINT}"
            )
        logger.info("Checkpointer ready (schema version %s)", version)

    async def delete_thread(self, thread_id: str) -> None:
        await self.saver.adelete_thread(thread_id)
