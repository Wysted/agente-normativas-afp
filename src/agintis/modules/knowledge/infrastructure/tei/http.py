"""Small sync + async JSON-over-HTTP client for Text Embeddings Inference.

Failure policy
* connect/write/pool timeouts are short fixed constants; the READ timeout is configurable.
* Transient failures (429/5xx, connect errors, pool/write timeouts) are retried with capped
  exponential backoff and jitter; `Retry-After` is honored (capped).
* A READ timeout is NOT retried: the server accepted the request and is just slow, so a retry
  would double its load for the same result.
* Every call has a total deadline (`asyncio.timeout` in async; checked between attempts in
  sync) that bounds retries and sleeps together.
* Error messages never include the service URL (they may reach end users); the URL is only
  logged.
"""

import asyncio
import logging
import random
import threading
import time
from collections.abc import Awaitable, Callable
from typing import Any, Final

import httpx

from agintis.modules.knowledge.infrastructure.tei.errors import TeiError, TeiUnavailableError

# Total attempts = 1 + retries. Backoff doubles from `backoff_seconds`, capped.
INGEST_MAX_RETRIES: Final = 3
QUERY_MAX_RETRIES: Final = 1
DEFAULT_BACKOFF_SECONDS: Final = 0.5
MAX_BACKOFF_SECONDS: Final = 8.0
CONNECT_TIMEOUT_SECONDS: Final = 5.0
WRITE_TIMEOUT_SECONDS: Final = 10.0
POOL_TIMEOUT_SECONDS: Final = 5.0
JITTER_FLOOR: Final = 0.5
RETRYABLE_STATUS: Final = frozenset({429, 500, 502, 503, 504})
ERROR_BODY_PREVIEW_CHARS: Final = 300

logger = logging.getLogger(__name__)

type AsyncSleep = Callable[[float], Awaitable[None]]
type SyncSleep = Callable[[float], None]
type Jitter = Callable[[], float]


def default_jitter() -> float:
    """Multiplier in [JITTER_FLOOR, 1.0] applied to the computed backoff."""
    return random.uniform(JITTER_FLOOR, 1.0)


class _Retry(Exception):
    def __init__(self, reason: str, response: httpx.Response | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.response = response


class TeiHttp:
    """Owns lazily created httpx clients. Call `aclose()` / `close()` when done.

    `transport` / `async_transport` / `sleep` / `sync_sleep` / `jitter` exist for tests. Async
    clients are bound to the running loop, so use one instance per loop. The lazy sync client
    is created under a lock (the sync API may be used from several threads).
    """

    def __init__(
        self,
        base_url: str,
        timeout_seconds: float,
        *,
        deadline_seconds: float,
        max_retries: int = INGEST_MAX_RETRIES,
        backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
        api_key: str | None = None,
        transport: httpx.BaseTransport | None = None,
        async_transport: httpx.AsyncBaseTransport | None = None,
        sleep: AsyncSleep = asyncio.sleep,
        sync_sleep: SyncSleep = time.sleep,
        jitter: Jitter = default_jitter,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must not be negative")
        if timeout_seconds <= 0 or deadline_seconds <= 0:
            raise ValueError("timeout_seconds and deadline_seconds must be greater than 0")
        self._base_url = base_url.rstrip("/")
        self._timeout = httpx.Timeout(
            connect=CONNECT_TIMEOUT_SECONDS,
            read=timeout_seconds,
            write=WRITE_TIMEOUT_SECONDS,
            pool=POOL_TIMEOUT_SECONDS,
        )
        self._deadline = deadline_seconds
        self._max_retries = max_retries
        self._backoff = backoff_seconds
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._transport = transport
        self._async_transport = async_transport
        self._sleep = sleep
        self._sync_sleep = sync_sleep
        self._jitter = jitter
        self._sync_client: httpx.Client | None = None
        self._async_client: httpx.AsyncClient | None = None
        self._client_lock = threading.Lock()

    @property
    def max_retries(self) -> int:
        return self._max_retries

    # -- policy ---------------------------------------------------------------------------

    def _delay(self, attempt: int, response: httpx.Response | None) -> float:
        retry_after = response.headers.get("retry-after") if response is not None else None
        if retry_after is not None and retry_after.isdigit():
            return min(float(retry_after), MAX_BACKOFF_SECONDS)
        return min(float(self._backoff * 2**attempt), MAX_BACKOFF_SECONDS) * self._jitter()

    @staticmethod
    def _interpret(path: str, response: httpx.Response) -> Any:
        """Return the JSON body, raise `_Retry` for transient statuses, `TeiError` otherwise."""
        if response.is_success:
            try:
                return response.json()
            except ValueError:
                raise TeiError(f"TEI {path} returned a body that is not JSON") from None
        if response.status_code in RETRYABLE_STATUS:
            raise _Retry(f"HTTP {response.status_code}", response)
        body = response.text[:ERROR_BODY_PREVIEW_CHARS]
        raise TeiError(f"TEI {path} failed with HTTP {response.status_code}: {body}")

    def _unavailable(self, path: str, cause: str, attempts: int) -> TeiUnavailableError:
        return TeiUnavailableError(
            f"TEI unavailable for {path} after {attempts} attempt(s): {cause}"
        )

    def _read_timeout(self, path: str) -> TeiUnavailableError:
        logger.warning("TEI %s read timeout at %s (not retried)", path, self._base_url)
        return TeiUnavailableError(
            f"TEI {path} timed out waiting for the response (read timeout, not retried)"
        )

    def _deadline_exceeded(self, path: str) -> TeiUnavailableError:
        return TeiUnavailableError(f"TEI {path} exceeded its {self._deadline:g}s deadline")

    # -- clients --------------------------------------------------------------------------

    def _get_sync_client(self) -> httpx.Client:
        with self._client_lock:
            if self._sync_client is None:
                self._sync_client = httpx.Client(
                    base_url=self._base_url,
                    timeout=self._timeout,
                    headers=self._headers,
                    transport=self._transport,
                )
            return self._sync_client

    def _get_async_client(self) -> httpx.AsyncClient:
        # No await between check and set: safe within one event loop.
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout,
                headers=self._headers,
                transport=self._async_transport,
            )
        return self._async_client

    # -- sync -----------------------------------------------------------------------------

    def post(self, path: str, payload: dict[str, Any], *, max_retries: int | None = None) -> Any:
        retries = self._max_retries if max_retries is None else max_retries
        client = self._get_sync_client()
        deadline_at = time.monotonic() + self._deadline
        reason = "no attempt made"
        for attempt in range(retries + 1):
            response: httpx.Response | None = None
            try:
                return self._interpret(path, client.post(path, json=payload))
            except httpx.ReadTimeout:
                raise self._read_timeout(path) from None
            except httpx.TransportError as exc:
                reason = f"{type(exc).__name__}: {exc}"
            except _Retry as retry:
                reason, response = retry.reason, retry.response
            if attempt < retries:
                delay = self._delay(attempt, response)
                if time.monotonic() + delay >= deadline_at:
                    raise self._deadline_exceeded(path)
                logger.warning("TEI %s attempt %d failed (%s); retrying", path, attempt + 1, reason)
                self._sync_sleep(delay)
        raise self._unavailable(path, reason, retries + 1)

    # -- async ----------------------------------------------------------------------------

    async def apost(
        self, path: str, payload: dict[str, Any], *, max_retries: int | None = None
    ) -> Any:
        return await self._arequest("POST", path, payload, max_retries)

    async def aget(self, path: str) -> Any:
        """Single attempt (no retries): used for best-effort probes such as `/info`."""
        return await self._arequest("GET", path, None, 0)

    async def _arequest(
        self, method: str, path: str, payload: dict[str, Any] | None, max_retries: int | None
    ) -> Any:
        retries = self._max_retries if max_retries is None else max_retries
        try:
            async with asyncio.timeout(self._deadline):
                return await self._attempts(method, path, payload, retries)
        except TimeoutError:
            raise self._deadline_exceeded(path) from None

    async def _attempts(
        self, method: str, path: str, payload: dict[str, Any] | None, retries: int
    ) -> Any:
        client = self._get_async_client()
        reason = "no attempt made"
        for attempt in range(retries + 1):
            response: httpx.Response | None = None
            try:
                sent = await client.request(method, path, json=payload)
                return self._interpret(path, sent)
            except httpx.ReadTimeout:
                raise self._read_timeout(path) from None
            except httpx.TransportError as exc:
                reason = f"{type(exc).__name__}: {exc}"
            except _Retry as retry:
                reason, response = retry.reason, retry.response
            if attempt < retries:
                logger.warning("TEI %s attempt %d failed (%s); retrying", path, attempt + 1, reason)
                await self._sleep(self._delay(attempt, response))
        raise self._unavailable(path, reason, retries + 1)

    # -- lifecycle ------------------------------------------------------------------------

    def close(self) -> None:
        with self._client_lock:
            client, self._sync_client = self._sync_client, None
        if client is not None:
            client.close()

    async def aclose(self) -> None:
        client, self._async_client = self._async_client, None
        if client is not None:
            await client.aclose()
