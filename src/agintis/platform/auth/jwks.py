"""Signing-key sources for JWT verification (JWKS), async and cached.

`FileJwks` is static (loaded at startup). `HttpJwks` fetches a JWKS URL with an async client
(PyJWT's `PyJWKClient` is synchronous urllib and would block the event loop) and caches keys:
- a refresh happens when the cache is older than the TTL, or when a token names an UNKNOWN `kid`;
- every fetch attempt, successful or not, is rate-limited by a minimum interval, so random kids
  (or an identity provider that is down) cannot turn into a request flood;
- concurrent callers share one fetch (single flight);
- every fetch is bounded by a total deadline (httpx timeouts are per operation, so a trickling
  server could otherwise hold the lock), and so is the wait for the lock;
- a failed fetch after the TTL expired is `AuthUnavailableError` (503), never a silent 401, EXCEPT
  inside the optional stale grace: while the refresh FAILS (network, timeout, non-2xx, malformed)
  the cached keys keep being served until `ttl + grace`. A refresh that SUCCEEDS always replaces
  the keys (a kid missing from a good document is gone: stale keys are never used then).
"""

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx
import jwt

from agintis.platform.auth.errors import AuthConfigError, AuthUnavailableError

logger = logging.getLogger(__name__)

MAX_JWKS_BYTES = 1024 * 1024
MAX_KID_CHARS = 256
PUBLIC_KEY_TYPES = frozenset({"RSA", "EC", "OKP"})
# Asymmetric JWS algorithms only. No `none`, no HS*: a public key must never act as a secret.
ASYMMETRIC_ALGORITHMS = frozenset(
    {
        "RS256",
        "RS384",
        "RS512",
        "PS256",
        "PS384",
        "PS512",
        "ES256",
        "ES384",
        "ES512",
        "EdDSA",
    }
)
# Kid prefix of the keys minted by `cli.dev_token`: refused in a JWKS outside APP_ENV=local.
DEV_KID_PREFIX = "dev-"
DEFAULT_FETCH_TIMEOUT_SECONDS = 10.0
STALE_WARNING_INTERVAL_SECONDS = 60.0
MAX_LOGGED_KID_CHARS = 64
Clock = Callable[[], float]
JwksFetcher = Callable[[], Awaitable[Mapping[str, Any]]]


@dataclass(frozen=True, slots=True, repr=False)
class SigningKey:
    """A public verification key and the algorithm its JWKS entry pins (None = unpinned)."""

    key: Any
    algorithm: str | None

    def __repr__(self) -> str:
        return "SigningKey(...)"


class JwksSource(Protocol):
    async def get_key(self, kid: str) -> SigningKey | None:
        """The key for `kid` (None if unknown); `AuthUnavailableError` if keys are unobtainable."""
        ...

    async def aclose(self) -> None: ...


def parse_jwks(document: Mapping[str, Any]) -> dict[str, SigningKey]:
    """Usable signature keys of a JWKS document by `kid`.

    Entries without a `kid`, for encryption (`use` != sig), of an unsupported type/curve or whose
    declared `alg` is not an asymmetric JWS algorithm (`none`, HS*, unknown) are skipped; so is any
    entry PyJWT cannot parse (one odd entry must never take the whole key set down). A duplicate
    `kid` keeps the first one.
    """
    if not isinstance(document, Mapping):
        raise ValueError("JWKS document is not an object")
    entries = document.get("keys")
    if not isinstance(entries, list):
        raise ValueError("JWKS document has no 'keys' list")
    keys: dict[str, SigningKey] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        kid = entry.get("kid")
        if not isinstance(kid, str) or not kid or len(kid) > MAX_KID_CHARS or kid in keys:
            continue
        if entry.get("use", "sig") != "sig" or entry.get("kty") not in PUBLIC_KEY_TYPES:
            continue  # never a shared secret (`oct`) as a verification key
        declared = entry.get("alg")
        # `PyJWK.from_dict` raises NotImplementedError for `alg: none`: filter BEFORE parsing.
        if declared is not None and (
            not isinstance(declared, str) or declared not in ASYMMETRIC_ALGORITHMS
        ):
            logger.warning(
                "JWKS entry %.64r skipped: algorithm not allowed", kid[:MAX_LOGGED_KID_CHARS]
            )
            continue
        try:
            parsed = jwt.PyJWK.from_dict(entry)
        except Exception as exc:
            logger.warning(
                "JWKS entry %.64r skipped: %s", kid[:MAX_LOGGED_KID_CHARS], type(exc).__name__
            )
            continue
        keys[kid] = SigningKey(parsed.key, declared)
    return keys


class FileJwks:
    """A JWKS read once at construction (dev and tests; also air-gapped deployments)."""

    def __init__(self, path: Path) -> None:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            self._keys = parse_jwks(document)
        except (OSError, ValueError) as exc:
            raise AuthConfigError(f"Cannot load the JWKS file {path}: {exc}") from exc
        if not self._keys:
            raise AuthConfigError(f"The JWKS file {path} contains no usable signature key")

    @property
    def kids(self) -> frozenset[str]:
        return frozenset(self._keys)

    async def get_key(self, kid: str) -> SigningKey | None:
        return self._keys.get(kid)

    async def aclose(self) -> None:
        return None


class HttpxJwksFetcher:
    """GET of the JWKS URL: no redirects, bounded body, JSON object required."""

    def __init__(
        self,
        url: str,
        timeout_seconds: float,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._url = url
        self._client = httpx.AsyncClient(
            timeout=timeout_seconds, follow_redirects=False, transport=transport
        )

    async def __call__(self) -> Mapping[str, Any]:
        async with self._client.stream("GET", self._url) as response:
            response.raise_for_status()  # a 3xx (redirects are not followed) fails here too
            body = bytearray()
            async for chunk in response.aiter_bytes():
                body.extend(chunk)
                if len(body) > MAX_JWKS_BYTES:
                    raise ValueError("JWKS response too large")
        document = json.loads(body)
        if not isinstance(document, dict):
            raise ValueError("JWKS response is not a JSON object")
        return document

    async def aclose(self) -> None:
        await self._client.aclose()


class HttpJwks:
    def __init__(
        self,
        fetch: JwksFetcher,
        *,
        ttl_seconds: float,
        min_refresh_interval_seconds: float,
        fetch_timeout_seconds: float = DEFAULT_FETCH_TIMEOUT_SECONDS,
        stale_grace_seconds: float = 0.0,
        clock: Clock = time.monotonic,
        closer: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self._fetch = fetch
        self._ttl = ttl_seconds
        self._min_interval = min_refresh_interval_seconds
        self._fetch_timeout = fetch_timeout_seconds
        self._grace = stale_grace_seconds
        self._clock = clock
        self._closer = closer
        self._keys: dict[str, SigningKey] = {}
        self._fetched_at: float | None = None
        self._last_attempt: float | None = None
        self._last_stale_warning: float | None = None
        # Created lazily: bound to no loop at build time.
        self._lock: asyncio.Lock | None = None

    def _fresh(self, now: float) -> bool:
        return self._fetched_at is not None and now - self._fetched_at < self._ttl

    def _may_fetch(self, now: float) -> bool:
        return self._last_attempt is None or now - self._last_attempt >= self._min_interval

    def _serve_stale(self, now: float, kid: str) -> SigningKey | None:
        """Cached key while a FAILED refresh is inside the grace; 503 otherwise.

        An unknown kid is a 503 too: with the refresh failing we cannot tell a rotated key from
        a bogus one.
        """
        fetched_at = self._fetched_at
        if self._grace <= 0 or fetched_at is None or now - fetched_at >= self._ttl + self._grace:
            raise AuthUnavailableError
        key = self._keys.get(kid)
        if key is None:
            raise AuthUnavailableError
        last = self._last_stale_warning
        if last is None or now - last >= STALE_WARNING_INTERVAL_SECONDS:
            self._last_stale_warning = now
            logger.warning(
                "Serving STALE JWKS keys (refresh failing): age %.0fs, grace ends in %.0fs",
                now - fetched_at,
                fetched_at + self._ttl + self._grace - now,
            )
        return key

    async def _acquire(self) -> None:
        if self._lock is None:
            self._lock = asyncio.Lock()
        try:
            async with asyncio.timeout(self._fetch_timeout):
                await self._lock.acquire()
        except TimeoutError:
            raise AuthUnavailableError from None

    async def get_key(self, kid: str) -> SigningKey | None:
        now = self._clock()
        if self._fresh(now):
            key = self._keys.get(kid)
            if key is not None or not self._may_fetch(now):
                return key
        try:
            await self._acquire()
        except AuthUnavailableError:
            # The leader is slow: same as a failed refresh.
            if self._fresh(self._clock()):
                raise
            return self._serve_stale(self._clock(), kid)
        assert self._lock is not None
        try:
            now = self._clock()  # another caller may have refreshed while we waited
            if self._fresh(now):
                key = self._keys.get(kid)
                if key is not None or not self._may_fetch(now):
                    return key
                await self._refresh(now)  # unknown kid inside the TTL: a failure is a 503
            elif not self._may_fetch(now):
                return self._serve_stale(now, kid)  # expired, last attempt (failed) too recent
            else:
                try:
                    await self._refresh(now)
                except AuthUnavailableError:
                    return self._serve_stale(now, kid)
        finally:
            self._lock.release()
        return self._keys.get(kid)

    async def _refresh(self, now: float) -> None:
        self._last_attempt = now
        try:
            async with asyncio.timeout(self._fetch_timeout):
                document = await self._fetch()
            keys = parse_jwks(document)
        except Exception as exc:
            # Failure class only: URLs of the IdP are config, bodies may be anything.
            logger.warning("JWKS refresh failed: %s", type(exc).__name__)
            raise AuthUnavailableError from exc
        if not keys:
            # Treated like a malformed document (a proxy error page, a broken IdP): a failure.
            logger.warning("JWKS refresh returned no usable signature key")
            raise AuthUnavailableError
        self._keys = keys
        self._fetched_at = now
        self._last_stale_warning = None

    async def aclose(self) -> None:
        if self._closer is not None:
            await self._closer()
