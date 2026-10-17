"""JWT verification: signature, `exp`, `iss`, `aud` required, `nbf` checked when present.

Algorithm-confusion safe: the allowlist (asymmetric algorithms only, `none` and HS* never
allowed, also enforced at config validation) is checked against the UNVERIFIED header before any
key is looked up, and the verification key is the JWKS public key object, never a shared secret.

Time claims: `exp` is required and `exp`/`nbf`/`iat` must be real numbers (no strings, no
booleans, finite); the token lifetime (`exp - iat`, or `exp - now` without `iat`) is capped by
`max_token_lifetime_seconds`.

There is NO `typ` check: an OIDC ID token signed by the same issuer is distinguished from an
access token only by its audience. `AUTH_AUDIENCE` must therefore never be the OAuth client_id
(ID tokens carry it as `aud`), or ID tokens would be accepted as access tokens.
"""

import logging
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import jwt

from agintis.platform.auth.errors import InvalidTokenError
from agintis.platform.auth.jwks import ASYMMETRIC_ALGORITHMS, MAX_KID_CHARS, JwksSource

logger = logging.getLogger(__name__)

DEFAULT_ALGORITHMS = ("RS256", "ES256")
REQUIRED_CLAIMS = ["exp", "iss", "aud"]
DEFAULT_MAX_TOKEN_LIFETIME_SECONDS = 86400
TIME_CLAIMS = ("exp", "nbf", "iat")


def disallowed_algorithms(algorithms: Sequence[str]) -> list[str]:
    return [alg for alg in algorithms if alg not in ASYMMETRIC_ALGORITHMS]


@dataclass(frozen=True, slots=True)
class VerifiedToken:
    """Claims of a token whose signature and registered claims were verified."""

    claims: Mapping[str, Any] = field(repr=False)

    @property
    def subject(self) -> str | None:
        sub = self.claims.get("sub")
        return sub if isinstance(sub, str) else None


class JwtVerifier:
    def __init__(
        self,
        keys: JwksSource,
        *,
        issuer: str,
        audience: str,
        algorithms: Sequence[str] = DEFAULT_ALGORITHMS,
        leeway_seconds: int = 0,
        max_token_lifetime_seconds: int = DEFAULT_MAX_TOKEN_LIFETIME_SECONDS,
        now: Callable[[], float] = time.time,
    ) -> None:
        bad = disallowed_algorithms(algorithms)
        if bad or not algorithms:
            raise ValueError(f"Unsupported or empty JWT algorithm allowlist: {bad}")
        self._keys = keys
        self._issuer = issuer
        self._audience = audience
        self._algorithms = frozenset(algorithms)
        self._leeway = leeway_seconds
        self._max_lifetime = max_token_lifetime_seconds
        self._now = now

    async def verify(self, token: str) -> VerifiedToken:
        """Claims of a valid token. `InvalidTokenError` (401) for any defect;
        `AuthUnavailableError` (503) when the signing keys cannot be obtained."""
        try:
            header = jwt.get_unverified_header(token)
        except jwt.PyJWTError:
            raise InvalidTokenError("malformed token") from None
        alg = header.get("alg")
        if not isinstance(alg, str) or alg not in self._algorithms:
            raise InvalidTokenError("algorithm not allowed")
        kid = header.get("kid")
        if not isinstance(kid, str) or not kid or len(kid) > MAX_KID_CHARS:
            raise InvalidTokenError("missing kid")
        signing = await self._keys.get_key(kid)
        if signing is None:
            raise InvalidTokenError("unknown kid")
        if signing.algorithm is not None and signing.algorithm != alg:
            raise InvalidTokenError("algorithm differs from the key's")
        try:
            claims = jwt.decode(
                token,
                signing.key,
                algorithms=[alg],
                issuer=self._issuer,
                audience=self._audience,
                leeway=self._leeway,
                options={"require": REQUIRED_CLAIMS},
            )
        except jwt.ExpiredSignatureError:
            raise InvalidTokenError("expired") from None
        except jwt.PyJWTError as exc:
            raise InvalidTokenError(type(exc).__name__) from None
        except (TypeError, ValueError):
            # A key of the wrong type for the algorithm (EC key with RS256, ...).
            raise InvalidTokenError("key does not match the algorithm") from None
        self._check_time_claims(claims)
        return VerifiedToken(claims)

    def _check_time_claims(self, claims: Mapping[str, Any]) -> None:
        for name in TIME_CLAIMS:
            value = claims.get(name)
            if name in claims and (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
            ):
                raise InvalidTokenError(f"{name} is not a number")
        exp = claims["exp"]
        iat = claims.get("iat")
        now = self._now()
        if exp - (iat if iat is not None else now) > self._max_lifetime:
            raise InvalidTokenError("lifetime too long")
        if exp - now > self._max_lifetime + self._leeway:
            raise InvalidTokenError("expires too far ahead")  # also a forged-future `iat`

    async def aclose(self) -> None:
        await self._keys.aclose()
