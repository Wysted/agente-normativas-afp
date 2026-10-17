"""Authentication: JWT verification against a JWKS (file or URL)."""

from agintis.platform.auth.errors import (
    AuthConfigError,
    AuthUnavailableError,
    InvalidTokenError,
    MissingCredentialsError,
    TenantClaimError,
)
from agintis.platform.auth.jwks import (
    ASYMMETRIC_ALGORITHMS,
    DEV_KID_PREFIX,
    FileJwks,
    HttpJwks,
    HttpxJwksFetcher,
    JwksSource,
    SigningKey,
    parse_jwks,
)
from agintis.platform.auth.verifier import (
    DEFAULT_ALGORITHMS,
    DEFAULT_MAX_TOKEN_LIFETIME_SECONDS,
    JwtVerifier,
    VerifiedToken,
    disallowed_algorithms,
)

__all__ = [
    "ASYMMETRIC_ALGORITHMS",
    "DEFAULT_ALGORITHMS",
    "DEFAULT_MAX_TOKEN_LIFETIME_SECONDS",
    "DEV_KID_PREFIX",
    "AuthConfigError",
    "AuthUnavailableError",
    "FileJwks",
    "HttpJwks",
    "HttpxJwksFetcher",
    "InvalidTokenError",
    "JwksSource",
    "JwtVerifier",
    "MissingCredentialsError",
    "SigningKey",
    "TenantClaimError",
    "VerifiedToken",
    "disallowed_algorithms",
    "parse_jwks",
]
