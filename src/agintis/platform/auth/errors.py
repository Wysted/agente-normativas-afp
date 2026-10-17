"""Client-facing authentication errors (RFC 6750 challenges) and the boot-time config error.

Messages are fixed: they never carry the token, its claims or the reason a token was refused
(the reason goes to the log, as a short fixed string).
"""

from agintis.shared_kernel.errors import AppError, ErrorKind

WWW_AUTHENTICATE = "WWW-Authenticate"
RETRY_AFTER = "Retry-After"
BEARER_CHALLENGE = "Bearer"
INVALID_TOKEN_CHALLENGE = 'Bearer error="invalid_token"'
# Hint for clients when the identity provider cannot be reached (seconds).
UNAVAILABLE_RETRY_AFTER_SECONDS = 5


class AuthConfigError(ValueError):
    """The authentication configuration cannot be used (bad JWKS file, ...); fails the boot."""


class MissingCredentialsError(AppError):
    """No bearer credentials: the challenge carries no `error` (RFC 6750 section 3.1)."""

    kind = ErrorKind.UNAUTHORIZED
    code = "authentication_required"

    def __init__(self) -> None:
        super().__init__("Authentication is required", headers={WWW_AUTHENTICATE: BEARER_CHALLENGE})


class InvalidTokenError(AppError):
    """The token is malformed, unsigned, expired, or not meant for this API."""

    kind = ErrorKind.UNAUTHORIZED
    code = "invalid_token"

    def __init__(self, reason: str) -> None:
        super().__init__(
            "The access token is invalid or has expired",
            headers={WWW_AUTHENTICATE: INVALID_TOKEN_CHALLENGE},
        )
        # Fixed short string for the log only; never rendered to the client.
        self.reason = reason


class TenantClaimError(AppError):
    """A valid token whose tenant claim is absent or not a valid tenant id."""

    kind = ErrorKind.FORBIDDEN
    code = "tenant_claim_invalid"


class AuthUnavailableError(AppError):
    """The signing keys cannot be obtained (identity provider unreachable): retryable."""

    kind = ErrorKind.UNAVAILABLE
    code = "auth_unavailable"

    def __init__(self) -> None:
        super().__init__(
            "Authentication is temporarily unavailable",
            headers={RETRY_AFTER: str(UNAVAILABLE_RETRY_AFTER_SECONDS)},
        )
