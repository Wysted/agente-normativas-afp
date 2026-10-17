"""Base error types shared across modules."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar


class DomainError(Exception):
    """Base class for all business-rule violations."""


class NotFoundError(DomainError):
    """Raised when a looked-up entity or registry entry does not exist."""


class ErrorKind(StrEnum):
    """Transport-neutral category of an `AppError`; the HTTP edge maps it to a status."""

    NOT_FOUND = "not_found"
    INVALID = "invalid"
    BAD_REQUEST = "bad_request"
    CONFLICT = "conflict"
    UNAVAILABLE = "unavailable"
    UNAUTHORIZED = "unauthorized"
    FORBIDDEN = "forbidden"


class AppError(Exception):
    """An error whose message is safe to show to the client, with a stable machine code.

    Subclasses set `kind` and `code` as class attributes; the HTTP layer renders them without
    knowing the concrete type (so platform code never imports business modules).
    """

    kind: ClassVar[ErrorKind] = ErrorKind.INVALID
    code: str = "invalid_request"

    def __init__(
        self, message: str, *, code: str | None = None, headers: Mapping[str, str] | None = None
    ) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code
        # Extra response headers the transport should send (`WWW-Authenticate`, `Retry-After`).
        self.headers: Mapping[str, str] = dict(headers) if headers else {}

    @property
    def message(self) -> str:
        return str(self)


class InvalidRequestError(AppError):
    """The request is well-formed but violates an input rule."""

    kind = ErrorKind.INVALID
    code = "invalid_request"


@dataclass(frozen=True, slots=True)
class FailureInfo:
    """Client-safe description of an unexpected failure (never carries exception text)."""

    code: str
    message: str
    retryable: bool = False
