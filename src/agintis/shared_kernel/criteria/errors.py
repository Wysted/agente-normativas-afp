"""Errors raised by Criteria helpers and backend converters."""

from agintis.shared_kernel.errors import DomainError


class UnknownFieldError(DomainError):
    """A filter or order references a field the repository does not declare as filterable."""


class UnsupportedFilterError(DomainError):
    """A converter cannot translate a filter faithfully; raise instead of approximating."""
