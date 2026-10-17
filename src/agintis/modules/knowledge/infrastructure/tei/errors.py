"""Errors raised by the TEI adapters."""


class TeiError(RuntimeError):
    """The TEI service failed or answered something unexpected."""


class TeiUnavailableError(TeiError):
    """The service stayed unreachable or overloaded (429/5xx/network) after all retries."""
