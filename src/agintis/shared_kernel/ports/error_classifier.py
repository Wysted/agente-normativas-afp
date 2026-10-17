from typing import Protocol

from agintis.shared_kernel.errors import FailureInfo


class ErrorClassifier(Protocol):
    """Turns an unexpected exception into a client-safe `FailureInfo` (no exception text)."""

    def classify(self, exc: BaseException) -> FailureInfo: ...
