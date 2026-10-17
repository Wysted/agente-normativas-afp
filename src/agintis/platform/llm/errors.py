"""Translate provider/runtime exceptions into client-safe failures.

Provider SDK errors are recognised by duck typing (`status_code`, class names) so this module
imports no provider package, and only for exceptions whose class comes from a provider/HTTP
package (`_PROVIDER_ROOTS`): any other exception is `internal`, even with a `status_code`.
Messages are fixed strings per code: exception text may contain request details and never
reaches the client.
"""

from http import HTTPStatus

from agintis.platform.llm.models import LlmError, LlmNotConfiguredError
from agintis.shared_kernel.errors import AppError, FailureInfo

_PROVIDER_ROOTS = frozenset(
    {
        "openai",
        "anthropic",
        "ollama",
        "httpx",
        "langchain_openai",
        "langchain_anthropic",
        "langchain_ollama",
    }
)
_SERVER_ERRORS = range(HTTPStatus.INTERNAL_SERVER_ERROR, HTTPStatus.INTERNAL_SERVER_ERROR + 100)
_TIMEOUT_CLASS_NAMES = frozenset(
    {
        "APITimeoutError",
        "TimeoutException",
        "ReadTimeout",
        "ConnectTimeout",
        "WriteTimeout",
        "PoolTimeout",
    }
)
_CONNECTION_CLASS_NAMES = frozenset({"APIConnectionError", "ConnectError", "NetworkError"})

_AUTH = FailureInfo("llm_auth", "The language model rejected the service credentials.", False)
_RATE_LIMITED = FailureInfo(
    "llm_rate_limited", "The language model is rate limiting requests; retry shortly.", True
)
_UNAVAILABLE = FailureInfo(
    "llm_unavailable", "The language model service is temporarily unavailable.", True
)
_TIMEOUT = FailureInfo("llm_timeout", "The language model took too long to answer.", True)
_NOT_CONFIGURED = FailureInfo("llm_not_configured", "The language model is not configured.", False)
_LLM_ERROR = FailureInfo("llm_error", "The language model request failed.", False)
_INTERNAL = FailureInfo("internal", "Internal error while running the agent.", False)


def _is_provider_error(exc: BaseException) -> bool:
    return any(cls.__module__.partition(".")[0] in _PROVIDER_ROOTS for cls in type(exc).__mro__)


def _has_class_named(exc: BaseException, names: frozenset[str]) -> bool:
    return any(cls.__name__ in names for cls in type(exc).__mro__)


def _status_code(exc: BaseException) -> int | None:
    status = getattr(exc, "status_code", None)
    return status if isinstance(status, int) else None


class ProviderErrorClassifier:
    """Implements `ErrorClassifier` for chat-model providers and the project's own errors."""

    def classify(self, exc: BaseException) -> FailureInfo:
        if isinstance(exc, AppError):
            return FailureInfo(exc.code, exc.message, False)
        if isinstance(exc, LlmNotConfiguredError):
            return _NOT_CONFIGURED
        if isinstance(exc, LlmError):
            return _LLM_ERROR
        if not _is_provider_error(exc):
            return _INTERNAL
        status = _status_code(exc)
        if status in (HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN):
            return _AUTH
        if status == HTTPStatus.TOO_MANY_REQUESTS:
            return _RATE_LIMITED
        if status in _SERVER_ERRORS:
            return _UNAVAILABLE
        # Timeouts first: openai's APITimeoutError is also an APIConnectionError.
        if _has_class_named(exc, _TIMEOUT_CLASS_NAMES):
            return _TIMEOUT
        if _has_class_named(exc, _CONNECTION_CLASS_NAMES):
            return _UNAVAILABLE
        return _LLM_ERROR
