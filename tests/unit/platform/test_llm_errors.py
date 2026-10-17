import pytest

from agintis.platform.llm import LlmError, LlmNotConfiguredError, ProviderErrorClassifier
from agintis.shared_kernel.errors import InvalidRequestError

classifier = ProviderErrorClassifier()


def _exc(name: str, module: str, *bases: type[Exception], status: int | None = None) -> Exception:
    """An exception of a class named `name` that claims to come from package `module`."""
    cls = type(name, bases or (Exception,), {"__module__": module})
    exc: Exception = cls("raw provider text sk-secret")
    if status is not None:
        setattr(exc, "status_code", status)  # noqa: B010
    return exc


@pytest.mark.parametrize(
    ("status", "code", "retryable"),
    [
        (401, "llm_auth", False),
        (403, "llm_auth", False),
        (429, "llm_rate_limited", True),
        (500, "llm_unavailable", True),
        (503, "llm_unavailable", True),
        (400, "llm_error", False),
    ],
)
def test_provider_status_code_mapping(status: int, code: str, retryable: bool) -> None:
    info = classifier.classify(_exc("APIStatusError", "openai", status=status))

    assert (info.code, info.retryable) == (code, retryable)
    assert "sk-secret" not in info.message


@pytest.mark.parametrize(
    "exc",
    [
        _exc("APITimeoutError", "openai._exceptions"),
        _exc("ReadTimeout", "httpx"),
        _exc("TimeoutException", "anthropic"),
    ],
)
def test_provider_timeouts(exc: Exception) -> None:
    info = classifier.classify(exc)

    assert (info.code, info.retryable) == ("llm_timeout", True)


def test_timeout_that_is_also_a_connection_error_is_a_timeout() -> None:
    connection = type("APIConnectionError", (Exception,), {"__module__": "openai"})
    info = classifier.classify(_exc("APITimeoutError", "openai", connection))

    assert info.code == "llm_timeout"


@pytest.mark.parametrize(
    "exc",
    [_exc("APIConnectionError", "openai"), _exc("ConnectError", "httpx.sub")],
)
def test_connection_errors_are_unavailable_and_retryable(exc: Exception) -> None:
    info = classifier.classify(exc)

    assert (info.code, info.retryable) == ("llm_unavailable", True)


def test_status_code_on_a_non_provider_exception_is_internal() -> None:
    info = classifier.classify(_exc("HTTPError", "requests", status=401))

    assert info.code == "internal"


def test_builtin_timeout_is_internal() -> None:
    assert classifier.classify(TimeoutError()).code == "internal"


def test_llm_errors() -> None:
    assert classifier.classify(LlmNotConfiguredError("x")).code == "llm_not_configured"
    assert classifier.classify(LlmError("boom")).code == "llm_error"


def test_app_error_keeps_its_safe_code_and_message() -> None:
    info = classifier.classify(InvalidRequestError("bad input"))

    assert (info.code, info.message) == ("invalid_request", "bad input")


def test_unknown_is_internal_without_leaking_the_message() -> None:
    info = classifier.classify(RuntimeError("password=hunter2"))

    assert info.code == "internal"
    assert not info.retryable
    assert "hunter2" not in info.message
