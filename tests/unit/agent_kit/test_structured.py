"""Which failures of a structured-output call degrade and which propagate."""

import httpx
import httpx2
import openai
import pytest
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, ValidationError

from agintis.agent_kit.nodes import is_structured_output_failure


class _Schema(BaseModel):
    n: int


def _validation_error() -> ValidationError:
    try:
        _Schema.model_validate({"n": "x"})
    except ValidationError as exc:
        return exc
    raise AssertionError("unreachable")


def _named(name: str) -> Exception:
    error: Exception = type(name, (Exception,), {})("refused")
    return error


@pytest.mark.parametrize(
    "error",
    [
        _validation_error(),
        OutputParserException("bad json"),
        _named("OpenAIRefusalError"),  # langchain-openai refusal
        _named("LengthFinishReasonError"),  # openai SDK: truncated structured output
        _named("ContentFilterFinishReasonError"),
        ValueError("Structured Output response does not have a 'parsed' field nor a 'refusal'"),
    ],
)
def test_parse_failures(error: Exception) -> None:
    assert is_structured_output_failure(error)


@pytest.mark.parametrize(
    "error",
    [
        ValueError("something else"),
        RuntimeError("boom"),
        TimeoutError(),
        httpx.ConnectError("down"),
        openai.RateLimitError(
            "429",
            response=httpx2.Response(429, request=httpx2.Request("POST", "http://x")),
            body=None,
        ),
        openai.APIConnectionError(request=httpx2.Request("POST", "http://x")),
    ],
)
def test_provider_and_other_errors_are_not_parse_failures(error: Exception) -> None:
    assert not is_structured_output_failure(error)
