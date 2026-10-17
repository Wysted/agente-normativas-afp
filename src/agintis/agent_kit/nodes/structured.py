"""Structured-output calls for the HINT nodes (analyzer, filter extractor).

Only PARSE failures degrade; provider and network errors always propagate (the run's error
classifier turns them into `llm_rate_limited`, `llm_unavailable`, ...). A parse failure is the
model's output not fitting the schema, per provider:
- langchain-openai json_schema: pydantic `ValidationError`, `OpenAIRefusalError`, and a plain
  `ValueError("Structured Output response does not have a 'parsed' field ...")`; the openai SDK's
  own `LengthFinishReasonError` / `ContentFilterFinishReasonError` (truncated or filtered output,
  no HTTP status);
- parser-based providers (tool calling / JSON mode): `OutputParserException`.
The classes are matched by name so no provider package is imported (they are optional extras).

`invoke_structured` binds with `include_raw=True`: when parsing fails the raw model text is still
there, so a node can salvage what it can from it (the extractor validates field by field).
Logs carry the failure class only, never model output or user text.
"""

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable, RunnableConfig
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

_PARSE_FAILURE_CLASS_NAMES = frozenset(
    {"OpenAIRefusalError", "LengthFinishReasonError", "ContentFilterFinishReasonError"}
)
_NO_PARSED_PREFIX = "Structured Output response does not have"


def is_structured_output_failure(exc: BaseException) -> bool:
    """True only for failures to PARSE the model output; never for provider/network errors."""
    if isinstance(exc, ValidationError | OutputParserException):
        return True
    if any(cls.__name__ in _PARSE_FAILURE_CLASS_NAMES for cls in type(exc).__mro__):
        return True
    return type(exc) is ValueError and str(exc).startswith(_NO_PARSED_PREFIX)


@dataclass(frozen=True, slots=True)
class StructuredOutcome:
    """`parsed` is the schema instance, or None when the output did not parse; `raw_values` is
    then the JSON object found in the raw output (None if there is none)."""

    parsed: BaseModel | None = None
    raw_values: dict[str, Any] | None = None


def bind_structured(model: BaseChatModel, schema: type[BaseModel]) -> Runnable[Any, Any]:
    """Bind `model` (never mutated: a copy is bound) for structured output with `include_raw`.

    Structured calls are not streamed: LangGraph's `messages` mode would otherwise force
    streaming, and langchain-openai then goes through the SDK's `beta.chat.completions.stream`,
    whose `ContentDoneEvent` carries the parsed instance in a field typed `None`, so dumping it
    raises a pydantic `UserWarning` ('parsed'). A non-streamed call still arrives in `messages`
    mode as one full `AIMessage` (usage included), so the run keeps aggregating its tokens.
    """
    return model.model_copy(update={"disable_streaming": True}).with_structured_output(
        schema, include_raw=True
    )


def _raw_object(raw: object) -> dict[str, Any] | None:
    """The JSON object in a raw model message: tool-call args, else the text."""
    if not isinstance(raw, BaseMessage):
        return None
    for call in getattr(raw, "tool_calls", None) or ():
        args = call.get("args")
        if isinstance(args, dict):
            return args
    try:
        value = json.loads(raw.text)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


async def invoke_structured(
    runnable: Runnable[Any, Any],
    messages: Sequence[BaseMessage],
    config: RunnableConfig | None = None,
) -> StructuredOutcome:
    """Run a bound structured call; `config` (callbacks, tags, metadata) is passed to the run."""
    try:
        result = await runnable.ainvoke(list(messages), config)
    except Exception as exc:
        if not is_structured_output_failure(exc):
            raise
        logger.debug("Structured output failed to parse (%s)", type(exc).__name__)
        return StructuredOutcome()
    if isinstance(result, BaseModel):  # providers/fakes that ignore include_raw
        return StructuredOutcome(parsed=result)
    if not isinstance(result, dict):
        return StructuredOutcome()
    parsed = result.get("parsed")
    if isinstance(parsed, BaseModel):
        return StructuredOutcome(parsed=parsed)
    error = result.get("parsing_error")
    logger.debug(
        "Structured output failed to parse (%s)",
        type(error).__name__ if error is not None else "no parsed value",
    )
    return StructuredOutcome(raw_values=_raw_object(result.get("raw")))
