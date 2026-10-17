"""Finish reason extraction and normalization, in one place for every runner."""

from langchain_core.messages import AIMessage

# Provider-specific keys of `response_metadata` that carry why generation stopped.
_FINISH_REASON_KEYS = ("finish_reason", "stop_reason", "done_reason")

# Provider vocabularies mapped to the one the API exposes; unknown values pass through.
_NORMALIZED = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
}


def finish_reason_of(message: AIMessage) -> str | None:
    """The normalized finish reason a message reports, if any."""
    for key in _FINISH_REASON_KEYS:
        value = message.response_metadata.get(key)
        if isinstance(value, str) and value:
            return _NORMALIZED.get(value, value)
    return None
