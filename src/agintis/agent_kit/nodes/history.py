"""Bounded views over the conversation, shared by the nodes that send it to a model."""

from collections.abc import Sequence

from langchain_core.messages import AnyMessage, HumanMessage

# Messages of the conversation a node sends to a model (thread history is unbounded in state).
MAX_HISTORY_MESSAGES = 10


def recent_history(
    messages: Sequence[AnyMessage], limit: int = MAX_HISTORY_MESSAGES
) -> list[AnyMessage]:
    """The last `limit` messages, starting at a human one (providers reject a leading reply).

    If the window holds no human message at all, the last human message of the whole
    conversation is returned alone: the question must never be lost to the window.
    """
    window = list(messages[-limit:]) if limit > 0 else []
    for index, message in enumerate(window):
        if isinstance(message, HumanMessage):
            return window[index:]
    last = last_human(messages)
    return [last] if last is not None else []


def last_human(messages: Sequence[AnyMessage]) -> HumanMessage | None:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message
    return None


def last_human_text(messages: Sequence[AnyMessage]) -> str:
    message = last_human(messages)
    return message.text.strip() if message is not None else ""
