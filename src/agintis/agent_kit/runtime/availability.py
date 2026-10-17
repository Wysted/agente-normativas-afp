"""Pre-stream availability checks that agents hand to `LangGraphRunner`."""

from collections.abc import Callable

from agintis.modules.agents_catalog.api import AgentUnavailableError
from agintis.platform.llm import ChatModels, ChatRole, LlmNotConfiguredError


def llm_availability(chat_models: ChatModels, *roles: ChatRole) -> Callable[[], None]:
    """A check that reports a disabled or misconfigured LLM (503) before a run streams.

    `ChatModels.get()` builds and caches the model without network access, so the check is
    cheap; every role the agent will call is tried.
    """

    def ensure_available() -> None:
        for role in roles:
            try:
                chat_models.get(role)
            except LlmNotConfiguredError as exc:
                raise AgentUnavailableError(
                    "The language model is not configured", code="llm_not_configured"
                ) from exc
            except Exception as exc:
                raise AgentUnavailableError(
                    "The language model is misconfigured", code="llm_misconfigured"
                ) from exc

    return ensure_available
