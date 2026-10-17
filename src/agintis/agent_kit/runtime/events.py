"""Node-side helpers: how a graph node talks to the client while it runs."""

from typing import Any

from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from langgraph.runtime import Runtime

from agintis.modules.agents_catalog.api import Clarification, Scope, Sources, Step, Token

# Opt-in marker for token streaming. `messages` mode streams the chunks of EVERY LLM call of
# the graph (analyzer, router, structured output, ...); only calls tagged with this reach the
# client as `Token` events.
STREAM_TOKENS_TAG = "agintis:stream"

# Application events a node may emit through the graph's custom stream.
NodeEvent = Token | Step | Sources | Scope | Clarification
NODE_EVENT_TYPES = (Token, Step, Sources, Scope, Clarification)


def streamed(model: BaseChatModel) -> Runnable[LanguageModelInput, AIMessage]:
    """Mark a chat model so its text reaches the client as `Token` events.

    Apply it LAST and only to the leaf chat model that writes user-facing text:
    - never to structured-output / internal calls (analyzer, router);
    - never to a parent runnable (tags are inherited by every child LLM call inside it);
    - `bind_tools()` / `with_structured_output()` called AFTER it drop the tag (they rebuild the
      binding from the bare model): bind first, then `streamed(...)`.
    """
    return model.with_config(tags=[STREAM_TOKENS_TAG])


def emit(runtime: Runtime[Any], event: NodeEvent) -> None:
    """Send an application event to the client from inside a node (`runtime` is the node's).

    Never emit a `Clarification` before `interrupt()`: nodes re-run from the start on resume, so
    it would be sent twice. The runner builds it from the interrupt payload instead.
    """
    runtime.stream_writer(event)
