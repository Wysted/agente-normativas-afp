"""AnalyzeNode: contextualizes the last question and rates how clear it is."""

from collections.abc import Sequence
from typing import Any, get_args

from langchain_core.messages import AnyMessage, SystemMessage
from langchain_core.runnables import Runnable
from langgraph.runtime import Runtime
from pydantic import BaseModel, ConfigDict, Field, model_validator

from agintis.agent_kit.nodes.history import MAX_HISTORY_MESSAGES, last_human_text, recent_history
from agintis.agent_kit.nodes.prompts import DEFAULT_ANALYZE_PROMPT
from agintis.agent_kit.nodes.state import RagState, RequestKind
from agintis.agent_kit.nodes.structured import bind_structured, invoke_structured
from agintis.agent_kit.runtime import AgentContext
from agintis.platform.llm import ChatModels, ChatRole


class QueryAnalysis(BaseModel):
    """Strict structured output: explicit typed fields only (no open objects, all required).

    No numeric bounds in the schema (some providers reject `minimum`/`maximum` in strict mode):
    `confidence` is clamped to [0, 1] in code.
    """

    model_config = ConfigDict(extra="forbid")

    standalone_query: str = Field(
        description="The last user message rewritten so it is understandable without the "
        "conversation."
    )
    is_clear: bool = Field(
        description="False only if the question is too vague or ambiguous to search."
    )
    confidence: float = Field(
        description="From 0 to 1: how sure you are that standalone_query captures the intent."
    )
    request_kind: RequestKind = Field(
        description="domain: a question or request about the agent's domain. social: greeting, "
        "thanks, small talk or asking what the agent can do. other: anything else (tasks such "
        "as writing code, personal or health advice, other topics).",
    )

    @model_validator(mode="before")
    @classmethod
    def _unknown_kind_is_domain(cls, data: object) -> object:
        # Fail-open: a missing or unknown kind must not discard the rest of the analysis. A
        # validator (not a default) keeps the field required in the JSON schema sent to providers.
        if isinstance(data, dict) and data.get("request_kind") not in get_args(RequestKind):
            return {**data, "request_kind": "domain"}
        return data


class AnalyzeNode:
    """Uses the ANALYZER model (never `streamed`: its output is internal, not user text).

    Writes `standalone_query`, `query` (the retrieval text, initially the same), `is_clear`,
    `confidence`, `request_kind` and, as the first node of a turn, `attempts = 0` and
    `previous_sources = []`
    (the rewrite loop's counter and kept evidence).
    The clarity fields are recorded for the clarification policy and not acted on here.
    The analysis is a hint: when the model output does not parse (see `structured.py`; provider
    and network errors still propagate) or the rewrite is empty, the node DEGRADES to
    the raw last user message with `is_clear=True` (search as-is: never block the user on a
    broken analyzer), `confidence=0.0` (unknown, so a clarification policy can tell it apart) and
    `request_kind="domain"` (fail-open: the question continues normally).
    """

    def __init__(
        self,
        chat_models: ChatModels,
        prompt: str = DEFAULT_ANALYZE_PROMPT,
        max_history_messages: int = MAX_HISTORY_MESSAGES,
    ) -> None:
        self._chat_models = chat_models
        self._prompt = prompt
        self._max_history = max_history_messages
        self._structured: Runnable[Any, Any] | None = None  # bound lazily, once

    async def analyze(self, messages: Sequence[AnyMessage]) -> dict[str, Any]:
        """The state update for a conversation (the node's whole job, usable without a graph)."""
        if self._structured is None:
            self._structured = bind_structured(
                self._chat_models.get(ChatRole.ANALYZER), QueryAnalysis
            )
        outcome = await invoke_structured(
            self._structured,
            [SystemMessage(self._prompt), *recent_history(messages, self._max_history)],
        )
        raw = last_human_text(messages)
        result = outcome.parsed
        if isinstance(result, QueryAnalysis) and result.standalone_query.strip():
            query = result.standalone_query.strip()
            is_clear, confidence = result.is_clear, min(1.0, max(0.0, result.confidence))
            request_kind = result.request_kind
        else:
            query, is_clear, confidence, request_kind = raw, True, 0.0, "domain"
        return {
            "standalone_query": query,
            "query": query,
            "is_clear": is_clear,
            "confidence": confidence,
            "request_kind": request_kind,
        }

    async def __call__(self, state: RagState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
        # Start of a turn: every key a later node reads or emits from is reset (a thread keeps
        # the previous turn's state): the rewrite counter, the carried evidence, `sources`
        # (synthesis emits them, so a path to it without a merge must not see the last turn's)
        # and the clarification verdict.
        return {
            **await self.analyze(state["messages"]),
            "attempts": 0,
            "previous_sources": [],
            "sources": [],
            "scope_level": None,  # not checked yet: the route step sets it when a policy runs
            "scope_score": None,
            "refusal_reason": None,
        }
