"""ClarificationPolicy strategies (3.5c) and the node that refuses an out of scope question.

`NoClarification` (in `router.py`, next to `RouteNode`, its only caller) checks nothing.
`EvidenceClarification` asks the corpus: the signal is the MAX over ALL topics of
`TopicEvidence.raw_score` (the best raw cosine similarity the evidence router found anywhere; NOT
`RoutedTopic.confidence`, which is biased, clamped and only covers the selected topics), computed
on the query of the first route of the turn. Bands, with STRICT comparisons:

- `score > low_confidence_below`: `in`, answer normally.
- `out_of_scope_below < score <= low_confidence_below`: `low`, answer normally but tell the
  client the match is weak (the `Scope` event). Routing is the weak spot of this band, so with
  `low_scope_max_topics` above the router's `max_topics` the best `low_scope_max_topics` topics
  are searched, whatever the margin.
- `score <= out_of_scope_below`, or no evidence at all: `out`, refuse with `RefuseNode`: no
  retrieval, no synthesis call, no `Sources`.

Before the bands, `RouteNode` also short-circuits requests the analyzer read as not being about
the domain (`request_kind`): `social` gets the courtesy message, `other` the out of scope one,
both through `RefuseNode`, with no probe, no retrieval and no synthesis call.

The probe is the router's own: `decide` gathers the evidence ONCE and also returns the topics
selected from it, so the fan-out reuses it (one embedding and one search per topic per route).

The score separates DOMAIN, not answerability: a question next to the corpus' subject that the
corpus does not answer still scores high and relies on the synthesis prompt's refusal. The
thresholds are valid only for the corpus snapshot they were calibrated on
(`cli.calibrate_evidence`).
"""

import logging
import math
from collections.abc import Sequence
from typing import Any, Protocol

from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from agintis.agent_kit.nodes.router import RouteScope, ScopeDecision, TopicEvidence
from agintis.agent_kit.nodes.state import RagState, RoutedTopic
from agintis.agent_kit.runtime import AgentContext, emit
from agintis.modules.agents_catalog.api import ScopeLevel, Token

logger = logging.getLogger(__name__)

# What a refused question is told when the spec says nothing.
DEFAULT_OUT_OF_SCOPE_MESSAGE = (
    "I can only answer questions about the documents I have access to, and I found nothing "
    "related to your question. Could you rephrase it or ask about something they cover?"
)
# What small talk (greeting, thanks, "what can you do") is answered when the spec says nothing.
DEFAULT_SOCIAL_MESSAGE = (
    "Hello! I can answer questions about the documents I have access to. Ask me about "
    "something they cover."
)


class EvidenceRouter(Protocol):
    """What `EvidenceClarification` needs from the evidence router (`EvidenceTopicRouter`)."""

    async def evidence(self, query: str, scope: RouteScope | None) -> list[TopicEvidence]: ...

    def select(
        self, evidence: Sequence[TopicEvidence], *, widen_to: int | None = None
    ) -> list[RoutedTopic]: ...


def max_raw_score(evidence: Sequence[TopicEvidence]) -> float | None:
    """The best raw similarity over ALL topics (no bias); None when no topic found anything."""
    scores = [
        e.raw_score for e in evidence if e.raw_score is not None and math.isfinite(e.raw_score)
    ]
    return max(scores) if scores else None


def scope_level(
    score: float | None, low_confidence_below: float, out_of_scope_below: float
) -> ScopeLevel:
    """The band of a score (strict `>` against the thresholds); no score at all is `out`."""
    if score is None or score <= out_of_scope_below:
        return "out"
    return "in" if score > low_confidence_below else "low"


class EvidenceClarification:
    """Scope check by corpus evidence (see the module docstring)."""

    def __init__(
        self,
        router: EvidenceRouter,
        low_confidence_below: float,
        out_of_scope_below: float,
        low_scope_max_topics: int | None = None,
    ) -> None:
        if not (
            math.isfinite(low_confidence_below)
            and math.isfinite(out_of_scope_below)
            and 0.0 <= out_of_scope_below < low_confidence_below <= 1.0
        ):
            raise ValueError("thresholds need 0 <= out_of_scope_below < low_confidence_below <= 1")
        if low_scope_max_topics is not None and low_scope_max_topics < 1:
            raise ValueError("low_scope_max_topics must be at least 1")
        self._router = router
        self._low_topics = low_scope_max_topics
        self._low = low_confidence_below
        self._out = out_of_scope_below

    @property
    def may_refuse(self) -> bool:
        return True

    async def decide(self, query: str, scope: RouteScope) -> ScopeDecision | None:
        evidence = await self._router.evidence(query, scope)
        score = max_raw_score(evidence)
        level = scope_level(score, self._low, self._out)
        logger.info(
            "Scope check: %s (max evidence %s, bands %.4f / %.4f)",
            level,
            "none" if score is None else f"{score:.4f}",
            self._out,
            self._low,
        )
        if score is None:
            logger.warning(
                "Scope check: no topic has any evidence for tenant %r (wrong tenant or the "
                "corpus is not ingested yet?): refusing",
                scope.tenant_id,
            )
        if level == "out":
            return ScopeDecision(level, score)
        widen = self._low_topics if level == "low" else None
        return ScopeDecision(level, score, self._router.select(evidence, widen_to=widen))


class RefuseNode:
    """Answers a refused turn with a fixed message and no model call.

    The message depends on `state["refusal_reason"]`: the courtesy one for `social`, the out of
    scope one for `out_of_scope` and `off_domain`. It reaches the client as ONE `Token`
    (node-returned messages are not streamed by the runner, only tagged LLM calls and custom
    events are), and is appended to `messages` as an AI message so a follow-up in the thread sees
    a coherent history. Writes the keys that `SynthesizeNode` owns (`answer`, `sources`) so
    nothing of an earlier turn survives.
    """

    def __init__(
        self,
        message: str = DEFAULT_OUT_OF_SCOPE_MESSAGE,
        social_message: str = DEFAULT_SOCIAL_MESSAGE,
    ) -> None:
        if not message.strip() or not social_message.strip():
            raise ValueError("the refusal messages cannot be blank")
        self._message = message
        self._social_message = social_message

    async def __call__(self, state: RagState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
        text = self._social_message if state.get("refusal_reason") == "social" else self._message
        emit(runtime, Token(text))
        return {"messages": [AIMessage(text)], "answer": text, "sources": []}
