"""TopicRouter (Strategy) and RouteNode: which topics a question is searched in.

`SingleTopicRouter` always answers the same topic (no model call: the path of an agent with one
topic). `LlmTopicRouter` asks the ROUTER model, with strict structured output, for
`[{topic: enum, confidence: number}]`, most relevant first. `EvidenceTopicRouter` asks the
CORPUS instead (no model call): see its docstring.

Selection (`select_topics`): the PRIMARY topic (highest confidence) is always searched; an extra
topic only when its confidence reaches `secondary_min_confidence`, up to `max_topics` in total.
A second topic that is not really needed costs a model call and, worse, pushes the right topic's
hits down in the merge: the confidence gate is what keeps the fan-out narrow when the router is
sure.

`RouteNode` turns the selection into a `Send` fan-out (one parallel run of the topic subgraph
per topic) written as `Command(goto=[Send(...)])`: the node itself decides the next step, so no
conditional edge is needed and the reset of `topic_results` (`Overwrite([])`, see `state.py`)
travels in the same `Command`. It guarantees a non-empty selection: `Command(goto=[])` would end
the run silently without an answer.

Scope check (3.5c): on the FIRST route of a turn (`attempts == 0`) `RouteNode` first asks its
`ClarificationPolicy` (Strategy) whether the question belongs to the corpus at all. A policy that
probes the corpus (`EvidenceClarification`) hands back the topics it selected from the SAME probe
(`ScopeDecision.routed`), so the router is not asked again. The verdict is announced with ONE
`Scope` event, and `out` goes to the refuse node instead of the fan-out. A rewrite -> route round
is not re-checked: the turn keeps its level.

Degradation of the LLM router (a routing failure must never block the user): output that does
not parse is first RESCUED from the raw JSON (known topic names only, order kept, like the filter
extractor does per field); with nothing valid left (or an empty selection) it falls back to
`fallback_topics`: ALL topics when they fit in `max_topics`, otherwise the first `max_topics` of
them in spec order (confidence 0.0 = unknown). Provider and network errors still propagate.
"""

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langgraph.runtime import Runtime
from langgraph.types import Command, Overwrite, Send
from pydantic import BaseModel, ConfigDict, Field, create_model

from agintis.agent_kit.nodes.history import last_human_text
from agintis.agent_kit.nodes.prompts import DEFAULT_ROUTE_PROMPT
from agintis.agent_kit.nodes.state import RagState, RoutedTopic, TopicTask
from agintis.agent_kit.nodes.structured import bind_structured, invoke_structured
from agintis.agent_kit.runtime import AgentContext, QueryVectors, emit
from agintis.modules.agents_catalog.api import Scope, ScopeLevel
from agintis.modules.knowledge.api import (
    DEFAULT_MAX_QUERY_CHARS,
    DEFAULT_PROBE_K,
    ProbeResult,
    RagProfile,
)
from agintis.platform.llm import ChatModels, ChatRole
from agintis.shared_kernel.ids import TenantId

logger = logging.getLogger(__name__)

# Topics searched in parallel when the spec says nothing.
DEFAULT_MAX_TOPICS = 2
# An extra topic is searched only when the router is at least this sure of it.
DEFAULT_SECONDARY_MIN_CONFIDENCE = 0.6
# Name of the graph node that runs the topic subgraph (the target of every Send).
TOPIC_NODE = "topic"
# Name of the graph node a refused (out of scope) question goes to.
REFUSE_NODE = "refuse"
UNKNOWN_CONFIDENCE = 0.0
_SCHEMA_NAME = "TopicRoute"
_CHOICE_NAME = "TopicChoice"


@dataclass(frozen=True, slots=True)
class TopicInfo:
    """What the router knows about a topic: its name and what it covers."""

    name: str
    description: str


@dataclass(frozen=True, slots=True)
class RouteScope:
    """Per-run data a router may need: who asks (tenant scope of corpus probes) and the run's
    query-embedding memo (`QueryVectors`, shared with the retrieval that follows)."""

    tenant_id: TenantId
    vectors: QueryVectors


class TopicRouter(Protocol):
    """Strategy: the topics to search for a query, primary first, never empty.

    `scope` is what the run knows (`RouteNode` always passes it); strategies that do not search
    the corpus ignore it, an evidence router requires it. `widen_to` (above the router's own
    `max_topics`) asks an evidence router for that many topics regardless of its margin (the
    clarification policy's `low` band); the others ignore it.
    """

    async def route(
        self, query: str, scope: RouteScope | None = None, *, widen_to: int | None = None
    ) -> list[RoutedTopic]: ...


def fallback_topics(names: Sequence[str], max_topics: int) -> list[RoutedTopic]:
    """All topics when they fit in `max_topics`, else the first `max_topics` in spec order."""
    return [RoutedTopic(topic=name, confidence=UNKNOWN_CONFIDENCE) for name in names[:max_topics]]


def clamp_confidence(value: object) -> float:
    """A model-written confidence as a number in [0, 1] (anything else is unknown)."""
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        return UNKNOWN_CONFIDENCE
    return min(1.0, max(0.0, float(value)))


def select_topics(
    candidates: Sequence[RoutedTopic], max_topics: int, secondary_min_confidence: float
) -> list[RoutedTopic]:
    """Deduplicated (first mention wins), ordered by confidence (stable: the model's order breaks
    ties), the primary always, extras only at or above `secondary_min_confidence`, capped."""
    unique: dict[str, RoutedTopic] = {}
    for candidate in candidates:
        unique.setdefault(candidate["topic"], candidate)
    ordered = sorted(unique.values(), key=lambda c: -c["confidence"])
    return [
        c for i, c in enumerate(ordered) if i == 0 or c["confidence"] >= secondary_min_confidence
    ][:max_topics]


def _check_topics(topics: Sequence[TopicInfo], max_topics: int) -> None:
    if not topics:
        raise ValueError("A router needs at least one topic")
    if len({t.name for t in topics}) != len(topics):
        raise ValueError("Topic names must be unique")
    if max_topics < 1:
        raise ValueError("max_topics must be at least 1")


class SingleTopicRouter:
    """Always the same topic; never calls a model."""

    def __init__(self, topic: str) -> None:
        self._topic = topic

    async def route(
        self, query: str, scope: RouteScope | None = None, *, widen_to: int | None = None
    ) -> list[RoutedTopic]:
        return [RoutedTopic(topic=self._topic, confidence=1.0)]


def build_route_schema(names: Sequence[str]) -> type[BaseModel]:
    """Strict structured output: one required array of `{topic, confidence}` objects, `topic` an
    ENUM of the topic names.

    No `minItems`/`maxItems` and no numeric bounds (some providers reject them in strict mode):
    emptiness, the cap and the [0, 1] range are handled in code.
    """
    topic_name = cast(Any, Literal)[tuple(names)]
    choice = create_model(
        _CHOICE_NAME,
        __config__=ConfigDict(extra="forbid"),
        topic=(topic_name, Field(description="A topic name from the list.")),
        confidence=(
            float,
            Field(description="From 0 to 1: how sure you are that this topic holds the answer."),
        ),
    )
    return create_model(
        _SCHEMA_NAME,
        __config__=ConfigDict(extra="forbid"),
        topics=(
            cast(Any, list)[choice],
            Field(description="The topics that can answer, most relevant first."),
        ),
    )


class LlmTopicRouter:
    """Uses the ROUTER model (never `streamed`: its output is internal).

    The system message is the prompt plus the topic list (name: description) and the cap; the
    user message is the query. See the module docstring for selection and degradation.
    """

    def __init__(
        self,
        chat_models: ChatModels,
        topics: Sequence[TopicInfo],
        max_topics: int = DEFAULT_MAX_TOPICS,
        prompt: str = DEFAULT_ROUTE_PROMPT,
        secondary_min_confidence: float = DEFAULT_SECONDARY_MIN_CONFIDENCE,
    ) -> None:
        _check_topics(topics, max_topics)
        if not 0.0 <= secondary_min_confidence <= 1.0:
            raise ValueError("secondary_min_confidence must be between 0 and 1")
        self._chat_models = chat_models
        self._names = [t.name for t in topics]
        self._max_topics = max_topics
        self._secondary_min_confidence = secondary_min_confidence
        self._schema = build_route_schema(self._names)
        listing = "\n".join(f"- {t.name}: {' '.join(t.description.split())}" for t in topics)
        self._system = (
            f"{prompt}\n\nTopics:\n{listing}\n\nReturn at most {max_topics} "
            f"topic{'s' if max_topics != 1 else ''}, each with your confidence."
        )
        self._structured: Runnable[Any, Any] | None = None  # bound lazily, once

    def _candidates(self, chosen: object) -> list[RoutedTopic]:
        """Known topics out of a model-written list of objects (or bare names, unknown
        confidence); anything else in it is ignored."""
        if not isinstance(chosen, list):
            return []
        found: list[RoutedTopic] = []
        for entry in chosen:
            if isinstance(entry, BaseModel):
                entry = entry.model_dump()
            name: object
            if isinstance(entry, str):
                name, confidence = entry, UNKNOWN_CONFIDENCE
            elif isinstance(entry, dict):
                name, confidence = entry.get("topic"), clamp_confidence(entry.get("confidence"))
            else:
                continue
            if isinstance(name, str) and name in self._names:
                found.append(RoutedTopic(topic=name, confidence=confidence))
        return found

    async def route(
        self, query: str, scope: RouteScope | None = None, *, widen_to: int | None = None
    ) -> list[RoutedTopic]:
        if self._structured is None:
            self._structured = bind_structured(self._chat_models.get(ChatRole.ROUTER), self._schema)
        outcome = await invoke_structured(
            self._structured, [SystemMessage(self._system), HumanMessage(query)]
        )
        if outcome.parsed is not None:
            candidates = self._candidates(getattr(outcome.parsed, "topics", None))
        else:  # rescue what is valid from the raw JSON
            candidates = self._candidates((outcome.raw_values or {}).get("topics"))
            if candidates:
                logger.info("Router output did not parse: rescued %d topic(s)", len(candidates))
        selected = select_topics(candidates, self._max_topics, self._secondary_min_confidence)
        if not selected:
            logger.info("Router degraded to the fallback topics (no usable selection)")
            return fallback_topics(self._names, self._max_topics)
        return selected


# Evidence router defaults. The margin is in cosine-similarity units (bge-m3 scores of Spanish
# passages sit around 0.4-0.8, and the best passages of two overlapping topics differ by a few
# hundredths): 0.02 keeps the fan-out narrow until the eval calibrates it.
DEFAULT_EVIDENCE_MARGIN = 0.02
# Float noise tolerance of the margin comparison (a margin of 0.02 admits a 0.02 gap).
_MARGIN_EPSILON = 1e-9

type EvidenceAggregate = Literal["max", "mean_top_k"]
DEFAULT_EVIDENCE_AGGREGATE: EvidenceAggregate = "max"


class TopicProbe(Protocol):
    """What the evidence router needs from the knowledge module (`ProbeProfiles` satisfies it)."""

    async def execute(
        self,
        tenant_id: TenantId,
        profiles: Sequence[RagProfile],
        query: str,
        k: int = DEFAULT_PROBE_K,
        query_vector: Sequence[float] | None = None,
    ) -> ProbeResult: ...


@dataclass(frozen=True, slots=True)
class EvidenceTopic:
    """A topic and the profile whose content is its evidence (`bias`: see the router)."""

    name: str
    profile: RagProfile
    bias: float = 0.0


@dataclass(frozen=True, slots=True)
class TopicEvidence:
    """What a topic's profile holds for a query: the probe similarities (best first) and their
    aggregate (`raw_score`, None = no hit at all). `score` adds the topic's `bias`."""

    topic: str
    similarities: tuple[float, ...]
    raw_score: float | None
    bias: float = 0.0

    @property
    def score(self) -> float | None:
        """The ranking score: the aggregate of the similarities plus the topic's bias."""
        return None if self.raw_score is None else self.raw_score + self.bias


def aggregate_similarities(
    similarities: Sequence[float], aggregate: EvidenceAggregate
) -> float | None:
    """One score out of a topic's probe similarities (best first); None without any."""
    if not similarities:
        return None
    if aggregate == "max":
        return max(similarities)
    return sum(similarities) / len(similarities)


def select_by_evidence(
    evidence: Sequence[TopicEvidence], max_topics: int, margin: float
) -> list[RoutedTopic]:
    """The topics worth searching given their evidence: the best-scoring one (the primary) plus
    those within `margin` of it, best first (ties keep the input order), at most `max_topics`.
    Topics without a hit are never selected; empty when none has one."""
    scored = [(e, e.score) for e in evidence if e.score is not None and math.isfinite(e.score)]
    if not scored:
        return []
    ordered = sorted(scored, key=lambda pair: -pair[1])
    best = ordered[0][1]
    return [
        RoutedTopic(
            topic=e.topic,
            confidence=clamp_confidence(score),
            similarities=[float(x) for x in e.similarities],
        )
        for e, score in ordered
        if best - score <= margin + _MARGIN_EPSILON
    ][:max_topics]


class EvidenceTopicRouter:
    """Routes by what the corpus says, not by what a model thinks: no LLM call.

    The query is embedded ONCE and each topic's profile is probed with a small dense search
    (`TopicProbe`, tenant-scoped, the profile's default filter applied). A topic's score is the
    aggregate of its top `probe_k` similarities:

    - `max` (default): the single best passage. Simple and not diluted by a profile's weak
      tail, but NOT neutral: the maximum of N samples grows with N, so a big profile tends to
      score higher than a small one for the same question, and chunk length per profile shifts
      the whole cosine distribution (short, dense chunks and long ones do not score alike).
      The margin is symmetric, so it cannot absorb a one-sided offset like that.
    - `mean_top_k`: the mean of the (up to `probe_k`) hits it has, robust to a lone outlier
      chunk but biased the other way (a small profile has fewer, worse tail hits to average;
      no padding is done, so a one-hit profile scores that hit).

    `EvidenceTopic.bias` (default 0.0) is added to a topic's score before ranking, to correct a
    measured one-sided offset. It must be calibrated on a DEV set, never on the questions the
    router is evaluated with (held-out): leave it at 0 until there is one. `similarities` stay
    raw; `TopicEvidence.raw_score` is the score without the bias.

    Selection: the best-scoring topic is the primary and is always searched; another topic only
    when `best - score <= evidence_margin`, up to `max_topics`. A topic with NO hit is never
    selected unless no topic has one (then `fallback_topics`, confidence 0.0). The returned
    `confidence` is the score (clamped to [0, 1]), `similarities` the raw evidence behind it.

    Similarities are comparable across topics because every profile shares one embedding space
    (see `ProbeProfiles`). The query vector goes into `scope.vectors`, keyed by the text the
    retrieval will search (the query cut to `max_query_chars`), so the topic subgraphs reuse it.
    """

    def __init__(
        self,
        probe: TopicProbe,
        topics: Sequence[EvidenceTopic],
        max_topics: int = DEFAULT_MAX_TOPICS,
        margin: float = DEFAULT_EVIDENCE_MARGIN,
        aggregate: EvidenceAggregate = DEFAULT_EVIDENCE_AGGREGATE,
        probe_k: int = DEFAULT_PROBE_K,
        max_query_chars: int = DEFAULT_MAX_QUERY_CHARS,
    ) -> None:
        if not topics:
            raise ValueError("A router needs at least one topic")
        if len({t.name for t in topics}) != len(topics):
            raise ValueError("Topic names must be unique")
        if len({t.profile.name for t in topics}) != len(topics):
            raise ValueError("Topic profiles must be unique")
        if max_topics < 1:
            raise ValueError("max_topics must be at least 1")
        if margin < 0.0 or not math.isfinite(margin):
            raise ValueError("margin must be a non-negative number")
        if aggregate not in ("max", "mean_top_k"):
            raise ValueError("aggregate must be 'max' or 'mean_top_k'")
        if probe_k < 1:
            raise ValueError("probe_k must be at least 1")
        if any(not math.isfinite(t.bias) for t in topics):
            raise ValueError("topic bias must be a finite number")
        if max_query_chars < 1:
            raise ValueError("max_query_chars must be at least 1")
        self._probe = probe
        self._topics = list(topics)
        self._names = [t.name for t in topics]
        self._max_topics = max_topics
        self._margin = margin
        self._aggregate: EvidenceAggregate = aggregate
        self._probe_k = probe_k
        self._max_query_chars = max_query_chars

    async def evidence(self, query: str, scope: RouteScope | None) -> list[TopicEvidence]:
        """The evidence of EVERY topic (spec order), one embedding and one search per topic."""
        if scope is None:
            raise ValueError("EvidenceTopicRouter needs a RouteScope (tenant and vector memo)")
        text = query[: self._max_query_chars]
        result = await self._probe.execute(
            scope.tenant_id,
            [t.profile for t in self._topics],
            text,
            self._probe_k,
            scope.vectors.get(text),
        )
        if result.query_vector is not None:
            scope.vectors.put(text, result.query_vector)
        found = {e.profile: e.similarities for e in result.evidence}
        evidence = []
        for topic in self._topics:
            similarities = tuple(found.get(topic.profile.name, ()))
            raw = aggregate_similarities(similarities, self._aggregate)
            evidence.append(TopicEvidence(topic.name, similarities, raw, topic.bias))
        return evidence

    def select(
        self, evidence: Sequence[TopicEvidence], *, widen_to: int | None = None
    ) -> list[RoutedTopic]:
        """The routed topics for already gathered `evidence` (see the class docstring).

        `widen_to` above `max_topics` (the clarification policy's `low` band, where the best
        topic is least trustworthy) takes the best `widen_to` topics regardless of the margin.

        Logs the selection, so both callers (`route` and a clarification policy that already
        probed) leave exactly one routing line per route.
        """
        widen = widen_to is not None and widen_to > self._max_topics
        top, margin = (cast(int, widen_to), math.inf) if widen else (self._max_topics, self._margin)
        routed = select_by_evidence(evidence, top, margin) or fallback_topics(self._names, top)
        logger.info(
            "Evidence router: %s -> %s",
            ", ".join(f"{e.topic}={e.score:.3f}" for e in evidence if e.score is not None) or "-",
            ", ".join(r["topic"] for r in routed),
        )
        return routed

    async def route(
        self, query: str, scope: RouteScope | None = None, *, widen_to: int | None = None
    ) -> list[RoutedTopic]:
        evidence = await self.evidence(query, scope)
        return self.select(evidence, widen_to=widen_to)


@dataclass(frozen=True, slots=True)
class ScopeDecision:
    """A clarification policy's verdict on a question.

    `score` is the evidence it rests on (None = there was none). `routed` are the topics the
    policy already selected from the same corpus probe (None = the router must be asked).
    """

    level: ScopeLevel
    score: float | None = None
    routed: list[RoutedTopic] | None = None


class ClarificationPolicy(Protocol):
    """Strategy: does the question belong to the corpus, before any retrieval?

    `None` = no opinion: route normally and announce nothing (the `none` policy). A future
    policy that asks the user (an `interrupt()` in a node of the parent graph, never inside the
    topic subgraph) plugs in here and in the graph builder, without touching the retrieval nodes.

    `may_refuse` tells the graph builder whether a turn can end in the refuse node (only then
    is that node added).
    """

    @property
    def may_refuse(self) -> bool: ...

    async def decide(self, query: str, scope: RouteScope) -> ScopeDecision | None: ...


class NoClarification:
    """The default policy: never checks anything, so the graph behaves as without a policy."""

    @property
    def may_refuse(self) -> bool:
        return False

    async def decide(self, query: str, scope: RouteScope) -> ScopeDecision | None:
        return None


class RouteNode:
    """Picks the topics and fans out: `Command(goto=[Send(topic node, TopicTask)...])`.

    Starting a fan-out also writes `routed_topics` (the merge order, with confidences) and resets
    `topic_results` with `Overwrite([])`, so results of an earlier turn or rewrite attempt never
    leak in. The query is `state["query"]` (the analyzer's standalone query, or the rewritten
    one). A router answer without any known topic (a buggy strategy) is replaced by
    `fallback_topics(topics, max_topics)`: an empty `goto` would end the run silently.

    With a policy (first route of the turn only) the policy writes `scope_level` / `scope_score`,
    emits the `Scope` event and, for `out`, goes to `refuse_node` with no fan-out
    (`refusal_reason` `out_of_scope`). The analyzer's `request_kind` `social` / `other` is
    refused (`refusal_reason` `social` / `off_domain`, no fan-out) unless the probe says `in`:
    a domain request phrased as a task continues as a domain one. `other` then reports scope
    `out` (no score) so clients and judges see a refusal; `social` announces no scope.
    `low_scope_max_topics` is the policy's widened fan-out for the `low` band, re-applied when a
    rewrite round routes again.
    """

    def __init__(
        self,
        router: TopicRouter,
        topics: Sequence[str],
        max_topics: int = DEFAULT_MAX_TOPICS,
        topic_node: str = TOPIC_NODE,
        policy: ClarificationPolicy | None = None,
        refuse_node: str = REFUSE_NODE,
        low_scope_max_topics: int | None = None,
    ) -> None:
        if not topics:
            raise ValueError("RouteNode needs at least one topic")
        if max_topics < 1:
            raise ValueError("max_topics must be at least 1")
        self._router = router
        self._topics = list(topics)
        self._max_topics = max_topics
        self._topic_node = topic_node
        self._policy: ClarificationPolicy = policy or NoClarification()
        self._refuse_node = refuse_node
        self._low_topics = low_scope_max_topics

    async def __call__(self, state: RagState, runtime: Runtime[AgentContext]) -> Command[Any]:
        query = state.get("query") or last_human_text(state["messages"])
        scope = RouteScope(runtime.context.tenant_id, runtime.context.query_vectors)
        update: dict[str, Any] = {}
        selected: list[RoutedTopic] | None = None
        if (state.get("attempts") or 0) == 0:  # a rewrite round keeps the turn's verdict
            kind = state.get("request_kind", "domain")
            decision = await self._policy.decide(query, scope)
            if (
                kind != "domain"
                and self._policy.may_refuse
                and (decision is None or decision.level != "in")
            ):
                # Only strong evidence (band `in`) lets a task-phrased domain request through.
                if kind == "other":
                    emit(runtime, Scope("out", None))
                    update = {"scope_level": "out", "scope_score": None}
                reason = "off_domain" if kind == "other" else "social"
                return Command(
                    update={
                        **update,
                        "refusal_reason": reason,
                        "routed_topics": [],
                        "topic_results": Overwrite([]),
                    },
                    goto=self._refuse_node,
                )
            if decision is not None:
                emit(runtime, Scope(decision.level, decision.score))
                update = {"scope_level": decision.level, "scope_score": decision.score}
                if decision.level == "out":
                    return Command(
                        update={
                            **update,
                            "refusal_reason": "out_of_scope",
                            "routed_topics": [],
                            "topic_results": Overwrite([]),
                        },
                        goto=self._refuse_node,
                    )
                selected = decision.routed
        if selected is None:
            # A rewrite round in the `low` band keeps the widened fan-out of the first attempt.
            widen = self._low_topics if state.get("scope_level") == "low" else None
            selected = await self._router.route(query, scope, widen_to=widen)
        routed = [r for r in selected if r["topic"] in self._topics]
        if not routed:
            logger.warning("Router returned no known topic: using the fallback topics")
            routed = fallback_topics(self._topics, self._max_topics)
        return Command(
            update={**update, "routed_topics": routed, "topic_results": Overwrite([])},
            goto=[Send(self._topic_node, TopicTask(topic=r["topic"], query=query)) for r in routed],
        )
