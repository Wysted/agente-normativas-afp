"""Builders shared by the rag-chat graph tests (multi-topic and clarification scenarios)."""

from collections.abc import Callable
from typing import Any

from agintis.modules.agents.rag_chat import AgentSpec, parse_agent_spec
from agintis.modules.agents_catalog.api import RunInput
from agintis.modules.knowledge.api import MetadataField, RagProfile, RagProfileRegistry
from agintis.shared_kernel.ids import RunId, TenantId, ThreadId
from tests.fakes.llm import ScriptedChatModel
from tests.fakes.retrieval import FakeRetrieval, FakeTopicProbe, RetrievalCall, make_hit

TENANT = TenantId("acme")
ALPHA = RagProfile(
    "alpha",
    3,
    100,
    "simple",
    "structured",
    metadata_fields=(MetadataField("kind", "str", filterable=True),),
)
BETA = RagProfile(
    "beta",
    3,
    100,
    "simple",
    "structured",
    metadata_fields=(MetadataField("num", "int", filterable=True),),
)
PROFILES = RagProfileRegistry([ALPHA, BETA])


def multi_topic_spec(**changes: Any) -> AgentSpec:
    return parse_agent_spec(
        {
            "id": "multi-rag",
            "description": "Answers from two topics.",
            "topics": [
                {
                    "name": "alpha",
                    "description": "Alpha documents",
                    "profile": "alpha",
                    "filters": {"kind": "The kind"},
                },
                {
                    "name": "beta",
                    "description": "Beta documents",
                    "profile": "beta",
                    "filters": {"num": "The number"},
                },
            ],
            # count/weight 1 keep the scenarios below simple; relevance and weights are tested
            # explicitly further down.
            "sufficiency": "count",
            "primary_weight": 1,
            **changes,
        },
        origin="test",
    )


class Script:
    """Answers each structured call by its schema (several run in parallel), and records them.

    `routes` = the router's selection per call (the last one repeats); `rewrites` = the queries
    the rewriter returns, one per call.
    """

    def __init__(
        self,
        routes: list[list[str]],
        rewrites: list[str] | None = None,
        confidence: float = 0.9,
        request_kind: str = "domain",
    ) -> None:
        self.confidence = confidence
        self.request_kind = request_kind
        self.routes = list(routes)
        self.rewrites = list(rewrites or ["broader query"])
        self.route_calls = 0

    def __call__(self, schema: Any, messages: list[Any]) -> Any:
        fields = {f.alias or name for name, f in schema.model_fields.items()}
        match schema.__name__:
            case "QueryAnalysis":
                return {
                    "standalone_query": str(messages[-1].content),
                    "is_clear": True,
                    "confidence": 0.9,
                    "request_kind": self.request_kind,
                }
            case "TopicRoute":
                self.route_calls += 1
                chosen = self.routes[min(self.route_calls, len(self.routes)) - 1]
                return {"topics": [{"topic": t, "confidence": self.confidence} for t in chosen]}
            case "QueryRewrite":
                return {"query": self.rewrites.pop(0) if self.rewrites else "again"}
            case "QueryFilters" if "kind" in fields:
                return {"kind": "memo"}
            case "QueryFilters":
                return {"num": 7}
        raise AssertionError(f"unexpected schema {schema.__name__}")


def make_hits(profile: str, *names: str) -> tuple[Any, ...]:
    return tuple(
        make_hit(f"text {n}", uri=f"file:///{n}", title=n, ordinal=i) for i, n in enumerate(names)
    )


def by_query(**per_query: dict[str, tuple[str, ...]]) -> Callable[[RetrievalCall], Any]:
    """query -> profile -> names. Unknown queries find nothing."""

    def respond(call: RetrievalCall) -> tuple[Any, ...]:
        names = per_query.get(call.query.replace(" ", "_"), {}).get(call.profile.name, ())
        return make_hits(call.profile.name, *names)

    return respond


def make_model(script: Script, replies: list[str]) -> ScriptedChatModel:
    return ScriptedChatModel(responder=script, replies=list(replies))


def make_input(message: str = "question", thread: ThreadId | None = None) -> RunInput:
    return RunInput(TENANT, message, RunId("0123456789abcdef0123456789abcdef"), thread)


LOW = 0.5  # exactly representable: the boundary tests compare exactly
OUT = 0.25
REFUSAL = "Out of scope."
CLARIFICATION = {
    "policy": "evidence",
    "low_confidence_below": LOW,
    "out_of_scope_below": OUT,
    "out_of_scope_message": REFUSAL,
}


def scoped_spec(**changes: Any) -> AgentSpec:
    return parse_agent_spec(
        {
            "id": "scoped-rag",
            "description": "Answers inside its corpus only.",
            "topics": [
                {"name": "alpha", "description": "Alpha", "profile": "alpha", "filters": {}},
                {"name": "beta", "description": "Beta", "profile": "beta", "filters": {}},
            ],
            "router": "evidence",
            "evidence_margin": 0.0,
            "sufficiency": "count",
            "primary_weight": 1,
            "max_rewrites": 0,
            "clarification": CLARIFICATION,
            **changes,
        },
        origin="test",
    )


def make_probe(**scores: float | None) -> FakeTopicProbe:
    """Per query: the alpha similarity it scores (None = no evidence at all)."""
    return FakeTopicProbe({q: ({} if s is None else {"alpha": [s]}) for q, s in scores.items()})


def make_retrieval() -> FakeRetrieval:
    return FakeRetrieval(respond=lambda call: make_hits(call.profile.name, "a1"))
