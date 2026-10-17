"""The multi-topic rag-chat graph: router, Send fan-out, merge and the sufficiency loop."""

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from agintis.agent_kit.runtime import DEFAULT_RECURSION_LIMIT, AgentContext
from agintis.bootstrap.adapters import ChainedErrorClassifier
from agintis.modules.agents.rag_chat import (
    AgentSpec,
    RagChatDeps,
    build_graph,
    parse_agent_spec,
    rag_chat_runner,
)
from agintis.modules.agents.rag_chat.graph import topic_step_label
from agintis.modules.agents.rag_chat.spec import MAX_REWRITES_LIMIT
from agintis.modules.agents_catalog.api import (
    AgentRegistry,
    AgentUnavailableError,
    RunAgent,
    RunCompleted,
    RunFailed,
    RunStarted,
    Sources,
    Step,
    Token,
)
from agintis.modules.knowledge.infrastructure.errors import RetrievalErrorClassifier
from agintis.modules.knowledge.infrastructure.tei import TeiUnavailableError
from agintis.platform.llm import ChatModels, ChatRole, LlmConfig, ProviderErrorClassifier
from agintis.shared_kernel.criteria import Filter
from agintis.shared_kernel.ids import RunId, ThreadId
from tests.fakes.llm import RaisingModel, ScriptedChatModel, chat_models_for, rate_limit_error
from tests.fakes.retrieval import FakeRetrieval, RetrievalCall
from tests.unit.agents.support import (
    PROFILES,
    TENANT,
    Script,
    by_query,
    make_hits,
    make_input,
    make_model,
    multi_topic_spec,
)


def _by_profile(
    **found: tuple[str, ...],
) -> Callable[[RetrievalCall], tuple[Any, ...]]:
    return lambda call: make_hits(call.profile.name, *found.get(call.profile.name, ()))


def _runner(
    model: BaseChatModel,
    retrieval: FakeRetrieval,
    spec: AgentSpec | None = None,
    saver: InMemorySaver | None = None,
    **by_role: BaseChatModel,
) -> Any:
    roles = {ChatRole(k): v for k, v in by_role.items()}
    deps = RagChatDeps(chat_models_for(model, by_role=roles), retrieval)
    return rag_chat_runner(spec or multi_topic_spec(), PROFILES, deps, saver)


async def _events(runner: Any, message: str = "question", thread: ThreadId | None = None) -> Any:
    return [e async for e in runner.astream(make_input(message, thread))]


async def _state(model: BaseChatModel, saver: InMemorySaver, spec: AgentSpec | None = None) -> Any:
    deps = RagChatDeps(chat_models_for(model), FakeRetrieval())
    graph = build_graph(spec or multi_topic_spec(), PROFILES, deps, saver)
    snapshot = await graph.aget_state({"configurable": {"thread_id": "t-1"}})
    return snapshot.values


class TestFanOut:
    async def test_two_topics_run_with_their_own_profile_and_filters(self) -> None:
        script = Script([["alpha", "beta"]])
        retrieval = FakeRetrieval(respond=_by_profile(alpha=("a1", "a2"), beta=("b1", "b2")))
        model = make_model(script, ["Answer [1][2]."])

        events = await _events(_runner(model, retrieval))

        calls = {c.profile.name: c for c in retrieval.calls}
        assert set(calls) == {"alpha", "beta"}
        assert len(retrieval.calls) == 2  # one search per topic, no relaxation needed
        assert calls["alpha"].extra_filter == Filter.eq("kind", "memo")
        assert calls["beta"].extra_filter == Filter.eq("num", 7)
        assert {c.tenant_id for c in retrieval.calls} == {TENANT}
        assert {c.query for c in retrieval.calls} == {"question"}
        # The router ran once, on the ROUTER model path, and the topics were extracted per topic.
        assert script.route_calls == 1
        started = [e.node for e in events if isinstance(e, Step) and e.status == "started"]
        # Send repeats the node names once per topic; the order of the branches is not fixed.
        assert started[:2] == ["analyze", "route"]
        assert sorted(started[2:]) == sorted(
            ["topic"] * 2
            + ["topic/extract_filters"] * 2
            + ["topic/retrieve"] * 2
            + ["merge", "synthesize"]
        )
        assert started[-2:] == ["merge", "synthesize"]

    async def test_sources_are_one_event_and_align_with_the_synthesis_numbering(self) -> None:
        script = Script([["alpha", "beta"]])
        retrieval = FakeRetrieval(respond=_by_profile(alpha=("a1", "a2"), beta=("b1", "b2")))
        model = make_model(script, ["Answer."])

        events = await _events(_runner(model, retrieval))

        sources = [e for e in events if isinstance(e, Sources)]
        assert len(sources) == 1
        assert [r.source for r in sources[0].items] == [
            "file:///a1",
            "file:///b1",
            "file:///a2",
            "file:///b2",
        ]
        block = str(model.seen[0][1].content)
        for number, name in enumerate(("a1", "b1", "a2", "b2"), start=1):
            assert f"[{number}] {name}" in block

    async def test_the_router_order_decides_the_interleaving(self) -> None:
        script = Script([["beta", "alpha"]])
        retrieval = FakeRetrieval(respond=_by_profile(alpha=("a1",), beta=("b1",)))

        events = await _events(_runner(make_model(script, ["ok"]), retrieval))

        (sources,) = [e for e in events if isinstance(e, Sources)]
        assert [r.source for r in sources.items] == ["file:///b1", "file:///a1"]

    async def test_a_degraded_router_searches_the_fallback_topics(self) -> None:
        def garbage(schema: Any, messages: list[Any]) -> Any:
            if schema.__name__ == "TopicRoute":
                return "not json"
            return Script([["alpha"]])(schema, messages)

        model = ScriptedChatModel(responder=garbage, replies=["ok"])
        retrieval = FakeRetrieval(respond=_by_profile(alpha=("a1",), beta=("b1",)))

        await _events(_runner(model, retrieval))

        assert {c.profile.name for c in retrieval.calls} == {"alpha", "beta"}

    async def test_max_topics_caps_the_fan_out(self) -> None:
        script = Script([["beta", "alpha"]])
        retrieval = FakeRetrieval(respond=_by_profile(alpha=("a1",), beta=("b1",)))

        await _events(
            _runner(make_model(script, ["ok"]), retrieval, multi_topic_spec(max_topics=1))
        )

        assert [c.profile.name for c in retrieval.calls] == ["beta"]

    async def test_a_single_topic_agent_never_calls_the_router_model(self) -> None:
        spec = parse_agent_spec(
            {
                **multi_topic_spec().model_dump(),
                "topics": [multi_topic_spec().model_dump()["topics"][0]],
            },
            origin="t",
        )
        retrieval = FakeRetrieval(respond=_by_profile(alpha=("a1",)))
        model = make_model(Script([]), ["ok"])  # a router call would raise IndexError

        events = await _events(
            _runner(model, retrieval, spec, router=RaisingModel(error=rate_limit_error()))
        )

        assert [c.profile.name for c in retrieval.calls] == ["alpha"]
        assert any(isinstance(e, Token) for e in events)
        assert "route" in {e.node for e in events if isinstance(e, Step)}

    async def test_private_topic_state_does_not_reach_the_parent(self) -> None:
        saver = InMemorySaver()
        script = Script([["alpha", "beta"]])
        retrieval = FakeRetrieval(respond=_by_profile(alpha=("a1",), beta=("b1",)))
        model = make_model(script, ["ok"])

        await _events(_runner(model, retrieval, saver=saver), thread=ThreadId("t-1"))

        values = await _state(model, saver)
        assert "filters" not in values
        assert "topic" not in values
        assert values["routed_topics"] == [
            {"topic": "alpha", "confidence": 0.9},
            {"topic": "beta", "confidence": 0.9},
        ]
        assert sorted(r["topic"] for r in values["topic_results"]) == ["alpha", "beta"]
        assert {r["filter_level"] for r in values["topic_results"]} == {"full"}


class TestResultsDoNotLeak:
    async def test_the_second_turn_of_a_thread_starts_from_empty_results(self) -> None:
        saver = InMemorySaver()
        script = Script([["alpha", "beta"], ["alpha"]])
        retrieval = FakeRetrieval(respond=_by_profile(alpha=("a1",), beta=("b1",)))
        model = make_model(script, ["first", "second"])
        runner = _runner(model, retrieval, saver=saver)
        thread = ThreadId("t-1")

        await _events(runner, "q1", thread)
        events = await _events(runner, "q2", thread)

        values = await _state(model, saver)
        assert [r["topic"] for r in values["topic_results"]] == ["alpha"]  # beta's is gone
        assert [r["topic"] for r in values["routed_topics"]] == ["alpha"]
        assert [s["source"] for s in values["sources"]] == ["file:///a1"]
        (sources,) = [e for e in events if isinstance(e, Sources)]
        assert [r.source for r in sources.items] == ["file:///a1"]
        assert values["attempts"] == 0

    async def test_a_rewrite_replaces_the_topic_results_but_keeps_the_previous_evidence(
        self,
    ) -> None:
        saver = InMemorySaver()
        script = Script([["alpha"], ["alpha", "beta"]], rewrites=["broader query"])
        retrieval = FakeRetrieval(
            respond=by_query(
                question={"alpha": ("a-old",)},  # attempt 1: one source, fewer than min_sources 2
                broader_query={"alpha": ("a-new",), "beta": ("b-new1", "b-new2")},
            )
        )
        model = make_model(script, ["ok"])
        runner = _runner(model, retrieval, multi_topic_spec(min_sources=2), saver)

        events = await _events(runner, thread=ThreadId("t-1"))

        values = await _state(model, saver, multi_topic_spec(min_sources=2))
        # `topic_results` are the attempt-2 ones only (no a-old entry): the reset worked.
        assert sorted(r["topic"] for r in values["topic_results"]) == ["alpha", "beta"]
        assert "file:///a-old" not in [
            s["source"] for r in values["topic_results"] for s in r["sources"]
        ]
        # The attempt-1 evidence is KEPT, as the last group of the round-robin.
        assert [s["source"] for s in values["sources"]] == [
            "file:///a-new",
            "file:///b-new1",
            "file:///a-old",
            "file:///b-new2",
        ]
        assert values["attempts"] == 1
        assert values["query"] == "broader query"
        # ONE Sources event, the final list (attempt 1 stayed silent).
        (sources,) = [e for e in events if isinstance(e, Sources)]
        assert [r.source for r in sources.items] == [s["source"] for s in values["sources"]]


class TestSufficiencyLoop:
    def _steps(self, events: list[Any]) -> list[str]:
        return [e.node for e in events if isinstance(e, Step) and e.status == "started"]

    async def test_empty_then_rewrite_then_found_then_synthesize(self) -> None:
        script = Script([["alpha"], ["alpha"]], rewrites=["broader query"])
        retrieval = FakeRetrieval(respond=by_query(broader_query={"alpha": ("a1",)}))
        model = make_model(script, ["Found it [1]."])

        events = await _events(_runner(model, retrieval))

        steps = self._steps(events)
        assert [s for s in steps if "/" not in s and s != "topic"] == [
            "analyze",
            "route",
            "merge",
            "rewrite",
            "route",
            "merge",
            "synthesize",
        ]
        assert [c.query for c in retrieval.calls if c.profile.name == "alpha"][-1] == (
            "broader query"
        )
        assert "".join(e.text for e in events if isinstance(e, Token)) == "Found it [1]."
        assert len([e for e in events if isinstance(e, Sources)]) == 1
        rewrite_prompt = next(
            str(seen[-1].content)
            for seen in model.structured_seen
            if "<previous_query>" in str(seen[-1].content)
        )
        assert "<question>question</question>" in rewrite_prompt
        assert "<previous_query>question</previous_query>" in rewrite_prompt

    async def test_max_rewrites_is_respected(self) -> None:
        script = Script([["alpha"]])
        retrieval = FakeRetrieval()  # never finds anything
        model = make_model(script, ["I do not know."])

        events = await _events(_runner(model, retrieval))

        assert self._steps(events).count("rewrite") == 1  # default max_rewrites = 1
        assert script.route_calls == 2
        assert not [e for e in events if isinstance(e, Sources)]
        assert "".join(e.text for e in events if isinstance(e, Token)) == "I do not know."
        assert isinstance(events[-1], RunCompleted)

    async def test_two_rewrites_when_configured_and_none_when_disabled(self) -> None:
        for max_rewrites, expected in ((2, 2), (0, 0)):
            script = Script([["alpha"]], rewrites=["one", "two"])
            model = make_model(script, ["x"])
            spec = multi_topic_spec(max_rewrites=max_rewrites)

            events = await _events(_runner(model, FakeRetrieval(), spec))

            assert self._steps(events).count("rewrite") == expected
            assert script.route_calls == expected + 1

    async def test_attempts_are_reset_for_the_next_turn(self) -> None:
        saver = InMemorySaver()
        script = Script([["alpha"]], rewrites=["one", "two"])
        model = make_model(script, ["first", "second"])
        runner = _runner(model, FakeRetrieval(), saver=saver)
        thread = ThreadId("t-1")

        first = await _events(runner, "q1", thread)
        second = await _events(runner, "q2", thread)

        # Turn 2 may rewrite again: the counter did not stay at 1.
        assert self._steps(first).count("rewrite") == 1
        assert self._steps(second).count("rewrite") == 1
        assert (await _state(model, saver))["attempts"] == 1

    async def test_a_rewriter_that_degrades_ends_the_loop_instead_of_repeating_the_search(
        self,
    ) -> None:
        def broken(schema: Any, messages: list[Any]) -> Any:
            if schema.__name__ == "QueryRewrite":
                return "garbage"
            return Script([["alpha"]])(schema, messages)

        retrieval = FakeRetrieval()
        script_model = ScriptedChatModel(responder=broken, replies=["ok"])

        events = await _events(_runner(script_model, retrieval))

        assert {c.query for c in retrieval.calls} == {"question"}
        assert len(retrieval.calls) == 2  # ONE attempt (full filter + no filter), no second one
        steps = self._steps(events)
        assert steps.count("rewrite") == 1
        assert steps.count("route") == 1
        assert steps[-1] == "synthesize"


class TestRunThroughRunAgent:
    async def test_event_order_streamed_tokens_usage_and_steps(self) -> None:
        script = Script([["alpha", "beta"]])
        retrieval = FakeRetrieval(respond=_by_profile(alpha=("a1",), beta=("b1",)))
        model = make_model(script, ["It is so [1][2]."])
        registry = AgentRegistry((_runner(model, retrieval),))
        run_agent = RunAgent(registry, ProviderErrorClassifier(), id_generator=lambda: "run-1")

        events = [e async for e in run_agent.start("multi-rag", TENANT, "question")]

        assert events[0] == RunStarted(RunId("run-1"), "multi-rag")
        kinds = [type(e) for e in events]
        assert kinds.index(Sources) < kinds.index(Token)
        assert kinds.count(Sources) == 1
        # Only the synthesis is streamed: no analyzer / router / filter / rewrite JSON.
        assert "".join(e.text for e in events if isinstance(e, Token)) == "It is so [1][2]."
        steps = {e.node for e in events if isinstance(e, Step)}
        assert steps == {
            "analyze",
            "route",
            "topic",
            "topic/extract_filters",
            "topic/retrieve",
            "merge",
            "synthesize",
        }
        done = events[-1]
        assert isinstance(done, RunCompleted)
        assert done.finish_reason == "stop"
        # analyzer + router + 2 filter extractions + synthesis, 5 tokens each.
        assert done.usage is not None
        assert done.usage.total_tokens == 25

    async def test_topic_steps_carry_the_topic_label_and_the_rest_none(self) -> None:
        script = Script([["alpha", "beta"]])
        retrieval = FakeRetrieval(respond=_by_profile(alpha=("a1",), beta=("b1",)))
        model = make_model(script, ["It is so [1][2]."])
        registry = AgentRegistry((_runner(model, retrieval),))
        run_agent = RunAgent(registry, ProviderErrorClassifier(), id_generator=lambda: "run-1")

        events = [e async for e in run_agent.start("multi-rag", TENANT, "question")]
        steps = [e for e in events if isinstance(e, Step)]

        by_node: dict[str, set[str | None]] = {}
        for step in steps:
            by_node.setdefault(step.node, set()).add(step.label)
        assert by_node["topic"] == {"alpha", "beta"}
        assert by_node["topic/retrieve"] == {"alpha", "beta"}
        assert by_node["topic/extract_filters"] == {"alpha", "beta"}
        for node in ("analyze", "route", "merge", "synthesize"):
            assert by_node[node] == {None}, node
        assert "question" not in repr(steps)

    def test_the_topic_labeler_only_returns_names_of_the_spec(self) -> None:
        label = topic_step_label(["alpha", "beta"])

        assert label("topic", {"topic": "alpha", "query": "secret"}) == "alpha"
        assert label("topic", {"topic": "free user text", "query": "x"}) is None
        assert label("topic", {"topic": 7}) is None
        assert label("topic", "not a mapping") is None
        assert label("route", {"topic": "alpha"}) is None

    async def test_the_router_role_is_checked_before_the_stream(self) -> None:
        _runner(make_model(Script([]), []), FakeRetrieval()).ensure_available()

        broken = rag_chat_runner(
            multi_topic_spec(), PROFILES, RagChatDeps(_missing_router_models(), FakeRetrieval())
        )
        with pytest.raises(AgentUnavailableError) as info:
            broken.ensure_available()
        assert info.value.code == "llm_misconfigured"

    async def test_a_single_topic_agent_does_not_need_the_router_role(self) -> None:
        single = parse_agent_spec(
            {
                **multi_topic_spec().model_dump(),
                "topics": [multi_topic_spec().model_dump()["topics"][0]],
            },
            origin="t",
        )

        rag_chat_runner(
            single, PROFILES, RagChatDeps(_missing_router_models(), FakeRetrieval())
        ).ensure_available()


def _missing_router_models() -> ChatModels:
    def builder(model_id: str, **_kwargs: Any) -> BaseChatModel:
        if model_id == ChatRole.ROUTER.value:
            raise RuntimeError("no router model")
        return ScriptedChatModel()

    config = LlmConfig(provider="openai", models={role: role.value for role in ChatRole})
    return ChatModels(config, builder=builder)


def test_the_graph_is_one_topic_node_for_every_topic() -> None:
    deps = RagChatDeps(chat_models_for(make_model(Script([]), [])), FakeRetrieval())
    graph = build_graph(multi_topic_spec(), PROFILES, deps)

    assert set(graph.nodes) == {
        "__start__",
        "analyze",
        "route",
        "topic",
        "merge",
        "rewrite",
        "synthesize",
    }


def _scored(profile: str, score: float | None, *names: str, **ranks: int) -> tuple[Any, ...]:
    """Hits carrying relevance evidence (reranker score and/or hybrid branch ranks)."""
    return tuple(replace(hit, rerank_score=score, **ranks) for hit in make_hits(profile, *names))


class TestRelevanceSufficiency:
    """Retrieval always returns top_k hits, so only the hits' own evidence can say "not enough"."""

    def _relevance_spec(self, **changes: Any) -> AgentSpec:
        return multi_topic_spec(sufficiency="relevance", min_rerank_score=0.3, **changes)

    async def test_weak_hits_trigger_a_rewrite_although_top_k_hits_came_back(self) -> None:
        script = Script([["alpha"], ["alpha"]], rewrites=["broader query"])

        def respond(call: RetrievalCall) -> tuple[Any, ...]:
            strong = call.query == "broader query"
            return _scored("alpha", 0.9 if strong else 0.02, "a1", "a2", "a3")

        retrieval = FakeRetrieval(respond=respond)
        model = make_model(script, ["ok"])

        events = await _events(_runner(model, retrieval, self._relevance_spec()))

        assert script.route_calls == 2  # the weak first attempt was not enough
        started = [e.node for e in events if isinstance(e, Step) and e.status == "started"]
        assert started.count("rewrite") == 1
        (sources,) = [e for e in events if isinstance(e, Sources)]
        # the weak attempt-1 evidence is kept behind the new hits, deduplicated by chunk
        assert [r.source for r in sources.items] == ["file:///a1", "file:///a2", "file:///a3"]

    async def test_strong_hits_do_not_rewrite(self) -> None:
        script = Script([["alpha"]])
        retrieval = FakeRetrieval(respond=lambda _c: _scored("alpha", 0.9, "a1"))

        await _events(_runner(make_model(script, ["ok"]), retrieval, self._relevance_spec()))

        assert script.route_calls == 1

    async def test_an_empty_corpus_never_rewrites(self) -> None:
        script = Script([["alpha"]])

        events = await _events(
            _runner(make_model(script, ["I do not know."]), FakeRetrieval(), self._relevance_spec())
        )

        assert script.route_calls == 1  # a broader query cannot find what is not there
        assert "rewrite" not in {e.node for e in events if isinstance(e, Step)}
        assert not [e for e in events if isinstance(e, Sources)]

    async def test_dense_only_without_a_reranker_never_rewrites(self) -> None:
        script = Script([["alpha"]])
        retrieval = FakeRetrieval(respond=_by_profile(alpha=("a1", "a2")))  # no evidence at all

        await _events(_runner(make_model(script, ["ok"]), retrieval, self._relevance_spec()))

        assert script.route_calls == 1

    async def test_hybrid_without_a_reranker_with_weak_hits_never_rewrites(self) -> None:
        script = Script([["alpha"]])
        # dense branch only, no lexical match, no reranker: "irrelevant" is not provable
        retrieval = FakeRetrieval(
            respond=lambda _c: _scored("alpha", None, "a1", "a2", dense_rank=1)
        )

        events = await _events(
            _runner(make_model(script, ["ok"]), retrieval, self._relevance_spec())
        )

        assert script.route_calls == 1
        assert "rewrite" not in {e.node for e in events if isinstance(e, Step)}
        assert len([e for e in events if isinstance(e, Sources)]) == 1

    async def test_a_failed_rewrite_still_emits_one_sources_event_matching_the_answer(
        self,
    ) -> None:
        """Insufficient attempt + unchanged query: synthesis skips the merge but still cites."""
        script = Script([["alpha"]])

        def broken(schema: Any, messages: list[Any]) -> Any:
            if schema.__name__ == "QueryRewrite":
                return "garbage"  # degraded -> query unchanged -> straight to synthesis
            return script(schema, messages)

        retrieval = FakeRetrieval(respond=lambda _c: _scored("alpha", 0.02, "a1", "a2"))
        model = ScriptedChatModel(responder=broken, replies=["answer [1][2]"])

        events = await _events(_runner(model, retrieval, self._relevance_spec()))

        started = [e.node for e in events if isinstance(e, Step) and e.status == "started"]
        assert started.count("rewrite") == 1
        assert started.count("merge") == 1  # the merge was skipped on the way to the answer
        (sources,) = [e for e in events if isinstance(e, Sources)]
        assert [r.source for r in sources.items] == ["file:///a1", "file:///a2"]
        kinds = [type(e) for e in events]
        assert kinds.index(Sources) < kinds.index(Token)
        block = str(model.seen[-1][-2].content)
        assert "[1] a1" in block
        assert "[2] a2" in block
        assert "[3]" not in block

    async def test_relevance_is_the_default_strategy(self) -> None:
        assert (
            parse_agent_spec(
                {k: v for k, v in multi_topic_spec().model_dump().items() if k != "sufficiency"},
                origin="t",
            ).sufficiency
            == "relevance"
        )


class TestWeightedMerge:
    async def test_the_primary_topic_gets_its_weight_in_the_merge(self) -> None:
        script = Script([["alpha", "beta"]])
        retrieval = FakeRetrieval(
            respond=_by_profile(alpha=("a1", "a2", "a3"), beta=("b1", "b2", "b3"))
        )

        events = await _events(
            _runner(make_model(script, ["ok"]), retrieval, multi_topic_spec(primary_weight=2))
        )

        (sources,) = [e for e in events if isinstance(e, Sources)]
        assert [r.source.rsplit("/", 1)[1] for r in sources.items] == [
            "a1",
            "a2",
            "b1",
            "a3",
            "b2",
            "b3",
        ]

    async def test_a_low_confidence_secondary_is_not_even_searched(self) -> None:
        script = Script([["alpha", "beta"]], confidence=0.4)  # below the 0.6 default
        retrieval = FakeRetrieval(respond=_by_profile(alpha=("a1",), beta=("b1",)))

        await _events(_runner(make_model(script, ["ok"]), retrieval))

        assert {c.profile.name for c in retrieval.calls} == {"alpha"}


class TestCheckpointsAndLimits:
    async def test_topic_branches_write_no_checkpoints_of_their_own(self) -> None:
        saver = InMemorySaver()
        script = Script([["alpha", "beta"]])
        retrieval = FakeRetrieval(respond=_by_profile(alpha=("a1",), beta=("b1",)))

        await _events(
            _runner(make_model(script, ["ok"]), retrieval, saver=saver), thread=ThreadId("t-1")
        )

        # Only the parent's namespace: no `topic:<task id>` checkpoints per branch.
        assert set(saver.storage["t-1"]) == {""}

    async def test_the_longest_loop_fits_the_recursion_limit_with_headroom_for_3_5c(self) -> None:
        """max_rewrites at its bound: analyze, then (route, topic, merge) x 4 attempts, 3 rewrites
        and the synthesis = 17 supersteps, which needs `recursion_limit` 18 (measured). Of the
        default 25, 7 stay free for the clarification step of 3.5c and its resume; the test
        keeps at least 5."""
        spec = multi_topic_spec(max_rewrites=MAX_REWRITES_LIMIT)
        script = Script([["alpha"]], rewrites=["r1", "r2", "r3"])
        model = make_model(script, ["I do not know."])
        deps = RagChatDeps(chat_models_for(model), FakeRetrieval())
        graph = build_graph(spec, PROFILES, deps)
        headroom = 5
        config: Any = {"recursion_limit": DEFAULT_RECURSION_LIMIT - headroom}

        result = await graph.ainvoke(
            {"messages": [HumanMessage("q")]},
            config,
            context=AgentContext(TENANT, RunId("run-1")),
        )

        assert result["attempts"] == MAX_REWRITES_LIMIT
        assert script.route_calls == MAX_REWRITES_LIMIT + 1


class TestBranchFailure:
    async def test_a_failing_topic_branch_ends_the_run_with_one_error_and_no_sources(self) -> None:
        def respond(call: RetrievalCall) -> tuple[Any, ...]:
            if call.profile.name == "beta":
                raise TeiUnavailableError("embeddings down")
            return make_hits("alpha", "a1")

        script = Script([["alpha", "beta"]])
        runner = _runner(make_model(script, ["never streamed"]), FakeRetrieval(respond=respond))
        classifier = ChainedErrorClassifier(ProviderErrorClassifier(), RetrievalErrorClassifier())
        run_agent = RunAgent(AgentRegistry((runner,)), classifier, id_generator=lambda: "run-1")

        events = [e async for e in run_agent.start("multi-rag", TENANT, "question")]

        failures = [e for e in events if isinstance(e, RunFailed)]
        assert [(f.code, f.retryable) for f in failures] == [("retrieval_unavailable", True)]
        assert events[-1] is failures[0]
        assert not [e for e in events if isinstance(e, Sources | Token | RunCompleted)]
