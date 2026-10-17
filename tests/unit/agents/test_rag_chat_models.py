"""Per-agent LLM profiles: the `models:` spec field, graph wiring with mixed providers, boot."""

import logging
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel

from agintis.bootstrap.container import Container
from agintis.modules.agents.rag_chat import (
    AgentSpec,
    InvalidAgentSpecError,
    RagChatDeps,
    agent_deps,
    build_graph,
    parse_agent_spec,
    rag_chat_runner,
)
from agintis.modules.agents_catalog.api import RunInput, Token
from agintis.modules.knowledge.api import MetadataField, RagProfile, RagProfileRegistry
from agintis.platform.llm import (
    ChatModels,
    ChatRole,
    InvalidLlmProfilesError,
    LlmConfig,
    LlmProfile,
    ModelCatalog,
)
from agintis.shared_kernel.ids import RunId, TenantId
from tests.fakes.llm import ScriptedChatModel
from tests.fakes.retrieval import FakeRetrieval, make_hit
from tests.support.settings import make_settings

PROFILE = RagProfile(
    "docs",
    3,
    100,
    "simple",
    "structured",
    metadata_fields=(MetadataField("kind", "str", filterable=True),),
)
PROFILES = RagProfileRegistry([PROFILE])
BASE = {
    "id": "docs-rag",
    "description": "Answers from the docs.",
    "topics": [{"name": "docs", "description": "d", "profile": "docs"}],
}
DEFAULTS = LlmConfig(provider="openai", models={role: f"base-{role.value}" for role in ChatRole})
CATALOG = {
    "fast": LlmProfile("openai", "gpt-fast", api_key_env="FAST_KEY"),
    "smart": LlmProfile("anthropic", "claude-smart", api_key_env="SMART_KEY"),
}
ENV = {"FAST_KEY": "sk-fast", "SMART_KEY": "sk-smart"}


def _spec(**extra: Any) -> AgentSpec:
    return parse_agent_spec({**BASE, **extra}, origin="test")


# --- the spec field ---------------------------------------------------------------------------


def test_models_default_to_none_and_keep_the_spec_backward_compatible() -> None:
    assert _spec().models == {}


def test_models_map_roles_to_profile_names() -> None:
    spec = _spec(models={"analyzer": "fast", "synthesis": "smart"})
    assert spec.models == {ChatRole.ANALYZER: "fast", ChatRole.SYNTHESIS: "smart"}


@pytest.mark.parametrize(
    "models", [{"judge": "fast"}, {"analyzer": ""}, {"analyzer": 3}, ["analyzer"]]
)
def test_invalid_models_are_rejected(models: Any) -> None:
    with pytest.raises(InvalidAgentSpecError, match="models"):
        _spec(models=models)


def test_a_router_binding_warns_when_no_router_model_is_ever_called(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        _spec(models={"router": "fast"})
    assert "models.router is ignored" in caplog.text


def test_a_router_binding_is_fine_for_an_llm_router(caplog: pytest.LogCaptureFixture) -> None:
    topics = [
        {"name": "a", "description": "d", "profile": "docs"},
        {"name": "b", "description": "d", "profile": "other"},
    ]
    with caplog.at_level(logging.WARNING):
        _spec(topics=topics, router="llm", models={"router": "fast"})
    assert caplog.text == ""


# --- the graph --------------------------------------------------------------------------------


class Routed:
    """A builder serving a fake per model id; records which ids were built."""

    def __init__(self, fakes: dict[str, ScriptedChatModel]) -> None:
        self.fakes = fakes
        self.built: list[tuple[str, str]] = []

    def __call__(self, model: str, **kwargs: Any) -> BaseChatModel:
        self.built.append((model, kwargs["model_provider"]))
        return self.fakes[model]


def _chat_models(builder: Routed) -> ChatModels:
    return ChatModels(ModelCatalog(DEFAULTS, CATALOG, environ=ENV.get, builder=builder))


async def test_a_bound_agent_mixes_providers_and_nodes_never_know() -> None:
    analyzer_side = ScriptedChatModel(
        structured=[
            {
                "standalone_query": "deadline?",
                "is_clear": True,
                "confidence": 0.9,
                "request_kind": "domain",
            },
            {"kind": None},
        ]
    )
    synthesis_side = ScriptedChatModel(replies=["It is 24 months."])
    builder = Routed({"gpt-fast": analyzer_side, "claude-smart": synthesis_side})
    spec = _spec(models={"analyzer": "fast", "synthesis": "smart"})
    deps = RagChatDeps(_chat_models(builder), FakeRetrieval((make_hit("24 months", title="A"),)))

    runner = rag_chat_runner(spec, PROFILES, deps)
    run_input = RunInput(TenantId("acme"), "deadline?", RunId("0123456789abcdef0123456789abcdef"))
    events = [e async for e in runner.astream(run_input)]

    assert "".join(e.text for e in events if isinstance(e, Token)) == "It is 24 months."
    assert len(analyzer_side.structured_seen) == 2  # analyzer + filter extractor, same profile
    assert len(synthesis_side.seen) == 1
    # Only the two bound profiles were built (the evidence-free single router needs no model).
    assert sorted(builder.built) == [("claude-smart", "anthropic"), ("gpt-fast", "openai")]


def test_an_unbound_agent_keeps_using_the_platform_defaults() -> None:
    builder = Routed({f"base-{r.value}": ScriptedChatModel() for r in ChatRole})
    deps = RagChatDeps(_chat_models(builder), FakeRetrieval())

    build_graph(_spec(), PROFILES, deps)
    rag_chat_runner(_spec(), PROFILES, deps).ensure_available()

    assert {model for model, _ in builder.built} <= {f"base-{r.value}" for r in ChatRole}
    assert "gpt-fast" not in {model for model, _ in builder.built}


def test_an_unknown_profile_fails_at_wiring_naming_the_profile_and_the_agent() -> None:
    deps = RagChatDeps(_chat_models(Routed({})), FakeRetrieval())

    with pytest.raises(InvalidAgentSpecError, match=r"'nope'.*'docs-rag'"):
        rag_chat_runner(_spec(models={"synthesis": "nope"}), PROFILES, deps)
    with pytest.raises(InvalidAgentSpecError, match="'nope'"):
        build_graph(_spec(models={"synthesis": "nope"}), PROFILES, deps)


def test_a_binding_for_a_role_the_agent_never_calls_is_never_resolved_or_built() -> None:
    builder = Routed({f"base-{r.value}": ScriptedChatModel() for r in ChatRole})
    deps = RagChatDeps(_chat_models(builder), FakeRetrieval())
    # `router` is unused by a single-topic agent: an unknown profile there must not matter.
    spec = _spec(models={"router": "nope"})

    runner = rag_chat_runner(spec, PROFILES, deps)
    runner.ensure_available()
    deps.chat_models.verify()

    assert "base-router" in {m for m, _ in builder.built}  # only the root checks the defaults
    assert all(m != "nope" for m, _ in builder.built)


def test_the_views_of_two_agents_are_registered_once_per_distinct_bindings() -> None:
    deps = RagChatDeps(_chat_models(Routed({})), FakeRetrieval())
    spec = _spec(models={"analyzer": "fast"})

    first, second = agent_deps(spec, deps), agent_deps(spec, deps)

    assert first.chat_models is second.chat_models
    assert len(deps.chat_models.catalog.views) == 1


def test_availability_reports_the_agents_own_models() -> None:
    builder = Routed({"gpt-fast": ScriptedChatModel(), "claude-smart": ScriptedChatModel()})
    deps = RagChatDeps(_chat_models(builder), FakeRetrieval())
    spec = _spec(models={"analyzer": "fast", "synthesis": "smart"})

    rag_chat_runner(spec, PROFILES, deps).ensure_available()

    assert sorted(builder.built) == [("claude-smart", "anthropic"), ("gpt-fast", "openai")]


# --- the container ----------------------------------------------------------------------------


class TestContainerWiring:
    PROFILE_YAML = """\
key: docs
schema_version: 1
top_k: 3
chunk_max_tokens: 200
parser_key: simple
chunker_key: structured
metadata_fields:
  - {name: kind, type: str, source: 'const:memo', filterable: true}
"""
    LLM_YAML = """\
profiles:
  fast: {provider: openai, model: gpt-fast, api_key_env: FAST_KEY}
  smart: {provider: anthropic, model: claude-smart, api_key_env: SMART_KEY}
"""

    def _settings(self, tmp_path: Path, binding: str, llm_yaml: str | None = None) -> Any:
        profiles_dir, agents_dir = tmp_path / "profiles", tmp_path / "agents"
        profiles_dir.mkdir()
        agents_dir.mkdir()
        (profiles_dir / "docs.yaml").write_text(self.PROFILE_YAML, encoding="utf-8")
        (agents_dir / "docs.yaml").write_text(
            "id: docs-rag\ndescription: Docs agent.\n"
            "topics: [{name: docs, description: d, profile: docs, filters: {kind: hint}}]\n"
            + binding,
            encoding="utf-8",
        )
        llm: dict[str, Any] = {"provider": "openai", "model": "base", "api_key": "sk-base"}
        if llm_yaml is not None:
            path = tmp_path / "llm_profiles.yaml"
            path.write_text(llm_yaml, encoding="utf-8")
            llm["profiles_file"] = path
        return make_settings(
            knowledge={"profiles_dir": profiles_dir}, agents={"dir": agents_dir}, llm=llm
        )

    def test_the_profiles_file_feeds_the_catalog_and_the_agent_binds_it(
        self, tmp_path: Path
    ) -> None:
        settings = self._settings(tmp_path, "models: {synthesis: fast}\n", self.LLM_YAML)

        container = Container.build(settings, environ=ENV.get)

        catalog = container.chat_models.catalog
        assert catalog.profile_names == ["fast", "smart"]
        (bound,) = container.chat_models.catalog.views.values()
        assert bound.describe(ChatRole.SYNTHESIS).label == "fast (openai:gpt-fast)"
        assert bound.describe(ChatRole.ANALYZER).label == "default (openai:base)"

    def test_without_a_profiles_file_nothing_changes(self, tmp_path: Path) -> None:
        container = Container.build(self._settings(tmp_path, ""))

        assert container.chat_models.catalog.profile_names == []
        (view,) = container.chat_models.catalog.views.values()  # the agent's default view
        assert view.describe(ChatRole.SYNTHESIS).label == "default (openai:base)"

    def test_a_binding_without_a_catalog_aborts_the_boot_naming_agent_and_profile(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(InvalidAgentSpecError, match=r"'fast'.*'docs-rag'"):
            Container.build(self._settings(tmp_path, "models: {synthesis: fast}\n"))

    def test_an_unset_key_variable_aborts_the_boot(self, tmp_path: Path) -> None:
        settings = self._settings(tmp_path, "models: {synthesis: fast}\n", self.LLM_YAML)
        with pytest.raises(
            InvalidAgentSpecError, match=r"profile 'fast' of agent 'docs-rag'.*FAST_KEY"
        ):
            Container.build(settings, environ=lambda _name: None)

    def test_an_unreferenced_profile_needs_no_key(self, tmp_path: Path) -> None:
        settings = self._settings(tmp_path, "", self.LLM_YAML)
        Container.build(settings, environ=lambda _name: None)

    def test_isolated_settings_never_read_the_dotenv_for_keys(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / ".env").write_text("FAST_KEY=from-dotenv\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("FAST_KEY", raising=False)
        settings = self._settings(tmp_path, "models: {synthesis: fast}\n", self.LLM_YAML)

        with pytest.raises(InvalidAgentSpecError, match="FAST_KEY"):
            Container.build(settings)

    def test_a_broken_profiles_file_aborts_the_boot(self, tmp_path: Path) -> None:
        settings = self._settings(tmp_path, "", "profiles: {a: {provider: openai}}")
        with pytest.raises(InvalidLlmProfilesError, match="model"):
            Container.build(settings)
