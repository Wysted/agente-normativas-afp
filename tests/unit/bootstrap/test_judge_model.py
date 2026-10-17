"""The judge model: a named LLM profile resolved through the catalog (no new ChatRole), the
`deepseek` provider and the EVAL_* settings."""

from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from agintis.agent_kit.nodes.structured import bind_structured
from agintis.bootstrap.llm import JUDGE_AGENT, bind_judge_model, build_chat_models
from agintis.bootstrap.observability import build_tracing_from_settings
from agintis.bootstrap.settings import ENV_PREFIXES
from agintis.platform.llm import (
    PROVIDER_EXTRAS,
    ChatModels,
    ChatRole,
    LlmError,
    LlmNotConfiguredError,
    UnknownProfileError,
)
from tests.support.settings import make_settings

PROFILES = """\
profiles:
  judge:
    provider: deepseek
    model: deepseek-chat
    api_key_env: DEEPSEEK_API_KEY
    temperature: 0
    timeout_seconds: 60
  other:
    provider: openai
    model: gpt-test
    api_key_env: OTHER_KEY
"""


class Schema(BaseModel):
    answer: str


def _models(tmp_path: Path, **llm: object) -> ChatModels:
    path = tmp_path / "profiles.yaml"
    path.write_text(PROFILES, encoding="utf-8")
    settings = make_settings(llm={"profiles_file": path, **llm})
    return build_chat_models(settings, {"DEEPSEEK_API_KEY": "sk-test", "OTHER_KEY": "k"}.get)


def test_the_judge_view_resolves_the_profile_without_a_judge_role(tmp_path: Path) -> None:
    chat_models = _models(tmp_path)
    view = bind_judge_model(chat_models, "judge")

    resolved = view.describe(ChatRole.ANALYZER)
    assert resolved.label == "judge (deepseek:deepseek-chat)"
    assert {r.value for r in ChatRole} == {"router", "analyzer", "synthesis"}  # no new role
    assert JUDGE_AGENT == "answer-judge"


def test_default_means_the_llm_defaults_analyzer_model(tmp_path: Path) -> None:
    chat_models = _models(tmp_path, provider="openai", model="gpt-x", analyzer_model="gpt-a")
    view = bind_judge_model(chat_models, "default")
    assert view.describe(ChatRole.ANALYZER).label == "default (openai:gpt-a)"


def test_default_without_a_configured_llm_fails_at_verify(tmp_path: Path) -> None:
    view = bind_judge_model(_models(tmp_path), "default")
    with pytest.raises(LlmNotConfiguredError):
        view.verify()


def test_an_unknown_profile_fails_at_bind(tmp_path: Path) -> None:
    with pytest.raises(UnknownProfileError, match="nope"):
        bind_judge_model(_models(tmp_path), "nope")


def test_an_unset_key_variable_fails_at_bind(tmp_path: Path) -> None:
    path = tmp_path / "profiles.yaml"
    path.write_text(PROFILES, encoding="utf-8")
    chat_models = build_chat_models(make_settings(llm={"profiles_file": path}), lambda _name: None)
    with pytest.raises(LlmError, match="DEEPSEEK_API_KEY"):
        bind_judge_model(chat_models, "judge")


def test_the_deepseek_provider_has_its_uv_extra() -> None:
    assert PROVIDER_EXTRAS["deepseek"] == "deepseek"


def test_deepseek_structured_output_uses_function_calling_not_json_schema(tmp_path: Path) -> None:
    """DeepSeek's chat completions API accepts only the `text` and `json_object` response formats
    (https://api-docs.deepseek.com/api/create-chat-completion), so the `json_schema` request that
    `ChatOpenAI.with_structured_output` sends by default is a 400. `ChatDeepSeek` maps it to
    function calling."""
    pytest.importorskip("langchain_deepseek")
    from langchain_openai import ChatOpenAI

    view = bind_judge_model(_models(tmp_path), "judge")
    view.verify()  # offline structured-output check of the judge profile
    deepseek = view.get(ChatRole.ANALYZER)
    openai_like = ChatOpenAI(model="x", api_key="k")

    sent = deepseek.with_structured_output(Schema).first.kwargs  # type: ignore[attr-defined]
    default = openai_like.with_structured_output(Schema).first.kwargs  # type: ignore[attr-defined]
    assert type(deepseek).__name__ == "ChatDeepSeek"
    assert "response_format" not in sent
    assert sent["tools"][0]["function"]["name"] == "Schema"
    assert sent["ls_structured_output_format"]["kwargs"]["method"] == "function_calling"
    assert default["ls_structured_output_format"]["kwargs"]["method"] == "json_schema"
    assert "response_format" in default


def test_the_judge_binding_keeps_the_non_streamed_copy_convention(tmp_path: Path) -> None:
    pytest.importorskip("langchain_deepseek")
    view = bind_judge_model(_models(tmp_path), "judge")
    model = view.get(ChatRole.ANALYZER)

    bind_structured(model, Schema)

    assert model.disable_streaming is not True  # bound to a copy, the cached client is untouched


# --- settings -----------------------------------------------------------------------------------


def test_eval_settings_defaults() -> None:
    settings = make_settings().eval
    assert (settings.judge_profile, settings.judge_concurrency) == (None, 4)


def test_eval_settings_read_flat_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVAL_JUDGE_PROFILE", "judge")
    monkeypatch.setenv("EVAL_JUDGE_CONCURRENCY", "8")
    settings = make_settings()
    assert settings.eval.judge_profile == "judge"
    assert settings.eval.judge_concurrency == 8


@pytest.mark.parametrize(
    "values",
    [{"judge_profile": "Bad Name"}, {"judge_concurrency": 0}, {"judge_concurrency": 65}],
)
def test_eval_settings_reject_bad_values(values: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        make_settings(eval=values)


def test_a_profile_key_variable_cannot_use_the_eval_prefix() -> None:
    assert "EVAL_" in ENV_PREFIXES


def test_tracing_defaults_to_none() -> None:
    assert not build_tracing_from_settings(make_settings()).enabled
