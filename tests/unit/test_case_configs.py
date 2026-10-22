"""Every shipped case (case/<name>/{profiles,agents}) must load: a broken case fails CI."""

import json
from pathlib import Path

import pytest

from agintis.bootstrap.container import Container
from agintis.bootstrap.settings import Settings
from agintis.platform.llm import load_llm_profiles
from agintis.platform.yaml_config import yaml_files
from tests.support.postgres import PROJECT_ROOT
from tests.support.settings import make_settings

# A case may rerank (`reranker_key: tei`), which is only registered with a reranker URL; the
# compose app always sets one. Building is network-free: the URL is never called here.
RERANKER_URL = "http://reranker:80"

CASES = sorted(
    path.parent
    for path in (PROJECT_ROOT / "case").glob("*/profiles")
    if path.is_dir() and (path.parent / "agents").is_dir()
)


def case_settings(case: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """The case's profiles and agents, plus its named LLM profiles when it ships some (with a
    dummy value for each key variable they name: building never calls a model)."""
    llm: dict[str, object] = {}
    profiles_file = case / "llm_profiles.yaml"
    if profiles_file.is_file():
        llm["profiles_file"] = profiles_file
        for profile in load_llm_profiles(profiles_file).values():
            if profile.api_key_env:
                monkeypatch.setenv(profile.api_key_env, "test-key")
    return make_settings(
        knowledge={"profiles_dir": case / "profiles"},
        agents={"dir": case / "agents"},
        reranker={"url": RERANKER_URL},
        **({"llm": llm} if llm else {}),
    )


def test_cases_are_discovered() -> None:
    assert CASES  # guards the glob itself


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_case_profiles_and_agents_load(case: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Container.build is network-free: it loads the profiles and validates every agent spec
    # against them (unknown profile, filter field the profile does not offer, router knobs).
    container = Container.build(case_settings(case, monkeypatch))
    # Exact counts: a lost profile/agent file (or one silently skipped) must fail. The built-ins
    # are the default profile and the `direct_chat` agent, so subtract them.
    assert len(container.profiles.all()) == len(yaml_files(case / "profiles")) + 1
    assert len(container.agents_catalog.registry.list()) == len(yaml_files(case / "agents")) + 1


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_eval_questions_name_loaded_profiles(case: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    questions = case / "profiles" / "eval_questions.jsonl"
    if not questions.is_file():
        pytest.skip("no eval questions")
    container = Container.build(case_settings(case, monkeypatch))
    known = {profile.name for profile in container.profiles.all()}
    # Only the `profile` field is read: the questions are held out.
    used = {json.loads(line)["profile"] for line in questions.read_text().splitlines() if line}
    assert used <= known
