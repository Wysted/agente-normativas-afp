"""Where a profile's `api_key_env` is looked up: the process environment, then `.env`."""

from collections.abc import Iterator
from pathlib import Path

import pytest

from agintis.bootstrap import llm as llm_module
from agintis.bootstrap.llm import process_and_dotenv_env


@pytest.fixture(autouse=True)
def _fresh_dotenv_cache() -> Iterator[None]:
    llm_module._dotenv_values.cache_clear()
    yield
    llm_module._dotenv_values.cache_clear()


def test_the_process_environment_wins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("JUDGE_KEY=from-file\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("JUDGE_KEY", "from-process")

    assert process_and_dotenv_env("JUDGE_KEY") == "from-process"


def test_a_key_kept_only_in_dotenv_is_found(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / ".env").write_text("JUDGE_KEY=from-file\nEMPTY=\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("JUDGE_KEY", raising=False)
    monkeypatch.delenv("EMPTY", raising=False)

    assert process_and_dotenv_env("JUDGE_KEY") == "from-file"
    assert process_and_dotenv_env("EMPTY") is None
    assert process_and_dotenv_env("MISSING") is None


def test_no_dotenv_file_is_fine(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("JUDGE_KEY", raising=False)

    assert process_and_dotenv_env("JUDGE_KEY") is None


def test_a_defined_but_empty_variable_never_falls_through_to_dotenv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / ".env").write_text("JUDGE_KEY=from-file\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("JUDGE_KEY", "")

    assert process_and_dotenv_env("JUDGE_KEY") is None
