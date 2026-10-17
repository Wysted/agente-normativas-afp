import pytest
from pydantic import ValidationError

from agintis.bootstrap.settings import DEFAULT_MAX_MESSAGE_CHARS, HttpSettings
from tests.support.settings import make_settings


def test_defaults() -> None:
    http = HttpSettings(_env_file=None)

    assert http.default_tenant is None
    assert http.max_message_chars == DEFAULT_MAX_MESSAGE_CHARS


def test_empty_default_tenant_means_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HTTP_DEFAULT_TENANT", "")

    assert HttpSettings(_env_file=None).default_tenant is None


@pytest.mark.parametrize("value", ["a b", "../x", "-lead", "x" * 65])
def test_default_tenant_must_match_the_tenant_pattern(value: str) -> None:
    with pytest.raises(ValidationError):
        HttpSettings(_env_file=None, default_tenant=value)


def test_default_tenant_enabled_needs_explicit_local(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("APP_ENV", raising=False)
    assert make_settings().default_tenant_enabled is False
    assert make_settings(app_env="local").default_tenant_enabled is True
    assert make_settings(app_env="prod").default_tenant_enabled is False


def test_env_sourced_app_env_counts_as_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "local")

    assert make_settings().default_tenant_enabled is True
