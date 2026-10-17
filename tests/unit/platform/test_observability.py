import subprocess
import sys
from typing import Any

import pytest
from pydantic import ValidationError

from agintis.platform.observability import (
    ForeignTracingError,
    NoTracing,
    OtlpConfig,
    RunAttributes,
    TracingNotInstalledError,
    build_tracing,
    parse_headers,
    reject_foreign_tracing,
)
from tests.support.settings import make_settings


class TestForeignTracingGuard:
    @pytest.mark.parametrize(
        "name", ["LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING_V2"]
    )
    @pytest.mark.parametrize("value", ["true", "True", "1", " yes ", "on"])
    def test_truthy_values_are_refused(self, name: str, value: str) -> None:
        with pytest.raises(ForeignTracingError, match=name):
            reject_foreign_tracing({name: value})

    @pytest.mark.parametrize("value", ["", "false", "0", "no", "off"])
    def test_falsy_values_pass(self, value: str) -> None:
        reject_foreign_tracing({"LANGSMITH_TRACING": value, "LANGCHAIN_TRACING_V2": value})

    def test_reads_the_process_environment_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LANGSMITH_TRACING", "true")
        with pytest.raises(ForeignTracingError):
            reject_foreign_tracing()

    def test_the_message_points_at_the_port(self) -> None:
        with pytest.raises(ForeignTracingError, match="OBSERVABILITY_TRACING=otlp"):
            reject_foreign_tracing({"LANGCHAIN_TRACING_V2": "true"})


class TestForeignTracingGuardIsStrict:
    def test_a_v1_var_is_refused_even_if_a_v2_var_says_false(self) -> None:
        with pytest.raises(ForeignTracingError, match="LANGSMITH_TRACING"):
            reject_foreign_tracing({"LANGSMITH_TRACING": "true", "LANGCHAIN_TRACING_V2": "false"})

    def test_all_active_variables_are_named(self) -> None:
        with pytest.raises(ForeignTracingError) as excinfo:
            reject_foreign_tracing({"LANGSMITH_TRACING": "1", "LANGCHAIN_TRACING": "yes"})
        assert "LANGSMITH_TRACING" in str(excinfo.value)
        assert "LANGCHAIN_TRACING" in str(excinfo.value)


class TestParseHeaders:
    def test_pairs(self) -> None:
        assert parse_headers("a=1, b = two ,") == {"a": "1", "b": "two"}

    def test_value_keeps_equals_signs(self) -> None:
        assert parse_headers("Authorization=Basic abc==") == {"Authorization": "Basic abc=="}

    @pytest.mark.parametrize("raw", ["novalue", "=x", "a=1,broken"])
    def test_malformed_pairs_are_rejected(self, raw: str) -> None:
        with pytest.raises(ValueError, match="name=value"):
            parse_headers(raw)


class TestBuildTracing:
    def test_none_config_is_a_noop(self) -> None:
        tracing = build_tracing(None)
        assert isinstance(tracing, NoTracing)
        assert not tracing.enabled
        tracing.start()
        with tracing.run_scope(RunAttributes("t", "a", "r")):
            pass
        tracing.force_flush()
        tracing.shutdown()

    def test_missing_extra_is_a_clear_boot_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # `None` in sys.modules makes the import fail like an uninstalled dependency.
        monkeypatch.setitem(sys.modules, "agintis.platform.observability.otlp", None)
        with pytest.raises(TracingNotInstalledError, match="'otel' extra"):
            build_tracing(OtlpConfig(endpoint="http://localhost:6006/v1/traces"))

    def test_tracing_off_imports_no_tracing_library(self) -> None:
        # Fresh interpreter: other tests import the adapter. (langgraph/fastapi import the
        # opentelemetry API themselves, so the check is on our adapter and on openinference.)
        script = (
            "import sys\n"
            "import agintis.bootstrap.container, agintis.main\n"
            "from agintis.platform.observability import build_tracing\n"
            "build_tracing(None)\n"
            "bad = [m for m in sys.modules if m.split('.')[0] == 'openinference'"
            " or m == 'agintis.platform.observability.otlp']\n"
            "assert not bad, bad\n"
        )
        result = subprocess.run(
            [sys.executable, "-I", "-c", script], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr


class TestObservabilitySettings:
    def test_default_is_off(self) -> None:
        obs = make_settings().observability
        assert obs.tracing == "none"
        assert obs.otlp_endpoint is None
        assert not obs.hide_content

    def test_otlp_needs_an_endpoint(self) -> None:
        with pytest.raises(ValidationError, match="OTLP_ENDPOINT"):
            make_settings(observability={"tracing": "otlp"})

    def test_unknown_mode_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            make_settings(observability={"tracing": "langsmith"})

    def test_malformed_headers_fail_at_boot(self) -> None:
        with pytest.raises(ValidationError, match="name=value"):
            make_settings(
                observability={
                    "tracing": "otlp",
                    "otlp_endpoint": "http://localhost:6006/v1/traces",
                    "otlp_headers": "oops",
                }
            )

    def test_empty_strings_mean_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OBSERVABILITY_OTLP_ENDPOINT", "")
        monkeypatch.setenv("OBSERVABILITY_OTLP_HEADERS", "")
        obs = make_settings().observability
        assert obs.otlp_endpoint is None
        assert obs.otlp_headers is None

    def test_timeout_defaults_to_five_seconds_and_must_be_positive(self) -> None:
        assert make_settings().observability.otlp_timeout_seconds == 5
        with pytest.raises(ValidationError):
            make_settings(observability={"otlp_timeout_seconds": 0})

    def test_endpoint_is_kept_verbatim(self) -> None:
        settings: Any = make_settings(
            observability={
                "tracing": "otlp",
                "otlp_endpoint": "http://localhost:6006/v1/traces",
                "otlp_headers": "Authorization=Bearer x",
            }
        )
        assert settings.observability.otlp_endpoint == "http://localhost:6006/v1/traces"
        assert "Bearer" not in repr(settings.observability)
