"""Tracing port: backend-agnostic, OpenTelemetry behind an OTLP exporter (adapter in `otlp.py`).

This module imports NOTHING from opentelemetry/openinference: with tracing off (`none`, the
default) no tracing library is ever imported. The OTLP adapter is imported lazily by
`build_tracing` and needs the `otel` uv extra.

Our port must be the ONLY tracing path. LangChain also traces by itself when
`LANGSMITH_TRACING` / `LANGCHAIN_TRACING_V2` are truthy in the process environment, which would
ship content to LangSmith around the port (and around `hide_content`): `reject_foreign_tracing`
refuses to boot in that case. LangSmith can still receive traces through the OTLP adapter.

The guard is STRICT ON PURPOSE (fail closed): it reads the process environment only, treats
1/true/yes/on (any case, surrounding spaces ignored) as "on" and checks every variable on its own.
It does not mirror LangChain's own parsing (which varies across versions), so it also refuses
"1", "yes" or "True", and it refuses `LANGSMITH_TRACING=true` even when a V2 variable says
false. A false positive costs a boot error with a clear message; a false negative would send
content to a third party around `hide_content`.
"""

import os
from abc import ABC, abstractmethod
from collections.abc import Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field

# Environment variables that switch LangChain's own (LangSmith) tracing on.
FOREIGN_TRACING_VARIABLES = (
    "LANGSMITH_TRACING",
    "LANGSMITH_TRACING_V2",
    "LANGCHAIN_TRACING_V2",
    "LANGCHAIN_TRACING",
)
_TRUTHY = frozenset({"1", "true", "yes", "on"})

OTEL_EXTRA = "otel"


class TracingError(RuntimeError):
    """Tracing is misconfigured: the app must not start."""


class TracingNotInstalledError(TracingError):
    """`OBSERVABILITY_TRACING=otlp` without the `otel` extra installed."""


class ForeignTracingError(TracingError):
    """LangChain's own tracing is enabled through the environment."""


def reject_foreign_tracing(environ: Mapping[str, str] | None = None) -> None:
    """Fail fast when LangChain's built-in tracing is on (reads the PROCESS environment only)."""
    env = os.environ if environ is None else environ
    active = [
        name for name in FOREIGN_TRACING_VARIABLES if env.get(name, "").strip().lower() in _TRUTHY
    ]
    if active:
        raise ForeignTracingError(
            f"{', '.join(active)} enables LangChain's own tracing, which would bypass the "
            "observability port (and OBSERVABILITY_HIDE_CONTENT). Unset it and configure "
            "OBSERVABILITY_TRACING=otlp instead (LangSmith accepts OTLP traces)."
        )


@dataclass(frozen=True, slots=True)
class RunAttributes:
    """What every span of one run carries (primitives: no domain types)."""

    tenant_id: str
    agent_id: str
    run_id: str
    # The conversation (thread id) when the run belongs to one; groups runs into a session.
    session_id: str | None = None


@dataclass(frozen=True, slots=True)
class OtlpConfig:
    """Built from primitives (no Settings here)."""

    # Full OTLP/HTTP traces URL, POSTed to verbatim (e.g. http://localhost:6006/v1/traces).
    endpoint: str
    service_name: str = "agintis"
    headers: Mapping[str, str] = field(default_factory=dict, repr=False)
    # Redact prompts, completions, retrieved texts and tool I/O (span attributes) and the text of
    # errors (status description, exception message and stack trace) in every span.
    hide_content: bool = False
    # Per-request timeout of the OTLP exporter (seconds); also bounds retries of one export.
    timeout_seconds: float = 5.0


def parse_headers(raw: str) -> dict[str, str]:
    """`name=value,name2=value2` -> dict (value may contain `=`; no percent-encoding)."""
    headers: dict[str, str] = {}
    for pair in raw.split(","):
        if not pair.strip():
            continue
        name, separator, value = pair.partition("=")
        if not separator or not name.strip():
            raise ValueError("headers must be 'name=value' pairs separated by commas")
        headers[name.strip()] = value.strip()
    return headers


class Tracing(ABC):
    """Lifecycle and per-run scope of tracing. `start`/`shutdown` are idempotent."""

    @property
    @abstractmethod
    def enabled(self) -> bool: ...

    @abstractmethod
    def start(self) -> None:
        """Install the instrumentation (no network)."""

    @abstractmethod
    def run_scope(self, attributes: RunAttributes) -> AbstractContextManager[object]:
        """Context in which every span started carries the run's attributes.

        Enter it around ONE await at a time, never across a `yield` (a context attached in one
        task cannot be detached from another one). The attributes are read when a span ENDS, so
        every await that can end spans (including closing a half-consumed stream) belongs in it.
        """

    @abstractmethod
    def force_flush(self) -> None:
        """Export what is buffered (blocking)."""

    @abstractmethod
    def shutdown(self) -> None:
        """Uninstall the instrumentation, flush and release the exporter (blocking)."""


class NoTracing(Tracing):
    """Tracing off: every operation is a no-op."""

    @property
    def enabled(self) -> bool:
        return False

    def start(self) -> None:
        return None

    def run_scope(self, attributes: RunAttributes) -> AbstractContextManager[object]:
        return nullcontext()

    def force_flush(self) -> None:
        return None

    def shutdown(self) -> None:
        return None


def build_tracing(config: OtlpConfig | None) -> Tracing:
    """`None` = tracing off; otherwise the OTLP adapter (imported here, needs the extra)."""
    if config is None:
        return NoTracing()
    try:
        from agintis.platform.observability.otlp import OtlpTracing
    except ImportError as exc:
        raise TracingNotInstalledError(
            "OBSERVABILITY_TRACING=otlp needs the OpenTelemetry packages: install the "
            f"'{OTEL_EXTRA}' extra (uv sync --extra {OTEL_EXTRA}; Docker: "
            f'--build-arg UV_EXTRAS="... {OTEL_EXTRA}"). Missing: {exc.name}'
        ) from exc
    return OtlpTracing(config)
