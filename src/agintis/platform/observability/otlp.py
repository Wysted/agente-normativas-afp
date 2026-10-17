"""OTLP/HTTP tracing adapter: OpenInference LangChain instrumentation + an OpenTelemetry SDK.

Imported only by `build_tracing` when tracing is `otlp` (needs the `otel` extra).

The provider is OURS and is never installed as the global one: the instrumentor takes
`tracer_provider=` explicitly and nothing else in the process is affected. The instrumentor
itself is a process singleton that patches `BaseCallbackManager`, so a second `start()` without
a `shutdown()` in between is refused instead of silently keeping the first provider.
"""

import logging
from collections.abc import Sequence
from contextlib import AbstractContextManager
from urllib.parse import urlsplit

from openinference.instrumentation import TraceConfig, using_attributes
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import Event, ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.trace.status import Status

from agintis.platform.observability.tracing import (
    OtlpConfig,
    RunAttributes,
    Tracing,
    TracingError,
)

logger = logging.getLogger(__name__)

PROJECT_NAME = "openinference.project.name"
REDACTED = "__REDACTED__"
# Span-event attributes that carry free text (messages and stack traces of exceptions, which can
# quote prompts, retrieved passages or raw model output); `exception.type` stays.
_EXCEPTION_TEXT_KEYS = ("exception.message", "exception.stacktrace")


def _redact_event(event: Event) -> Event:
    attributes = dict(event.attributes or {})
    if not any(key in attributes for key in _EXCEPTION_TEXT_KEYS):
        return event
    for key in _EXCEPTION_TEXT_KEYS:
        if key in attributes:
            attributes[key] = REDACTED
    return Event(event.name, attributes, event.timestamp)


def _redact_span(span: ReadableSpan) -> ReadableSpan:
    """A copy of `span` without the free text of its status and exception events."""
    status = span.status
    if status.description:
        status = Status(status.status_code, REDACTED)
    return ReadableSpan(
        name=span.name,
        context=span.context,
        parent=span.parent,
        resource=span.resource,
        attributes=span.attributes,
        events=[_redact_event(event) for event in span.events],
        links=span.links,
        kind=span.kind,
        status=status,
        start_time=span.start_time,
        end_time=span.end_time,
        instrumentation_scope=span.instrumentation_scope,
    )


class RedactingSpanExporter(SpanExporter):
    """Hides the error text that `TraceConfig(hide_inputs/hide_outputs)` does not reach.

    OpenInference masks span ATTRIBUTES only. The status description (`run.error`, with the full
    stack trace) and the `exception` events (`exception.message`, `exception.stacktrace`; a
    structured-output parse failure quotes the raw model text) would leave raw. Span name, kind,
    timings, status code and `exception.type` are kept.
    """

    def __init__(self, exporter: SpanExporter) -> None:
        self._exporter = exporter

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        return self._exporter.export([_redact_span(span) for span in spans])

    def shutdown(self) -> None:
        self._exporter.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._exporter.force_flush(timeout_millis)


def endpoint_for_log(endpoint: str) -> str:
    """`scheme://host:port` only: userinfo, path and query can carry credentials."""
    parts = urlsplit(endpoint)
    host = parts.hostname or ""
    if ":" in host:  # IPv6 literal
        host = f"[{host}]"
    try:
        port = parts.port
    except ValueError:
        port = None
    return f"{parts.scheme}://{host}" + (f":{port}" if port else "")


class OtlpTracing(Tracing):
    """`exporter` is injectable for tests (an in-memory exporter); default = OTLP/HTTP."""

    def __init__(self, config: OtlpConfig, exporter: SpanExporter | None = None) -> None:
        self._config = config
        self._exporter = exporter
        self._provider: TracerProvider | None = None

    @property
    def enabled(self) -> bool:
        return True

    def start(self) -> None:
        if self._provider is not None:
            return
        instrumentor = LangChainInstrumentor()
        if instrumentor.is_instrumented_by_opentelemetry:
            raise TracingError(
                "LangChain is already instrumented in this process (another container that was "
                "not closed): only one tracing adapter can be active at a time"
            )
        config = self._config
        exporter = self._exporter or OTLPSpanExporter(
            endpoint=config.endpoint,
            headers=dict(config.headers) or None,
            timeout=config.timeout_seconds,
        )
        if config.hide_content:
            exporter = RedactingSpanExporter(exporter)
        # `openinference.project.name` is the project Phoenix files the traces under.
        resource = Resource.create(
            {SERVICE_NAME: config.service_name, PROJECT_NAME: config.service_name}
        )
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        try:
            instrumentor.instrument(
                tracer_provider=provider,
                config=TraceConfig(
                    hide_inputs=config.hide_content, hide_outputs=config.hide_content
                ),
                raise_exception_on_conflict=True,
            )
        except Exception as exc:
            provider.shutdown()
            raise TracingError(f"Cannot instrument LangChain: {exc}") from exc
        self._provider = provider
        logger.info(
            "Tracing on: OTLP/HTTP to %s as service %r (hide_content=%s)",
            endpoint_for_log(config.endpoint),
            config.service_name,
            config.hide_content,
        )

    def run_scope(self, attributes: RunAttributes) -> AbstractContextManager[object]:
        # `session.id` comes from the thread id; the rest travels as the `metadata` attribute.
        return using_attributes(
            session_id=attributes.session_id or "",
            metadata={
                "tenant_id": attributes.tenant_id,
                "agent_id": attributes.agent_id,
                "run_id": attributes.run_id,
            },
        )

    def force_flush(self) -> None:
        if self._provider is not None:
            self._provider.force_flush()

    def shutdown(self) -> None:
        provider, self._provider = self._provider, None
        if provider is None:
            return
        LangChainInstrumentor().uninstrument()
        # No explicit force_flush: the processor's shutdown drains the queue, and a flush ignores
        # its timeout (the caller bounds the whole shutdown).
        provider.shutdown()
