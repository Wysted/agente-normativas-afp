"""Observability: the tracing port (OpenTelemetry/OTLP adapter, optional `otel` extra)."""

from agintis.platform.observability.tracing import (
    FOREIGN_TRACING_VARIABLES,
    OTEL_EXTRA,
    ForeignTracingError,
    NoTracing,
    OtlpConfig,
    RunAttributes,
    Tracing,
    TracingError,
    TracingNotInstalledError,
    build_tracing,
    parse_headers,
    reject_foreign_tracing,
)

__all__ = [
    "FOREIGN_TRACING_VARIABLES",
    "OTEL_EXTRA",
    "ForeignTracingError",
    "NoTracing",
    "OtlpConfig",
    "RunAttributes",
    "Tracing",
    "TracingError",
    "TracingNotInstalledError",
    "build_tracing",
    "parse_headers",
    "reject_foreign_tracing",
]
