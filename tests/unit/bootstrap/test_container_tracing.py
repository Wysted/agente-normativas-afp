"""Tracing wiring in the composition root (no network: the OTLP exporter never sends here)."""

import asyncio
import logging
import threading
import time

import pytest
from openinference.instrumentation.langchain import LangChainInstrumentor

from agintis.agent_kit.runtime import TracedRunner
from agintis.bootstrap.container import Container, StartupError
from agintis.bootstrap.observability import shutdown_tracing
from agintis.platform.observability import (
    ForeignTracingError,
    NoTracing,
    TracingNotInstalledError,
)
from agintis.platform.observability.otlp import OtlpTracing
from agintis.platform.observability.tracing import Tracing
from tests.support.settings import make_settings

OTLP = {"tracing": "otlp", "otlp_endpoint": "http://localhost:6006/v1/traces"}


@pytest.fixture(autouse=True)
def _no_boot_side_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agintis.bootstrap.container.ChatModels.verify", lambda self: None)
    for name in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"):
        monkeypatch.delenv(name, raising=False)


def test_tracing_is_off_by_default_and_runners_are_not_wrapped() -> None:
    container = Container.build(make_settings())
    assert isinstance(container.tracing, NoTracing)
    assert not any(
        isinstance(runner, TracedRunner) for runner in container.agents_catalog.registry.list()
    )


async def test_otlp_wraps_every_runner_and_instruments_for_the_container_lifetime() -> None:
    container = Container.build(make_settings(observability=OTLP))
    assert isinstance(container.tracing, OtlpTracing)
    runners = container.agents_catalog.registry.list()
    assert runners
    assert all(isinstance(runner, TracedRunner) for runner in runners)
    assert not LangChainInstrumentor().is_instrumented_by_opentelemetry  # build starts nothing

    async with container:
        assert LangChainInstrumentor().is_instrumented_by_opentelemetry
    assert not LangChainInstrumentor().is_instrumented_by_opentelemetry


async def test_containers_can_restart_in_one_process() -> None:
    for _ in range(2):
        async with Container.build(make_settings(observability=OTLP)):
            assert LangChainInstrumentor().is_instrumented_by_opentelemetry
        assert not LangChainInstrumentor().is_instrumented_by_opentelemetry


async def test_a_second_live_container_is_a_startup_error() -> None:
    async with Container.build(make_settings(observability=OTLP)):
        with pytest.raises(StartupError, match="already instrumented"):
            await Container.build(make_settings(observability=OTLP)).__aenter__()
    assert not LangChainInstrumentor().is_instrumented_by_opentelemetry


@pytest.mark.parametrize("name", ["LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"])
@pytest.mark.parametrize("mode", [{}, OTLP])
def test_langchain_own_tracing_refuses_to_boot(
    monkeypatch: pytest.MonkeyPatch, name: str, mode: dict[str, str]
) -> None:
    monkeypatch.setenv(name, "true")
    with pytest.raises(ForeignTracingError, match=name):
        Container.build(make_settings(observability=mode))


def test_otlp_without_the_extra_is_a_boot_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(__import__("sys").modules, "agintis.platform.observability.otlp", None)
    with pytest.raises(TracingNotInstalledError, match="otel"):
        Container.build(make_settings(observability=OTLP))


class _HangingTracing(NoTracing):
    """A shutdown that blocks (an unreachable exporter) until the test releases it."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.finished = threading.Event()

    def shutdown(self) -> None:
        self.release.wait(timeout=10)
        self.finished.set()


class _FailingTracing(NoTracing):
    def shutdown(self) -> None:
        raise RuntimeError("exporter exploded")


async def test_a_hanging_tracing_shutdown_is_bounded(caplog: pytest.LogCaptureFixture) -> None:
    tracing = _HangingTracing()
    started = time.monotonic()
    try:
        with caplog.at_level(logging.WARNING):
            await shutdown_tracing(tracing, 0.2)
        assert time.monotonic() - started < 2
        assert "did not finish" in caplog.text
    finally:
        tracing.release.set()
    assert await asyncio.to_thread(tracing.finished.wait, 5)  # the daemon thread winds down


async def test_a_fast_tracing_shutdown_returns_without_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    done = threading.Event()

    class _Quick(NoTracing):
        def shutdown(self) -> None:
            done.set()

    tracing: Tracing = _Quick()
    with caplog.at_level(logging.WARNING):
        await shutdown_tracing(tracing, 5)
    assert done.is_set()
    assert not caplog.records


async def test_a_failing_tracing_shutdown_propagates() -> None:
    with pytest.raises(RuntimeError, match="exploded"):
        await shutdown_tracing(_FailingTracing(), 5)


def test_the_otlp_timeout_reaches_the_config() -> None:
    container = Container.build(make_settings(observability={**OTLP, "otlp_timeout_seconds": 2.5}))
    assert isinstance(container.tracing, OtlpTracing)
    assert container.tracing._config.timeout_seconds == 2.5
