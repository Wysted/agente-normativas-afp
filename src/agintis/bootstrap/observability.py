"""Tracing wiring shared by the container and the CLIs that call models without a container."""

import asyncio
import logging
import threading
from contextlib import suppress

from agintis.bootstrap.settings import Settings
from agintis.platform.observability import (
    OtlpConfig,
    Tracing,
    build_tracing,
    parse_headers,
    reject_foreign_tracing,
)

logger = logging.getLogger(__name__)

TRACING_SHUTDOWN_GRACE_SECONDS = 5.0


def build_tracing_from_settings(settings: Settings) -> Tracing:
    """The tracing port the settings ask for (nothing is installed until `Tracing.start`).

    Our tracing port must be the ONLY tracing path: LangChain's own is refused before anything
    else is built. Fails fast too when `otlp` is chosen without the `otel` extra.
    """
    reject_foreign_tracing()
    obs = settings.observability
    return build_tracing(
        OtlpConfig(
            endpoint=obs.otlp_endpoint or "",
            service_name=obs.service_name,
            headers=parse_headers(obs.otlp_headers.get_secret_value()) if obs.otlp_headers else {},
            hide_content=obs.hide_content,
            timeout_seconds=obs.otlp_timeout_seconds,
        )
        if obs.tracing == "otlp"
        else None
    )


def _settle(done: asyncio.Future[None], error: BaseException | None) -> None:
    if done.done():  # the waiter gave up (timeout)
        return
    if error is None:
        done.set_result(None)
    else:
        done.set_exception(error)


async def shutdown_tracing(tracing: Tracing, max_wait_seconds: float) -> None:
    """Drain and release the exporter off the event loop, waiting at most `max_wait_seconds`.

    It runs in a DAEMON thread (not `to_thread`: executor threads are joined at interpreter exit,
    so a hung exporter would still block the process). On timeout the buffered spans are dropped
    with a warning and the shutdown goes on.
    """
    loop = asyncio.get_running_loop()
    done: asyncio.Future[None] = loop.create_future()

    def work() -> None:
        error: BaseException | None = None
        try:
            tracing.shutdown()
        except Exception as exc:
            error = exc
        with suppress(RuntimeError):  # the loop may be closed after a timeout
            loop.call_soon_threadsafe(_settle, done, error)

    threading.Thread(target=work, name="tracing-shutdown", daemon=True).start()
    try:
        await asyncio.wait_for(done, max_wait_seconds)
    except TimeoutError:
        logger.warning(
            "Tracing shutdown did not finish within %.1fs: buffered spans may be lost",
            max_wait_seconds,
        )
