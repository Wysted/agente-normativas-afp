"""ASGI-level client disconnect: the run's stream must be closed promptly, not at GC time."""

import asyncio
import json
from collections.abc import MutableMapping
from typing import Any

from agintis.main import create_app
from agintis.modules.agents_catalog.api import Token
from agintis.modules.agents_catalog.http import get_services
from tests.fakes.agents import BlockingRunner, EndlessRunner
from tests.support.settings import make_settings
from tests.unit.test_agents_http import services_for

CLEANUP_TIMEOUT_SECONDS = 2


def _scope(agent_id: str) -> dict[str, Any]:
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/runs/stream",
        "raw_path": b"/v1/runs/stream",
        "query_string": b"",
        "headers": [
            (b"host", b"test"),
            (b"content-type", b"application/json"),
            (b"x-tenant-id", b"acme"),
        ],
        "server": ("test", 80),
        "client": ("127.0.0.1", 1234),
    }


async def _run_and_disconnect(runner: BlockingRunner | EndlessRunner, *, slow_send: bool) -> None:
    app = create_app(make_settings())
    app.dependency_overrides[get_services] = lambda: services_for(runner)
    body = json.dumps({"agent_id": runner.agent_id, "message": "hi"}).encode()
    disconnect = asyncio.Event()
    request_sent = False
    body_chunks = 0

    async def receive() -> MutableMapping[str, Any]:
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message: MutableMapping[str, Any]) -> None:
        nonlocal body_chunks
        if message["type"] != "http.response.body" or not message.get("body"):
            return
        body_chunks += 1
        if body_chunks == 1:
            disconnect.set()
        if slow_send:
            await asyncio.sleep(30)  # a consumer that cannot keep up

    async with app.router.lifespan_context(app):
        call = asyncio.create_task(app(_scope(runner.agent_id), receive, send))
        try:
            await asyncio.wait_for(runner.finalized_event.wait(), CLEANUP_TIMEOUT_SECONDS)
        finally:
            call.cancel()
            await asyncio.gather(call, return_exceptions=True)


async def test_disconnect_while_runner_is_blocked_closes_the_runner() -> None:
    runner = BlockingRunner(Token("a"))

    await _run_and_disconnect(runner, slow_send=False)

    assert runner.finalized
    assert runner.cancelled


async def test_disconnect_with_a_slow_consumer_closes_the_runner() -> None:
    runner = EndlessRunner()

    await _run_and_disconnect(runner, slow_send=True)

    assert runner.finalized_event.is_set()
