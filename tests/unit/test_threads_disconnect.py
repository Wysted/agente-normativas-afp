"""ASGI-level client disconnect of a THREAD run: the lease must still be released."""

import asyncio
import json
from collections.abc import MutableMapping
from typing import Any

from agintis.main import create_app
from agintis.modules.agents_catalog.api import (
    AgentRegistry,
    AgentsCatalogServices,
    RunAgent,
    ThreadOutcome,
    Token,
)
from agintis.modules.agents_catalog.http import get_services
from agintis.platform.llm import ProviderErrorClassifier
from tests.fakes.agents import BlockingRunner, EndlessRunner, FakeThreadAccess
from tests.support.settings import make_settings

TIMEOUT_SECONDS = 2
PATH = b"/v1/threads/thread-1/runs/stream"


async def _disconnect_after_first_frame(
    runner: BlockingRunner | EndlessRunner, access: FakeThreadAccess, *, slow_send: bool
) -> None:
    registry = AgentRegistry((runner,))
    catalog = AgentsCatalogServices(
        registry, RunAgent(registry, ProviderErrorClassifier(), threads=access), 100
    )
    app = create_app(make_settings())
    app.dependency_overrides[get_services] = lambda: catalog
    body = json.dumps({"message": "hi"}).encode()
    scope: dict[str, Any] = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": PATH.decode(),
        "raw_path": PATH,
        "query_string": b"",
        "headers": [
            (b"host", b"test"),
            (b"content-type", b"application/json"),
            (b"x-tenant-id", b"acme"),
        ],
        "server": ("test", 80),
        "client": ("127.0.0.1", 1234),
    }
    disconnect = asyncio.Event()
    sent_request = False
    chunks = 0

    async def receive() -> MutableMapping[str, Any]:
        nonlocal sent_request
        if not sent_request:
            sent_request = True
            return {"type": "http.request", "body": body, "more_body": False}
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message: MutableMapping[str, Any]) -> None:
        nonlocal chunks
        if message["type"] != "http.response.body" or not message.get("body"):
            return
        chunks += 1
        if chunks == 1:
            disconnect.set()
        if slow_send:
            await asyncio.sleep(30)

    async with app.router.lifespan_context(app):
        call = asyncio.create_task(app(scope, receive, send))
        try:
            await asyncio.wait_for(access.released_event.wait(), TIMEOUT_SECONDS)
        finally:
            call.cancel()
            await asyncio.gather(call, return_exceptions=True)


async def test_disconnect_while_the_runner_is_blocked_releases_the_thread() -> None:
    runner = BlockingRunner(Token("a"))
    access = FakeThreadAccess("blocking")

    await _disconnect_after_first_frame(runner, access, slow_send=False)

    assert runner.finalized
    assert [outcome for _, outcome in access.released] == [ThreadOutcome.IDLE]
    assert len(access.released) == 1


async def test_disconnect_with_a_slow_consumer_releases_the_thread() -> None:
    runner = EndlessRunner()
    access = FakeThreadAccess("endless")

    await _disconnect_after_first_frame(runner, access, slow_send=True)

    assert runner.finalized_event.is_set()
    assert [outcome for _, outcome in access.released] == [ThreadOutcome.IDLE]
