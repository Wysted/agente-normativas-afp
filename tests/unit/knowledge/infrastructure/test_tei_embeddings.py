import asyncio
import json
import logging
import threading
from typing import Any

import httpx
import pytest

from agintis.modules.knowledge.infrastructure.tei import (
    TeiEmbeddings,
    TeiError,
    TeiUnavailableError,
    verify_served_model,
)
from agintis.modules.knowledge.infrastructure.tei import http as tei_http

BASE = "http://tei.test"
SECRET_HOST = "tei.internal.example"


class Server:
    """Scripted TEI: records requests, answers with `statuses` first, then OK vectors."""

    def __init__(self, statuses: list[int] | None = None, dim: int = 3) -> None:
        self.statuses = list(statuses or [])
        self.dim = dim
        self.requests: list[dict[str, Any]] = []
        self.headers: list[httpx.Headers] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/embed"
        body = json.loads(request.content)
        self.requests.append(body)
        self.headers.append(request.headers)
        if self.statuses:
            status = self.statuses.pop(0)
            if status != httpx.codes.OK:
                return httpx.Response(status, text="nope")
        return httpx.Response(200, json=[[float(len(t))] * self.dim for t in body["inputs"]])


class Sleeps:
    """Records requested delays instead of sleeping."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    async def asleep(self, delay: float) -> None:
        self.delays.append(delay)

    def sleep(self, delay: float) -> None:
        self.delays.append(delay)


def _adapter(
    server: Server | None = None,
    sleeps: Sleeps | None = None,
    handler: Any = None,
    **kwargs: Any,
) -> TeiEmbeddings:
    sleeps = sleeps or Sleeps()
    handler = handler or (server or Server()).handler
    options: dict[str, Any] = {
        "batch_size": 16,
        "timeout_seconds": 5.0,
        "deadline_seconds": 30.0,
        "jitter": lambda: 1.0,
        "sleep": sleeps.asleep,
        "sync_sleep": sleeps.sleep,
        "transport": httpx.MockTransport(handler),
        "async_transport": httpx.MockTransport(handler),
    }
    options.update(kwargs)
    return TeiEmbeddings(BASE, **options)


class TestBatching:
    async def test_splits_into_batches_and_keeps_order(self) -> None:
        server = Server()
        texts = [f"{'x' * i}" for i in range(1, 8)]
        vectors = await _adapter(server, batch_size=3).aembed_documents(texts)
        assert [len(r["inputs"]) for r in server.requests] == [3, 3, 1]
        assert [v[0] for v in vectors] == [float(i) for i in range(1, 8)]

    async def test_batches_group_texts_of_similar_length_but_results_keep_input_order(self) -> None:
        server = Server()
        texts = ["x" * n for n in (9, 1, 8, 2, 7, 3, 6, 4, 5)]
        vectors = await _adapter(server, batch_size=3).aembed_documents(texts)
        sent = [[len(t) for t in r["inputs"]] for r in server.requests]
        assert sent == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]  # padding waste stays within a batch
        assert [v[0] for v in vectors] == [float(n) for n in (9, 1, 8, 2, 7, 3, 6, 4, 5)]

    def test_sync_paths_restore_the_input_order_too(self) -> None:
        texts = ["ccc", "a", "bb", "dddd"]
        vectors = _adapter(batch_size=2).embed_documents(texts)
        assert [v[0] for v in vectors] == [3.0, 1.0, 2.0, 4.0]

    def test_equal_lengths_keep_their_relative_order(self) -> None:
        server = Server()
        _adapter(server, batch_size=2).embed_documents(["bb", "aa", "cc"])
        assert [r["inputs"] for r in server.requests] == [["bb", "aa"], ["cc"]]

    def test_sync_matches_async_contract(self) -> None:
        server = Server()
        vectors = _adapter(server, batch_size=2).embed_documents(["a", "bb", "ccc"])
        assert [v[0] for v in vectors] == [1.0, 2.0, 3.0]
        assert len(server.requests) == 2

    async def test_requests_normalized_truncated_embeddings(self) -> None:
        server = Server()
        await _adapter(server).aembed_query("hola")
        assert server.requests == [{"inputs": ["hola"], "normalize": True, "truncate": True}]

    async def test_empty_input_makes_no_request(self) -> None:
        server = Server()
        assert await _adapter(server).aembed_documents([]) == []
        assert server.requests == []

    def test_embed_query_sync(self) -> None:
        assert _adapter().embed_query("abcd") == [4.0, 4.0, 4.0]

    def test_batch_size_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            _adapter(batch_size=0)


class TestPrefixes:
    async def test_query_and_document_prefixes_are_applied(self) -> None:
        server = Server()
        adapter = _adapter(server, query_prefix="query: ", document_prefix="passage: ")
        await adapter.aembed_query("hola")
        await adapter.aembed_documents(["uno", "dos"])
        assert [r["inputs"] for r in server.requests] == [
            ["query: hola"],
            ["passage: uno", "passage: dos"],
        ]

    def test_sync_paths_apply_prefixes_too(self) -> None:
        server = Server()
        adapter = _adapter(server, query_prefix="q:", document_prefix="d:")
        adapter.embed_query("a")
        adapter.embed_documents(["b"])
        assert [r["inputs"] for r in server.requests] == [["q:a"], ["d:b"]]

    async def test_no_prefix_by_default(self) -> None:
        server = Server()
        await _adapter(server).aembed_documents(["a"])
        assert server.requests[0]["inputs"] == ["a"]


class TestRetries:
    async def test_retries_429_and_5xx_then_succeeds(self) -> None:
        server = Server(statuses=[429, 503])
        vectors = await _adapter(server).aembed_documents(["a"])
        assert vectors == [[1.0, 1.0, 1.0]]
        assert len(server.requests) == 3

    def test_sync_retries_too(self) -> None:
        server = Server(statuses=[500])
        assert _adapter(server).embed_documents(["a"]) == [[1.0, 1.0, 1.0]]

    async def test_documents_give_up_after_four_attempts(self) -> None:
        server = Server(statuses=[503] * 10)
        with pytest.raises(TeiUnavailableError, match="4 attempt"):
            await _adapter(server).aembed_documents(["a"])
        assert len(server.requests) == 4

    async def test_queries_retry_only_once(self) -> None:
        server = Server(statuses=[503] * 10)
        with pytest.raises(TeiUnavailableError, match="2 attempt"):
            await _adapter(server).aembed_query("a")
        assert len(server.requests) == 2

    def test_sync_queries_retry_only_once(self) -> None:
        server = Server(statuses=[503] * 10)
        with pytest.raises(TeiUnavailableError):
            _adapter(server).embed_query("a")
        assert len(server.requests) == 2

    async def test_client_errors_are_not_retried(self) -> None:
        server = Server(statuses=[413])
        with pytest.raises(TeiError, match="HTTP 413: nope") as excinfo:
            await _adapter(server).aembed_documents(["a"])
        assert not isinstance(excinfo.value, TeiUnavailableError)
        assert len(server.requests) == 1

    async def test_connect_errors_are_retried_then_reported(self) -> None:
        calls = 0

        def boom(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            raise httpx.ConnectError("refused", request=request)

        with pytest.raises(TeiUnavailableError, match="ConnectError"):
            await _adapter(handler=boom, max_retries=2).aembed_documents(["a"])
        assert calls == 3

    async def test_read_timeout_is_not_retried(self) -> None:
        calls = 0
        sleeps = Sleeps()

        def slow(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            raise httpx.ReadTimeout("slow", request=request)

        with pytest.raises(TeiUnavailableError, match="read timeout"):
            await _adapter(sleeps=sleeps, handler=slow).aembed_documents(["a"])
        assert calls == 1
        assert sleeps.delays == []

    def test_sync_read_timeout_is_not_retried(self) -> None:
        calls = 0

        def slow(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            raise httpx.ReadTimeout("slow", request=request)

        with pytest.raises(TeiUnavailableError, match="read timeout"):
            _adapter(handler=slow).embed_documents(["a"])
        assert calls == 1

    async def test_retry_after_is_honored_but_capped(self) -> None:
        sleeps = Sleeps()
        responses = iter([httpx.Response(429, headers={"retry-after": "30"}), None])

        def handler(request: httpx.Request) -> httpx.Response:
            return next(responses) or httpx.Response(200, json=[[1.0]])

        assert await _adapter(sleeps=sleeps, handler=handler).aembed_documents(["a"]) == [[1.0]]
        assert sleeps.delays == [tei_http.MAX_BACKOFF_SECONDS]  # 8.0, not 30

    async def test_small_retry_after_is_used_as_is(self) -> None:
        sleeps = Sleeps()
        responses = iter([httpx.Response(429, headers={"retry-after": "3"}), None])

        def handler(request: httpx.Request) -> httpx.Response:
            return next(responses) or httpx.Response(200, json=[[1.0]])

        await _adapter(sleeps=sleeps, handler=handler).aembed_documents(["a"])
        assert sleeps.delays == [3.0]

    async def test_backoff_grows_exponentially_and_is_capped(self) -> None:
        sleeps = Sleeps()
        server = Server(statuses=[503] * 6 + [200])
        await _adapter(server, sleeps, max_retries=6, backoff_seconds=0.5).aembed_documents(["a"])
        assert sleeps.delays == [0.5, 1.0, 2.0, 4.0, 8.0, 8.0]

    async def test_jitter_scales_the_backoff(self) -> None:
        sleeps = Sleeps()
        server = Server(statuses=[503, 503, 200])
        adapter = _adapter(server, sleeps, backoff_seconds=1.0, jitter=lambda: 0.5)
        await adapter.aembed_documents(["a"])
        assert sleeps.delays == [0.5, 1.0]

    def test_default_jitter_stays_in_range(self) -> None:
        samples = [tei_http.default_jitter() for _ in range(200)]
        assert all(tei_http.JITTER_FLOOR <= s <= 1.0 for s in samples)

    def test_sync_backoff_uses_the_sync_sleep(self) -> None:
        sleeps = Sleeps()
        _adapter(Server(statuses=[503, 503]), sleeps, backoff_seconds=0.5).embed_documents(["a"])
        assert sleeps.delays == [0.5, 1.0]


class TestDeadline:
    async def test_total_deadline_bounds_a_hanging_call(self) -> None:
        async def hang(request: httpx.Request) -> httpx.Response:
            await asyncio.sleep(30)
            return httpx.Response(200, json=[[1.0]])  # pragma: no cover

        adapter = _adapter(handler=hang, deadline_seconds=0.05)
        with pytest.raises(TeiUnavailableError, match="deadline"):
            await adapter.aembed_documents(["a"])

    def test_sync_deadline_stops_retrying(self) -> None:
        server = Server(statuses=[503] * 10)
        adapter = _adapter(server, deadline_seconds=1.0, backoff_seconds=5.0)
        with pytest.raises(TeiUnavailableError, match="deadline"):
            adapter.embed_documents(["a"])
        assert len(server.requests) == 1

    async def test_error_messages_do_not_leak_the_service_url(self) -> None:
        server = Server(statuses=[503] * 10)
        adapter = TeiEmbeddings(
            f"http://{SECRET_HOST}",
            batch_size=1,
            timeout_seconds=1,
            deadline_seconds=5,
            sleep=Sleeps().asleep,
            async_transport=httpx.MockTransport(server.handler),
        )
        with pytest.raises(TeiUnavailableError) as excinfo:
            await adapter.aembed_documents(["a"])
        assert SECRET_HOST not in str(excinfo.value)


class TestBadPayloads:
    @pytest.mark.parametrize(
        "payload",
        [
            {"error": "x"},
            [[1.0]],
            [1.0, 2.0],
            [[], [1.0]],
            [["a"], [1.0]],
            [[1.0, 2.0], [1.0]],
        ],
    )
    async def test_unexpected_shape_raises(self, payload: object) -> None:
        adapter = _adapter(handler=lambda r: httpx.Response(200, json=payload))
        with pytest.raises(TeiError):
            await adapter.aembed_documents(["a", "b"])

    async def test_non_finite_values_raise(self) -> None:
        adapter = _adapter(handler=lambda r: httpx.Response(200, content=b"[[NaN], [1.0]]"))
        with pytest.raises(TeiError, match="non-finite"):
            await adapter.aembed_documents(["a", "b"])

    async def test_expected_dimension_is_enforced(self) -> None:
        adapter = _adapter(Server(dim=3), expected_dim=4)
        with pytest.raises(TeiError, match="dimension 3, expected 4"):
            await adapter.aembed_query("a")

    async def test_expected_dimension_accepts_matching_vectors(self) -> None:
        assert len(await _adapter(Server(dim=3), expected_dim=3).aembed_query("a")) == 3

    async def test_non_json_body_is_a_tei_error(self) -> None:
        adapter = _adapter(handler=lambda r: httpx.Response(200, text="<html>"))
        with pytest.raises(TeiError, match="not JSON"):
            await adapter.aembed_query("a")


class TestAuth:
    async def test_api_key_is_sent_as_bearer_token(self) -> None:
        server = Server()
        await _adapter(server, api_key="s3cret").aembed_query("a")
        assert server.headers[0]["authorization"] == "Bearer s3cret"

    async def test_no_authorization_header_without_key(self) -> None:
        server = Server()
        await _adapter(server).aembed_query("a")
        assert "authorization" not in server.headers[0]


class TestLifecycle:
    async def test_aclose_is_idempotent(self) -> None:
        adapter = _adapter()
        await adapter.aembed_query("a")
        await adapter.aclose()
        await adapter.aclose()
        adapter.close()

    def test_sync_client_is_created_once_under_concurrency(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        created = 0
        real_client = httpx.Client

        class CountingClient(real_client):  # type: ignore[valid-type, misc]
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                nonlocal created
                created += 1
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(httpx, "Client", CountingClient)
        adapter = _adapter()
        barrier = threading.Barrier(8)

        def work() -> None:
            barrier.wait()
            adapter.embed_query("a")

        threads = [threading.Thread(target=work) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert created == 1
        adapter.close()


class TestServedModelCheck:
    @staticmethod
    def _info(payload: object, status: int = 200) -> TeiEmbeddings:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/info"
            return httpx.Response(status, json=payload)

        return _adapter(handler=handler)

    async def test_returns_the_model_id(self) -> None:
        assert await self._info({"model_id": "BAAI/bge-m3"}).aserved_model_id() == "BAAI/bge-m3"

    async def test_mismatch_logs_a_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            await verify_served_model(self._info({"model_id": "other/model"}), "BAAI/bge-m3")
        assert "other/model" in caplog.text
        assert "BAAI/bge-m3" in caplog.text

    async def test_match_is_silent(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            await verify_served_model(self._info({"model_id": "BAAI/bge-m3"}), "BAAI/bge-m3")
        assert caplog.text == ""

    async def test_unreachable_info_only_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            await verify_served_model(self._info({}, status=503), "BAAI/bge-m3")
        assert "Could not verify" in caplog.text
