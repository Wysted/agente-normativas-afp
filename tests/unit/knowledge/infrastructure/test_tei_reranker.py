import json
import logging

import httpx
import pytest

from agintis.modules.knowledge.domain import Chunk, ScoredChunk
from agintis.modules.knowledge.infrastructure.tei import TeiError, TeiReranker, TeiUnavailableError
from agintis.shared_kernel.ids import ChunkId, DocumentId, TenantId


def _chunks(n: int) -> list[ScoredChunk]:
    return [
        ScoredChunk(Chunk(ChunkId(f"c{i}"), DocumentId("d"), TenantId("t"), "p", i, f"t{i}"), 0.5)
        for i in range(n)
    ]


def _reranker(handler: httpx.MockTransport | None = None, **kwargs: object) -> TeiReranker:
    async def no_sleep(delay: float) -> None:
        return None

    options: dict[str, object] = {
        "timeout_seconds": 5.0,
        "deadline_seconds": 30.0,
        "backoff_seconds": 0,
        "sleep": no_sleep,
        "async_transport": handler,
    }
    options.update(kwargs)
    return TeiReranker("http://tei.test", **options)  # type: ignore[arg-type]


def _answer(payload: object) -> httpx.MockTransport:
    return httpx.MockTransport(lambda r: httpx.Response(200, json=payload))


async def test_maps_scores_back_orders_and_truncates() -> None:
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/rerank"
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json=[
                {"index": 1, "score": 0.2},
                {"index": 2, "score": 0.9},
                {"index": 0, "score": 0.5},
            ],
        )

    ranked = await _reranker(httpx.MockTransport(handler)).rerank("q", _chunks(3), top_n=2)

    assert [(r.chunk.id, r.score) for r in ranked] == [("c2", 0.9), ("c0", 0.5)]
    assert seen[0]["texts"] == ["t0", "t1", "t2"]
    assert seen[0]["query"] == "q"


async def test_ties_are_broken_by_candidate_index() -> None:
    payload = [{"index": 2, "score": 0.7}, {"index": 0, "score": 0.7}, {"index": 1, "score": 0.7}]
    ranked = await _reranker(_answer(payload)).rerank("q", _chunks(3), top_n=3)
    assert [r.chunk.id for r in ranked] == ["c0", "c1", "c2"]


async def test_unknown_and_repeated_indices_are_ignored(caplog: pytest.LogCaptureFixture) -> None:
    payload = [
        {"index": 9, "score": 5.0},
        {"index": -1, "score": 4.0},
        {"index": 1, "score": 0.3},
        {"index": 1, "score": 0.9},
        {"index": 0, "score": 0.5},
    ]
    with caplog.at_level(logging.WARNING):
        ranked = await _reranker(_answer(payload)).rerank("q", _chunks(2), top_n=5)
    assert [(r.chunk.id, r.score) for r in ranked] == [("c0", 0.5), ("c1", 0.3)]
    assert "unknown or repeated index" in caplog.text


async def test_empty_candidates_or_zero_top_n_skip_the_call() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("must not be called")

    reranker = _reranker(httpx.MockTransport(handler))
    assert await reranker.rerank("q", [], 3) == []
    assert await reranker.rerank("q", _chunks(2), 0) == []


@pytest.mark.parametrize(
    "payload",
    [
        {"error": "x"},
        [{"score": 1.0}],
        [{"index": 0, "score": "high"}],
        [{"index": 7, "score": 1.0}],  # nothing usable at all
    ],
)
async def test_malformed_responses_raise(payload: object) -> None:
    with pytest.raises(TeiError):
        await _reranker(_answer(payload)).rerank("q", _chunks(2), 2)


async def test_non_finite_score_raises() -> None:
    transport = httpx.MockTransport(
        lambda r: httpx.Response(200, content=b'[{"index": 0, "score": NaN}]')
    )
    with pytest.raises(TeiError, match="non-finite"):
        await _reranker(transport).rerank("q", _chunks(1), 1)


async def test_query_path_retries_once_then_reports_unavailable() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503)

    with pytest.raises(TeiUnavailableError):
        await _reranker(httpx.MockTransport(handler)).rerank("q", _chunks(1), 1)
    assert calls == 2


async def test_read_timeout_is_not_retried() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(TeiUnavailableError, match="read timeout"):
        await _reranker(httpx.MockTransport(handler)).rerank("q", _chunks(1), 1)
    assert calls == 1


async def test_api_key_is_sent() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["authorization"])
        return httpx.Response(200, json=[{"index": 0, "score": 1.0}])

    await _reranker(httpx.MockTransport(handler), api_key="k").rerank("q", _chunks(1), 1)
    assert seen == ["Bearer k"]
