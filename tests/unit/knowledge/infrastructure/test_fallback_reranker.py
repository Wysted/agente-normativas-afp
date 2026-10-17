import asyncio
import logging
from collections.abc import Sequence

import httpx
import pytest

from agintis.modules.knowledge.application import NoopReranker, Reranker
from agintis.modules.knowledge.domain import Chunk, ScoredChunk
from agintis.modules.knowledge.infrastructure.tei import (
    FallbackReranker,
    TeiError,
    TeiReranker,
    TeiUnavailableError,
)
from agintis.shared_kernel.ids import ChunkId, DocumentId, TenantId

SECRET_HOST = "reranker.internal.example"


def _chunks(n: int) -> list[ScoredChunk]:
    return [
        ScoredChunk(Chunk(ChunkId(f"c{i}"), DocumentId("d"), TenantId("t"), "p", i, f"t{i}"), 0.5)
        for i in range(n)
    ]


class Raising(Reranker):
    def __init__(self, error: BaseException) -> None:
        self.error = error
        self.closed = False

    async def rerank(
        self, query: str, chunks: Sequence[ScoredChunk], top_n: int
    ) -> list[ScoredChunk]:
        raise self.error

    async def aclose(self) -> None:
        self.closed = True


class Hanging(Reranker):
    async def rerank(
        self, query: str, chunks: Sequence[ScoredChunk], top_n: int
    ) -> list[ScoredChunk]:
        await asyncio.sleep(30)
        return []  # pragma: no cover


class Reversing(Reranker):
    async def rerank(
        self, query: str, chunks: Sequence[ScoredChunk], top_n: int
    ) -> list[ScoredChunk]:
        return [ScoredChunk(c.chunk, 9.0) for c in reversed(chunks)][:top_n]


@pytest.mark.parametrize("error", [TeiError("boom"), TeiUnavailableError("down"), TimeoutError()])
async def test_failures_degrade_to_the_vector_order(error: BaseException) -> None:
    chunks = _chunks(4)
    result = await FallbackReranker(Raising(error), deadline_seconds=5).rerank("q", chunks, 2)
    assert result == chunks[:2]
    assert all(r is c for r, c in zip(result, chunks, strict=False))  # untouched candidates


async def test_slow_primary_hits_the_deadline_and_degrades() -> None:
    chunks = _chunks(3)
    result = await FallbackReranker(Hanging(), deadline_seconds=0.05).rerank("q", chunks, 3)
    assert result == chunks


async def test_healthy_primary_is_used() -> None:
    result = await FallbackReranker(Reversing(), deadline_seconds=5).rerank("q", _chunks(3), 2)
    assert [r.chunk.id for r in result] == ["c2", "c1"]


async def test_custom_fallback_is_used() -> None:
    result = await FallbackReranker(
        Raising(TeiError("x")), deadline_seconds=5, fallback=Reversing()
    ).rerank("q", _chunks(2), 2)
    assert [r.chunk.id for r in result] == ["c1", "c0"]


async def test_programming_errors_and_cancellation_propagate() -> None:
    with pytest.raises(ValueError, match="bug"):
        await FallbackReranker(Raising(ValueError("bug")), deadline_seconds=5).rerank(
            "q", _chunks(1), 1
        )
    with pytest.raises(asyncio.CancelledError):
        await FallbackReranker(Raising(asyncio.CancelledError()), deadline_seconds=5).rerank(
            "q", _chunks(1), 1
        )


async def test_warning_is_logged_without_the_service_url(
    caplog: pytest.LogCaptureFixture,
) -> None:
    transport = httpx.MockTransport(lambda r: httpx.Response(500, text="oops"))
    primary = TeiReranker(
        f"http://{SECRET_HOST}",
        timeout_seconds=1,
        deadline_seconds=5,
        max_retries=0,
        async_transport=transport,
    )
    with caplog.at_level(logging.WARNING):
        result = await FallbackReranker(primary, deadline_seconds=5).rerank("q", _chunks(2), 1)
    assert [r.chunk.id for r in result] == ["c0"]
    fallback_logs = [r.getMessage() for r in caplog.records if "Reranker unavailable" in r.message]
    assert fallback_logs
    assert all(SECRET_HOST not in m for m in fallback_logs)


async def test_aclose_delegates_to_the_primary() -> None:
    primary = Raising(TeiError("x"))
    await FallbackReranker(primary, deadline_seconds=5).aclose()
    assert primary.closed


def test_deadline_must_be_positive() -> None:
    with pytest.raises(ValueError, match="deadline"):
        FallbackReranker(NoopReranker(), deadline_seconds=0)
