"""Reranker port adapter backed by a TEI `/rerank` endpoint."""

import asyncio
import logging
import math
from collections.abc import Sequence

import httpx

from agintis.modules.knowledge.application import Reranker
from agintis.modules.knowledge.domain import ScoredChunk
from agintis.modules.knowledge.infrastructure.tei.errors import TeiError
from agintis.modules.knowledge.infrastructure.tei.http import (
    DEFAULT_BACKOFF_SECONDS,
    QUERY_MAX_RETRIES,
    AsyncSleep,
    Jitter,
    TeiHttp,
    default_jitter,
)

RERANK_PATH = "/rerank"

logger = logging.getLogger(__name__)


class TeiReranker(Reranker):
    """Cross-encoder rerank. Callers must keep the candidate list small (CPU: ~45 s / 16 texts).

    Reranking sits on the query path (a user is waiting): one retry at most, and no retry on a
    read timeout. Wrap it in `FallbackReranker` to degrade to the vector order on failure.
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float,
        deadline_seconds: float,
        api_key: str | None = None,
        max_retries: int = QUERY_MAX_RETRIES,
        backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
        async_transport: httpx.AsyncBaseTransport | None = None,
        sleep: AsyncSleep = asyncio.sleep,
        jitter: Jitter = default_jitter,
    ) -> None:
        self._http = TeiHttp(
            base_url,
            timeout_seconds,
            deadline_seconds=deadline_seconds,
            max_retries=max_retries,
            backoff_seconds=backoff_seconds,
            api_key=api_key,
            async_transport=async_transport,
            sleep=sleep,
            jitter=jitter,
        )

    async def rerank(
        self, query: str, chunks: Sequence[ScoredChunk], top_n: int
    ) -> list[ScoredChunk]:
        if not chunks or top_n <= 0:
            return []
        payload = {
            "query": query,
            "texts": [c.chunk.text for c in chunks],
            "truncate": True,
            "return_text": False,
        }
        result = await self._http.apost(RERANK_PATH, payload)
        return self._map(chunks, result)[:top_n]

    @staticmethod
    def _map(chunks: Sequence[ScoredChunk], result: object) -> list[ScoredChunk]:
        """Order by `(-score, index)` (deterministic ties). Unknown/duplicate indices are ignored
        with a warning; a payload with no usable item at all is an error."""
        if not isinstance(result, list):
            raise TeiError("TEI /rerank returned an unexpected payload")
        scored: dict[int, float] = {}
        for item in result:
            try:
                index, score = int(item["index"]), float(item["score"])
            except (KeyError, TypeError, ValueError):
                raise TeiError("TEI /rerank returned a malformed item") from None
            if not math.isfinite(score):
                raise TeiError("TEI /rerank returned a non-finite score")
            if not 0 <= index < len(chunks) or index in scored:
                logger.warning(
                    "TEI /rerank returned an unknown or repeated index %d; ignored", index
                )
                continue
            scored[index] = score
        if result and not scored:
            raise TeiError("TEI /rerank returned no usable index")
        ordered = sorted(scored.items(), key=lambda pair: (-pair[1], pair[0]))
        return [ScoredChunk(chunk=chunks[index].chunk, score=score) for index, score in ordered]

    async def aclose(self) -> None:
        await self._http.aclose()
