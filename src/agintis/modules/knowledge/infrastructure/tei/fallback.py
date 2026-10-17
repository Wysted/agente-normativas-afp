"""Graceful degradation for the optional reranker (Strategy decorator)."""

import asyncio
import logging
from collections.abc import Sequence

from agintis.modules.knowledge.application import NoopReranker, Reranker
from agintis.modules.knowledge.domain import ScoredChunk
from agintis.modules.knowledge.infrastructure.tei.errors import TeiError

logger = logging.getLogger(__name__)


class FallbackReranker(Reranker):
    """Runs `primary` under a deadline; on a TEI failure or timeout returns `fallback` instead.

    Reranking only improves ordering, so an unavailable or slow reranker must not fail the
    whole retrieval: the default fallback keeps the vector order (`NoopReranker`), and the
    returned chunks are the untouched candidates, so callers see no rerank score.

    Only TEI errors and timeouts are absorbed; programming errors and cancellation propagate.
    The warning carries the error class and message, which never include the service URL.
    """

    def __init__(
        self,
        primary: Reranker,
        *,
        deadline_seconds: float,
        fallback: Reranker | None = None,
    ) -> None:
        if deadline_seconds <= 0:
            raise ValueError("deadline_seconds must be greater than 0")
        self._primary = primary
        self._fallback = fallback if fallback is not None else NoopReranker()
        self._deadline = deadline_seconds

    async def rerank(
        self, query: str, chunks: Sequence[ScoredChunk], top_n: int
    ) -> list[ScoredChunk]:
        try:
            async with asyncio.timeout(self._deadline):
                return await self._primary.rerank(query, chunks, top_n)
        except (TeiError, TimeoutError) as exc:
            logger.warning(
                "Reranker unavailable, keeping the vector order: %s: %s", type(exc).__name__, exc
            )
            return await self._fallback.rerank(query, chunks, top_n)

    async def aclose(self) -> None:
        close = getattr(self._primary, "aclose", None)
        if close is not None:
            await close()
