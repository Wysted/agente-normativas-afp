"""Reranker implementations that need no infrastructure."""

from collections.abc import Sequence

from agintis.modules.knowledge.application.ports import Reranker
from agintis.modules.knowledge.domain import ScoredChunk

NO_RERANKER_KEY = "none"


class NoopReranker(Reranker):
    """Keeps the incoming (retrieval) order and truncates to `top_n`.

    That order is the vector similarity for dense retrieval and the fused RRF score for hybrid.
    """

    async def rerank(
        self, query: str, chunks: Sequence[ScoredChunk], top_n: int
    ) -> list[ScoredChunk]:
        return list(chunks[: max(top_n, 0)])
