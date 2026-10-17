"""langchain_core `Embeddings` backed by a TEI `/embed` endpoint (httpx, no torch)."""

import asyncio
import logging
import math
import time
from typing import Any

import httpx
from langchain_core.embeddings import Embeddings

from agintis.modules.knowledge.infrastructure.tei.errors import TeiError
from agintis.modules.knowledge.infrastructure.tei.http import (
    DEFAULT_BACKOFF_SECONDS,
    INGEST_MAX_RETRIES,
    QUERY_MAX_RETRIES,
    AsyncSleep,
    Jitter,
    SyncSleep,
    TeiHttp,
    default_jitter,
)

EMBED_PATH = "/embed"
INFO_PATH = "/info"

logger = logging.getLogger(__name__)


class TeiEmbeddings(Embeddings):
    """Batches by `batch_size` (must not exceed the server's --max-client-batch-size).

    Vectors are L2-normalized by the server (`normalize=True`), so cosine == dot product.
    Batches are sent sequentially: the CPU server serializes them anyway and parallel batches
    would only trip its concurrency limit (429). Texts are grouped into batches by length (the
    results are put back in the caller's order): TEI pads every batch to its longest text, and
    measured on the CPU image (bge-m3, 128 chunks of 180 tokens median, batches of 16) that cut
    the time by ~15% (42.0 s -> 35.6 s) compared with batches of mixed lengths.

    Two retry policies: documents (ingest, background work) retry `max_retries` times; queries
    (a user is waiting) retry `query_max_retries` times. `query_prefix` / `document_prefix`
    are prepended to the texts for models trained with instructions (e5: "query: ",
    "passage: "); empty for bge-m3. A document prefix consumes tokens of the chunk budget.
    """

    def __init__(
        self,
        base_url: str,
        *,
        batch_size: int,
        timeout_seconds: float,
        deadline_seconds: float,
        query_prefix: str = "",
        document_prefix: str = "",
        expected_dim: int | None = None,
        api_key: str | None = None,
        max_retries: int = INGEST_MAX_RETRIES,
        query_max_retries: int = QUERY_MAX_RETRIES,
        backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
        transport: httpx.BaseTransport | None = None,
        async_transport: httpx.AsyncBaseTransport | None = None,
        sleep: AsyncSleep = asyncio.sleep,
        sync_sleep: SyncSleep = time.sleep,
        jitter: Jitter = default_jitter,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
        if expected_dim is not None and expected_dim <= 0:
            raise ValueError("expected_dim must be greater than 0")
        if query_max_retries < 0:
            raise ValueError("query_max_retries must not be negative")
        self._batch_size = batch_size
        self._query_prefix = query_prefix
        self._document_prefix = document_prefix
        self._expected_dim = expected_dim
        self._query_max_retries = query_max_retries
        self._http = TeiHttp(
            base_url,
            timeout_seconds,
            deadline_seconds=deadline_seconds,
            max_retries=max_retries,
            backoff_seconds=backoff_seconds,
            api_key=api_key,
            transport=transport,
            async_transport=async_transport,
            sleep=sleep,
            sync_sleep=sync_sleep,
            jitter=jitter,
        )

    def _batches(self, texts: list[str]) -> tuple[list[int], list[list[str]]]:
        """Indices of `texts` sorted by length (stable) and the batches cut from that order."""
        order = sorted(range(len(texts)), key=lambda index: len(texts[index]))
        size = self._batch_size
        batches = [
            [texts[i] for i in order[start : start + size]] for start in range(0, len(order), size)
        ]
        return order, batches

    @staticmethod
    def _in_input_order(order: list[int], vectors: list[list[float]]) -> list[list[float]]:
        restored: list[list[float]] = [[] for _ in vectors]
        for position, index in enumerate(order):
            restored[index] = vectors[position]
        return restored

    @staticmethod
    def _payload(batch: list[str], prefix: str) -> dict[str, Any]:
        return {"inputs": [prefix + t for t in batch], "normalize": True, "truncate": True}

    def _validate(self, batch: list[str], result: Any) -> list[list[float]]:
        """Shape, numeric and dimension checks; every violation is a `TeiError`."""
        if not isinstance(result, list) or len(result) != len(batch):
            raise TeiError(f"TEI /embed returned an unexpected payload for {len(batch)} inputs")
        vectors: list[list[float]] = []
        for vector in result:
            if not isinstance(vector, list) or not vector:
                raise TeiError("TEI /embed returned an unexpected payload (bad vector)")
            try:
                floats = [float(x) for x in vector]
            except (TypeError, ValueError):
                raise TeiError("TEI /embed returned non-numeric vector values") from None
            if not all(math.isfinite(x) for x in floats):
                raise TeiError("TEI /embed returned non-finite vector values")
            if self._expected_dim is not None and len(floats) != self._expected_dim:
                raise TeiError(
                    f"TEI /embed returned dimension {len(floats)}, expected {self._expected_dim}"
                )
            vectors.append(floats)
        if len({len(v) for v in vectors}) > 1:
            raise TeiError("TEI /embed returned vectors of different dimensions")
        return vectors

    def _embed(self, texts: list[str], prefix: str, retries: int) -> list[list[float]]:
        order, batches = self._batches(texts)
        vectors: list[list[float]] = []
        for batch in batches:
            result = self._http.post(EMBED_PATH, self._payload(batch, prefix), max_retries=retries)
            vectors.extend(self._validate(batch, result))
        return self._in_input_order(order, vectors)

    async def _aembed(self, texts: list[str], prefix: str, retries: int) -> list[list[float]]:
        order, batches = self._batches(texts)
        vectors: list[list[float]] = []
        for batch in batches:
            result = await self._http.apost(
                EMBED_PATH, self._payload(batch, prefix), max_retries=retries
            )
            vectors.extend(self._validate(batch, result))
        return self._in_input_order(order, vectors)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, self._document_prefix, self._http.max_retries)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], self._query_prefix, self._query_max_retries)[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._aembed(texts, self._document_prefix, self._http.max_retries)

    async def aembed_query(self, text: str) -> list[float]:
        return (await self._aembed([text], self._query_prefix, self._query_max_retries))[0]

    async def aserved_model_id(self) -> str | None:
        """`model_id` reported by TEI `GET /info` (None when absent). Raises `TeiError`."""
        info = await self._http.aget(INFO_PATH)
        model_id = info.get("model_id") if isinstance(info, dict) else None
        return model_id if isinstance(model_id, str) else None

    def close(self) -> None:
        self._http.close()

    async def aclose(self) -> None:
        await self._http.aclose()


async def verify_served_model(embeddings: TeiEmbeddings, expected_model: str) -> None:
    """Best-effort boot check: warn (never fail) when TEI serves another model than configured.

    A mismatch silently produces vectors that do not belong to the configured model's space.
    """
    try:
        served = await embeddings.aserved_model_id()
    except TeiError as exc:
        logger.warning("Could not verify the embeddings model served by TEI: %s", exc)
        return
    if served is not None and served != expected_model:
        logger.warning(
            "TEI serves model %r but EMBEDDINGS_MODEL is %r: tokenizer and vectors may not match",
            served,
            expected_model,
        )
