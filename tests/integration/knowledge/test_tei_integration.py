"""Real TEI containers: `docker compose up -d --wait embeddings` (reranker: --profile rerank).

Excluded from the default run (pytest `addopts`); run with `uv run pytest -m integration`.
Skipped automatically when the service is unreachable.
"""

import math

import httpx
import pytest

from agintis.bootstrap.settings import EmbeddingsSettings, RerankerSettings
from agintis.modules.knowledge.api import ParsedDocument, ParsedSection, RagProfile
from agintis.modules.knowledge.application import NoopReranker
from agintis.modules.knowledge.domain import Chunk, ScoredChunk
from agintis.modules.knowledge.infrastructure.chunkers import HfTokenCounter, StructuredChunker
from agintis.modules.knowledge.infrastructure.chunkers.splitting import SPECIAL_TOKENS_RESERVE
from agintis.modules.knowledge.infrastructure.tei import (
    FallbackReranker,
    TeiEmbeddings,
    TeiReranker,
    verify_served_model,
)
from agintis.shared_kernel.ids import ChunkId, DocumentId, TenantId
from tests.conftest import skip_or_fail

pytestmark = pytest.mark.integration

BGE_M3_DIM = 1024
RERANKER_FALLBACK_URL = "http://localhost:8082"


def _reachable(url: str) -> bool:
    try:
        return httpx.get(f"{url}/health", timeout=2).is_success
    except httpx.HTTPError:
        return False


@pytest.fixture(scope="module")
def embeddings_settings() -> EmbeddingsSettings:
    settings = EmbeddingsSettings(_env_file=None)
    if not _reachable(settings.url):
        skip_or_fail(f"TEI embeddings not reachable at {settings.url}")
    return settings


@pytest.fixture
def embeddings(embeddings_settings: EmbeddingsSettings) -> TeiEmbeddings:
    return TeiEmbeddings(
        embeddings_settings.url,
        batch_size=embeddings_settings.batch_size,
        timeout_seconds=embeddings_settings.timeout_seconds,
        deadline_seconds=embeddings_settings.deadline_seconds,
        expected_dim=BGE_M3_DIM,
    )


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


async def test_embeds_normalized_vectors_in_batches(embeddings: TeiEmbeddings) -> None:
    # 20 texts > batch_size 16 -> two requests.
    texts = [f"Documento de prueba numero {i}" for i in range(20)]
    vectors = await embeddings.aembed_documents(texts)
    await embeddings.aclose()

    assert len(vectors) == 20
    assert all(len(v) == BGE_M3_DIM for v in vectors)
    assert all(math.isclose(math.sqrt(sum(x * x for x in v)), 1.0, abs_tol=1e-3) for v in vectors)


async def test_semantic_similarity_and_sync_api(embeddings: TeiEmbeddings) -> None:
    query = embeddings.embed_query("¿Cuál es la capital de Francia?")
    related, unrelated = embeddings.embed_documents(
        ["París es la capital de Francia.", "El motor diésel consume gasoil."]
    )
    embeddings.close()
    assert _cosine(query, related) > _cosine(query, unrelated)


async def test_chunker_budget_holds_against_the_real_tokenizer(
    embeddings_settings: EmbeddingsSettings,
) -> None:
    try:
        counter = HfTokenCounter(embeddings_settings.tokenizer_source)
        counter.count("warm up")
    except Exception as exc:  # hub unreachable and not cached
        skip_or_fail(f"tokenizer unavailable: {exc}")
    profile = RagProfile("p", 3, 64, "simple", "structured")
    text = "\n\n".join(
        f"Oración número {i} con varias palabras para ocupar tokens. " * 6 for i in range(8)
    )
    drafts = await StructuredChunker(counter).chunk(
        ParsedDocument("t", (ParsedSection(text=text),)), profile
    )
    assert len(drafts) > 1
    assert all(counter.count(d.text) <= 64 - SPECIAL_TOKENS_RESERVE for d in drafts)


async def test_served_model_matches_the_configured_one(
    embeddings: TeiEmbeddings, embeddings_settings: EmbeddingsSettings
) -> None:
    assert await embeddings.aserved_model_id() == embeddings_settings.model
    await verify_served_model(embeddings, embeddings_settings.model)
    await embeddings.aclose()


async def test_unreachable_reranker_degrades_to_the_vector_order() -> None:
    dead = TeiReranker("http://127.0.0.1:9", timeout_seconds=1, deadline_seconds=5, max_retries=0)
    reranker = FallbackReranker(dead, deadline_seconds=5, fallback=NoopReranker())
    candidates = [
        ScoredChunk(Chunk(ChunkId(f"c{i}"), DocumentId("d"), TenantId("t"), "p", i, "x"), 0.1)
        for i in range(3)
    ]
    assert await reranker.rerank("q", candidates, top_n=2) == candidates[:2]
    await reranker.aclose()


async def test_reranker_orders_by_relevance() -> None:
    settings = RerankerSettings(_env_file=None)
    url = settings.url or RERANKER_FALLBACK_URL
    if not _reachable(url):
        pytest.skip(f"TEI reranker not reachable at {url}")
    texts = [
        "El motor diésel consume gasoil.",
        "París es la capital de Francia.",
        "Me gusta el té.",
    ]
    candidates = [
        ScoredChunk(Chunk(ChunkId(f"c{i}"), DocumentId("d"), TenantId("t"), "p", i, t), 0.1)
        for i, t in enumerate(texts)
    ]
    reranker = TeiReranker(
        url, timeout_seconds=settings.timeout_seconds, deadline_seconds=settings.deadline_seconds
    )
    ranked = await reranker.rerank("¿Cuál es la capital de Francia?", candidates, top_n=2)
    await reranker.aclose()

    assert len(ranked) == 2
    assert ranked[0].chunk.id == "c1"
    assert ranked[0].score >= ranked[1].score
