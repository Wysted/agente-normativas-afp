from agintis.modules.knowledge.domain import Chunk
from agintis.shared_kernel.criteria import Filter
from agintis.shared_kernel.ids import ChunkId, DocumentId, TenantId
from tests.fakes.knowledge import InMemoryChunkRepository


async def test_fake_never_lets_metadata_shadow_core_fields() -> None:
    # Chunk rejects reserved keys, so corrupt an instance to prove the fake is defensive too.
    chunk = Chunk(ChunkId("c"), DocumentId("d"), TenantId("A"), "p", 0, "t")
    object.__setattr__(chunk, "metadata", {"tenant_id": "B", "text": "forged"})
    repo = InMemoryChunkRepository()
    await repo.add_many([chunk], [[1.0]])

    assert await repo.search(TenantId("B"), [1.0], None, k=5) == []
    assert await repo.search(TenantId("A"), [1.0], Filter.eq("text", "forged"), k=5) == []
    assert len(await repo.search(TenantId("A"), [1.0], Filter.eq("text", "t"), k=5)) == 1
