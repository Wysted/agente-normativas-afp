"""Profile metadata schema: validation, fingerprint and profile-scoped filterable fields."""

from collections.abc import Sequence

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from agintis.modules.knowledge.api import RagProfile, RetrieveForTopic
from agintis.modules.knowledge.domain import (
    MetadataField,
    ScoredChunk,
    chunk_filterable_fields,
)
from agintis.modules.knowledge.domain.ports import CHUNK_FILTERABLE_FIELDS
from agintis.shared_kernel.criteria import FilterExpr
from agintis.shared_kernel.ids import TenantId
from tests.fakes.knowledge import InMemoryChunkRepository, InMemoryDocumentRepository

ARTICLE = MetadataField("article", "str", filterable=True, indexed=True, source="group:article")
CHAPTER = MetadataField("chapter", "str", source="group:chapter")


def _profile(*fields: MetadataField) -> RagProfile:
    return RagProfile("p", 3, 100, "simple", "structured", metadata_fields=fields)


class TestMetadataField:
    def test_defaults_are_not_filterable_nor_indexed(self) -> None:
        field = MetadataField("article")
        assert (field.type, field.filterable, field.indexed, field.source) == (
            "str",
            False,
            False,
            None,
        )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"name": "1bad"},
            {"name": "has space"},
            {"name": "tenant_id"},  # reserved core field
            {"name": "text"},
            {"name": "a", "type": "enum"},
            {"name": "a", "indexed": True},  # indexed needs filterable
            {"name": "a", "source": "regex:x"},
            {"name": "a", "source": "group:"},
        ],
    )
    def test_invalid_declarations_are_rejected(self, kwargs: dict[str, object]) -> None:
        with pytest.raises(ValueError, match=r"."):
            MetadataField(**kwargs)  # type: ignore[arg-type]

    @pytest.mark.parametrize("type_", ["str", "int", "float", "date"])
    def test_supported_types(self, type_: str) -> None:
        assert MetadataField("a", type_).type == type_  # type: ignore[arg-type]

    def test_bool_is_not_a_declarable_type(self) -> None:
        # Strict-typed filters keep bool apart from int and nothing extracts booleans.
        with pytest.raises(ValueError, match="unsupported type"):
            MetadataField("a", "bool")  # type: ignore[arg-type]


class TestProfileSchema:
    def test_duplicate_names_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="unique"):
            _profile(ARTICLE, MetadataField("article", "int"))

    def test_builtin_fields_cannot_be_redeclared(self) -> None:
        with pytest.raises(ValueError, match="built-in"):
            _profile(MetadataField("page", "int", filterable=True))

    def test_fields_are_frozen_into_a_tuple(self) -> None:
        assert _profile(ARTICLE, CHAPTER).metadata_fields == (ARTICLE, CHAPTER)

    def test_filterable_fields_are_core_plus_builtin_plus_declared_filterable(self) -> None:
        assert chunk_filterable_fields(()) == CHUNK_FILTERABLE_FIELDS
        allowed = chunk_filterable_fields(_profile(ARTICLE, CHAPTER).metadata_fields)
        assert allowed == CHUNK_FILTERABLE_FIELDS | {"article"}
        assert "chapter" not in allowed  # declared but not filterable
        assert {"tenant_id", "profile", "document_id", "page"} <= allowed


class TestFingerprint:
    def test_profiles_without_fields_keep_their_previous_fingerprint(self) -> None:
        assert _profile().fingerprint == RagProfile("p", 3, 100, "simple", "structured").fingerprint

    def test_schema_changes_force_a_reindex(self) -> None:
        base = _profile(ARTICLE).fingerprint
        assert _profile().fingerprint != base  # added field
        assert _profile(MetadataField("article", "int", source="group:article")).fingerprint != base
        assert _profile(MetadataField("art", "str", source="group:article")).fingerprint != base
        assert _profile(MetadataField("article", "str", source="group:other")).fingerprint != base

    def test_declaration_order_and_query_time_flags_do_not_matter(self) -> None:
        assert _profile(ARTICLE, CHAPTER).fingerprint == _profile(CHAPTER, ARTICLE).fingerprint
        relaxed = MetadataField("article", "str", filterable=False, source="group:article")
        assert _profile(relaxed).fingerprint == _profile(ARTICLE).fingerprint


class RecordingChunks(InMemoryChunkRepository):
    def __init__(self) -> None:
        super().__init__()
        self.schemas: list[Sequence[MetadataField]] = []

    async def search(
        self,
        tenant_id: TenantId,
        query_vector: Sequence[float],
        filter: FilterExpr | None,
        k: int,
        *,
        metadata_fields: Sequence[MetadataField] = (),
    ) -> list[ScoredChunk]:
        self.schemas.append(metadata_fields)
        return await super().search(
            tenant_id, query_vector, filter, k, metadata_fields=metadata_fields
        )


async def test_retrieval_hands_the_profile_schema_to_the_repository() -> None:
    chunks = RecordingChunks()
    retrieve = RetrieveForTopic(
        InMemoryDocumentRepository(), chunks, DeterministicFakeEmbedding(size=4)
    )
    profile = _profile(ARTICLE, CHAPTER)
    await retrieve.execute(TenantId("t"), profile, "query")
    assert chunks.schemas == [(ARTICLE, CHAPTER)]
