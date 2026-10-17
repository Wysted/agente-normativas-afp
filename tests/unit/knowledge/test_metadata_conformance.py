"""Chunk metadata is checked against the profile schema at ingest (and frozen safely)."""

import math

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from agintis.modules.knowledge.api import RagProfile, SourceFile
from agintis.modules.knowledge.application import (
    ChunkDraft,
    Chunker,
    IngestDocument,
    MetadataResolver,
    ParsedDocument,
)
from agintis.modules.knowledge.domain import (
    Chunk,
    InvalidMetadataError,
    MetadataField,
    conform_metadata,
    freeze_metadata,
)
from agintis.shared_kernel.ids import ChunkId, DocumentId, TenantId
from tests.fakes.clock import FixedClock
from tests.fakes.knowledge import (
    FakeParser,
    InMemoryChunkRepository,
    InMemoryDocumentRepository,
    InMemoryIngestLock,
)

YEAR = MetadataField("year", "int", filterable=True, source="group:year")
WEIGHT = MetadataField("weight", "float", filterable=True)
EFFECTIVE = MetadataField("effective", "date", filterable=True)
ARTICLE = MetadataField("article", "str", filterable=True)
SCHEMA = (YEAR, WEIGHT, EFFECTIVE, ARTICLE)


class TestFreezeMetadata:
    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
    def test_non_finite_floats_are_rejected(self, value: float) -> None:
        with pytest.raises(InvalidMetadataError, match="finite"):
            freeze_metadata({"weight": value})

    @pytest.mark.parametrize("value", ["a\x00b", "\x00", "lone \ud800 surrogate"])
    def test_strings_postgres_cannot_store_are_rejected(self, value: str) -> None:
        with pytest.raises(InvalidMetadataError, match=r"NUL|UTF-8"):
            freeze_metadata({"heading": value})

    def test_chunks_validate_on_construction(self) -> None:
        with pytest.raises(InvalidMetadataError):
            Chunk(ChunkId("c"), DocumentId("d"), TenantId("t"), "p", 0, "x", {"weight": math.nan})

    def test_ordinary_values_pass(self) -> None:
        assert freeze_metadata({"a": "ñandú", "b": 1, "c": -2.5}) == {
            "a": "ñandú",
            "b": 1,
            "c": -2.5,
        }


class TestConformMetadata:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [(2024, 2024), ("2024", 2024), (" -7 ", -7), ("+3", 3), (5.0, 5)],
    )
    def test_int_fields_coerce_unambiguous_values(self, raw: object, expected: int) -> None:
        result = conform_metadata({"year": raw}, SCHEMA)  # type: ignore[dict-item]
        assert result["year"] == expected
        assert type(result["year"]) is int

    @pytest.mark.parametrize(
        "raw", ["20x4", "", "1_000", "1,000", "2024.5", "1e3", "\u0663", 2.5, math.inf, "0x10"]
    )
    def test_int_fields_reject_ambiguous_values(self, raw: object) -> None:
        with pytest.raises(InvalidMetadataError, match="year"):
            conform_metadata({"year": raw}, SCHEMA)  # type: ignore[dict-item]

    @pytest.mark.parametrize(
        ("raw", "expected"), [(2, 2), (0.5, 0.5), ("0.25", 0.25), ("-1e2", -100.0)]
    )
    def test_float_fields_accept_numbers_and_decimal_strings(
        self, raw: object, expected: float
    ) -> None:
        assert conform_metadata({"weight": raw}, SCHEMA)["weight"] == expected  # type: ignore[dict-item]

    @pytest.mark.parametrize("raw", ["abc", "", "nan", "inf", "1e999", "1_0", "\u0661"])
    def test_float_fields_reject_the_rest(self, raw: object) -> None:
        with pytest.raises(InvalidMetadataError, match="weight"):
            conform_metadata({"weight": raw}, SCHEMA)  # type: ignore[dict-item]

    def test_date_fields_require_iso_dates(self) -> None:
        assert conform_metadata({"effective": "2024-06-30"}, SCHEMA)["effective"] == "2024-06-30"
        for raw in ("2024-6-30", "20240630", "2024-02-30", "30/06/2024", "2024-06-30T10:00", 2024):
            with pytest.raises(InvalidMetadataError, match="effective"):
                conform_metadata({"effective": raw}, SCHEMA)

    def test_str_fields_do_not_stringify_numbers(self) -> None:
        with pytest.raises(InvalidMetadataError, match="article"):
            conform_metadata({"article": 12}, SCHEMA)

    def test_builtin_page_is_typed_too(self) -> None:
        assert conform_metadata({"page": "3"}, ())["page"] == 3
        with pytest.raises(InvalidMetadataError, match="page"):
            conform_metadata({"page": "three"}, ())

    def test_undeclared_keys_are_kept_but_still_frozen_safely(self) -> None:
        result = conform_metadata({"heading": "Title", "extra": 1.5}, SCHEMA)
        assert dict(result) == {"heading": "Title", "extra": 1.5}
        with pytest.raises(InvalidMetadataError):
            conform_metadata({"extra": math.nan}, SCHEMA)
        with pytest.raises(InvalidMetadataError, match="reserved"):
            conform_metadata({"tenant_id": "x"}, SCHEMA)

    def test_result_is_read_only(self) -> None:
        result = conform_metadata({"year": "1999"}, SCHEMA)
        with pytest.raises(TypeError):
            result["year"] = 1  # type: ignore[index]


class RegexGroupChunker(Chunker):
    """Emits what a regex-group extractor yields: every captured group is a string."""

    def __init__(self, metadata: dict[str, str | int | float]) -> None:
        self._metadata = metadata

    async def chunk(self, parsed: ParsedDocument, profile: RagProfile) -> list[ChunkDraft]:
        return [ChunkDraft(text=s.text, metadata=self._metadata) for s in parsed.sections]


def _ingest(
    metadata: dict[str, str | int | float],
) -> tuple[IngestDocument, InMemoryDocumentRepository, InMemoryChunkRepository]:
    documents = InMemoryDocumentRepository()
    chunks = InMemoryChunkRepository(documents)
    ingest = IngestDocument(
        documents,
        chunks,
        {"fake": FakeParser()},
        {"fake": RegexGroupChunker(metadata)},
        DeterministicFakeEmbedding(size=4),
        FixedClock(),
        InMemoryIngestLock(),
        metadata=MetadataResolver({}, {}, FixedClock()),
    )
    return ingest, documents, chunks


PROFILE = RagProfile("p", 3, 100, "fake", "fake", metadata_fields=SCHEMA)
SOURCE = SourceFile("file:///a.txt", "text/plain", b"one\n\ntwo")


class TestIngestValidatesMetadata:
    async def test_regex_group_strings_are_coerced_before_storage(self) -> None:
        ingest, _, chunks = _ingest({"year": "2024", "effective": "2024-06-30", "article": "12"})
        await ingest.execute(TenantId("t"), PROFILE, SOURCE)
        stored = [c.metadata for c, _ in chunks.items]
        assert stored == [{"year": 2024, "effective": "2024-06-30", "article": "12"}] * 2
        assert type(stored[0]["year"]) is int

    async def test_invalid_metadata_fails_before_any_side_effect(self) -> None:
        ingest, documents, chunks = _ingest({"year": "MMXXIV"})
        with pytest.raises(InvalidMetadataError, match=r"Chunk 0 of 'file:///a.txt'.*year"):
            await ingest.execute(TenantId("t"), PROFILE, SOURCE)
        assert documents.items == {}
        assert chunks.items == []

    async def test_undeclared_metadata_still_passes(self) -> None:
        ingest, _, chunks = _ingest({"heading": "Intro", "page": 2})
        await ingest.execute(TenantId("t"), PROFILE, SOURCE)
        assert chunks.items[0][0].metadata == {"heading": "Intro", "page": 2}
