"""Document-level metadata supplied at ingest (`SourceFile.document_metadata`)."""

import math
from dataclasses import replace
from typing import Any

import pytest

from agintis.modules.knowledge.api import ParsedDocument, SourceFile
from agintis.modules.knowledge.application import (
    ChunkDraft,
    freeze_document_metadata,
    metadata_hash,
)
from agintis.modules.knowledge.application.dto import (
    MAX_DOCUMENT_METADATA_KEYS,
    MAX_DOCUMENT_STR_CHARS,
)
from agintis.modules.knowledge.domain import InvalidMetadataError


class TestFreeze:
    def test_lists_become_tuples_and_the_result_is_read_only(self) -> None:
        source: dict[str, Any] = {"a": 1, "b": ["x", "y"], "c": None, "d": True, "e": 1.5}
        frozen = freeze_document_metadata(source)
        assert dict(frozen) == {"a": 1, "b": ("x", "y"), "c": None, "d": True, "e": 1.5}
        with pytest.raises(TypeError):
            frozen["a"] = 2  # type: ignore[index]

    def test_it_copies_the_input(self) -> None:
        source: dict[str, Any] = {"b": ["x"]}
        frozen = freeze_document_metadata(source)
        source["b"].append("y")
        source["z"] = 1
        assert dict(frozen) == {"b": ("x",)}

    @pytest.mark.parametrize(
        "bad",
        [{"a": {"nested": 1}}, {"a": [["x"]]}, {"a": object()}, {"a": math.inf}, {"": 1}],
    )
    def test_unsupported_values_are_rejected(self, bad: dict[str, Any]) -> None:
        with pytest.raises(InvalidMetadataError):
            freeze_document_metadata(bad)

    def test_the_item_count_is_capped(self) -> None:
        with pytest.raises(InvalidMetadataError, match="more than"):
            freeze_document_metadata({"a": list(range(65))})


class TestIngestBoundary:
    """Caller data is validated once, here: later text built from it is storable."""

    @pytest.mark.parametrize("bad", ["a\x00b", "lone\ud800surrogate"])
    def test_strings_must_be_storable_text(self, bad: str) -> None:
        for metadata in ({"k": bad}, {"k": [bad]}, {bad: 1}):
            with pytest.raises(InvalidMetadataError, match=r"NUL|UTF-8"):
                freeze_document_metadata(metadata)

    def test_strings_are_capped(self) -> None:
        freeze_document_metadata({"k": "x" * MAX_DOCUMENT_STR_CHARS})
        with pytest.raises(InvalidMetadataError, match="longer than"):
            freeze_document_metadata({"k": "x" * (MAX_DOCUMENT_STR_CHARS + 1)})
        with pytest.raises(InvalidMetadataError, match="longer than"):
            freeze_document_metadata({"k": ["ok", "x" * (MAX_DOCUMENT_STR_CHARS + 1)]})

    def test_the_number_of_keys_is_capped(self) -> None:
        freeze_document_metadata({f"k{i}": 1 for i in range(MAX_DOCUMENT_METADATA_KEYS)})
        with pytest.raises(InvalidMetadataError, match="keys"):
            freeze_document_metadata({f"k{i}": 1 for i in range(MAX_DOCUMENT_METADATA_KEYS + 1)})

    def test_a_source_file_rejects_unstorable_metadata_at_construction(self) -> None:
        with pytest.raises(InvalidMetadataError, match="NUL"):
            SourceFile("u", "text/plain", b"x", document_metadata={"title": "bad\x00title"})

    def test_chunk_text_is_sanitized_whatever_built_it(self) -> None:
        draft = ChunkDraft("head\x00ing\ud800\nbody", body="bo\x00dy")
        assert draft.text == "heading\nbody"
        assert draft.body == "body"


class TestMetadataHash:
    def test_it_is_stable_and_order_independent(self) -> None:
        one = freeze_document_metadata({"a": 1, "b": ["x", "y"], "c": None})
        two = freeze_document_metadata({"c": None, "b": ("x", "y"), "a": 1})
        assert metadata_hash(one) == metadata_hash(two)
        assert len(metadata_hash(one)) == 64

    @pytest.mark.parametrize(
        "other",
        [{"a": 2, "b": ["x", "y"]}, {"a": 1, "b": ["y", "x"]}, {"a": "1", "b": ["x", "y"]}, {}],
    )
    def test_any_difference_changes_it(self, other: dict[str, Any]) -> None:
        base = freeze_document_metadata({"a": 1, "b": ["x", "y"]})
        assert metadata_hash(base) != metadata_hash(freeze_document_metadata(other))

    def test_ints_bools_and_floats_are_distinct(self) -> None:
        hashes = {metadata_hash(freeze_document_metadata({"a": v})) for v in (1, True, 1.0)}
        assert len(hashes) == 3

    def test_non_ascii_text_is_hashed(self) -> None:
        assert metadata_hash(freeze_document_metadata({"t": "ñandú"})) != metadata_hash({})


class TestCarriers:
    def test_source_files_default_to_no_metadata_and_freeze_what_they_get(self) -> None:
        assert dict(SourceFile("u", "text/plain", b"x").document_metadata) == {}
        source = SourceFile("u", "text/plain", b"x", document_metadata={"s": ["a", "b"]})  # type: ignore[dict-item]
        assert source.document_metadata["s"] == ("a", "b")

    def test_parsed_documents_keep_it_through_replace(self) -> None:
        parsed = ParsedDocument("t", (), document_metadata={"n": 1})
        assert dict(replace(parsed, title="u").document_metadata) == {"n": 1}
        with pytest.raises(TypeError):
            parsed.document_metadata["n"] = 2  # type: ignore[index]
