import hashlib
import json
from collections.abc import Mapping
from types import MappingProxyType

import pytest

from agintis.modules.knowledge.api import DEFAULT_PROFILE, CleanerSpec, RagProfile
from agintis.modules.knowledge.application import ParsedDocument, ParsedSection, join_sections
from agintis.modules.knowledge.domain import MetadataField, freeze_params, thaw_params


def _profile(**overrides: object) -> RagProfile:
    base: dict[str, object] = {
        "name": "p",
        "top_k": 3,
        "chunk_max_tokens": 100,
        "parser_key": "simple",
        "chunker_key": "structured",
    }
    return RagProfile(**{**base, **overrides})  # type: ignore[arg-type]


class TestFreezeParams:
    def test_nested_data_becomes_read_only_and_hashable_by_content(self) -> None:
        frozen = freeze_params({"levels": [{"name": "a", "depth": 0}], "flag": True})
        assert isinstance(frozen, Mapping)
        levels = frozen["levels"]
        assert isinstance(levels, tuple)
        assert isinstance(levels[0], MappingProxyType)
        with pytest.raises(TypeError):
            levels[0]["name"] = "b"  # type: ignore[index]

    def test_freezing_copies_the_input(self) -> None:
        source = {"levels": [{"name": "a"}]}
        frozen = freeze_params(source)
        source["levels"][0]["name"] = "changed"
        assert thaw_params(frozen) == {"levels": [{"name": "a"}]}

    def test_thaw_round_trips_to_plain_json_types(self) -> None:
        data = {"a": [1, 2.5, "x", None, True], "b": {"c": []}}
        assert thaw_params(freeze_params(data)) == data

    @pytest.mark.parametrize(
        "bad",
        [{"": 1}, {"a": float("nan")}, {"a": object()}, {"a": {1: 2}}, {"a": {"": 1}}, {"a": {1}}],
    )
    def test_unsupported_values_are_rejected(self, bad: object) -> None:
        with pytest.raises(ValueError, match=r"."):
            freeze_params(bad)


class TestProfileParams:
    def test_nested_chunker_params_are_immutable(self) -> None:
        profile = _profile(chunker_params={"levels": [{"name": "a"}]})
        with pytest.raises(TypeError):
            profile.chunker_params["x"] = 1  # type: ignore[index]
        with pytest.raises(TypeError):
            profile.chunker_params["levels"][0]["name"] = "b"  # type: ignore[index]

    def test_cleaner_spec_requires_a_key_and_freezes_params(self) -> None:
        with pytest.raises(ValueError, match="key"):
            CleanerSpec("")
        spec = CleanerSpec("strip_lines", {"patterns": ("a",)})
        assert spec.params["patterns"] == ("a",)
        with pytest.raises(ValueError, match="mapping"):
            CleanerSpec("k", ["x"])  # type: ignore[arg-type]

    def test_chunker_params_must_be_a_mapping(self) -> None:
        with pytest.raises(ValueError, match="mapping"):
            _profile(chunker_params=["x"])


class TestFingerprint:
    def test_legacy_profiles_keep_their_fingerprint(self) -> None:
        legacy = json.dumps(
            ["simple", "structured", 512, [["include_heading", True]]], separators=(",", ":")
        )
        assert DEFAULT_PROFILE.fingerprint == hashlib.sha256(legacy.encode()).hexdigest()

    def test_nested_params_ignore_key_order_but_not_list_order(self) -> None:
        a = _profile(chunker_params={"levels": [{"name": "a", "depth": 0}, {"name": "b"}]})
        b = _profile(chunker_params={"levels": [{"depth": 0, "name": "a"}, {"name": "b"}]})
        c = _profile(chunker_params={"levels": [{"name": "b"}, {"name": "a", "depth": 0}]})
        assert a.fingerprint == b.fingerprint
        assert a.fingerprint != c.fingerprint

    def test_nested_param_change_changes_it(self) -> None:
        a = _profile(chunker_params={"levels": [{"pattern": "^a"}]})
        b = _profile(chunker_params={"levels": [{"pattern": "^b"}]})
        assert a.fingerprint != b.fingerprint

    def test_cleaners_are_part_of_it_in_order_and_params(self) -> None:
        strip = CleanerSpec("strip_lines", {"patterns": ("^x",)})
        other = CleanerSpec("normalize_whitespace")
        base = _profile().fingerprint
        with_strip = _profile(cleaners=(strip,)).fingerprint
        assert with_strip != base
        assert (
            _profile(cleaners=(strip, other)).fingerprint
            != _profile(cleaners=(other, strip)).fingerprint
        )
        assert (
            _profile(cleaners=(CleanerSpec("strip_lines", {"patterns": ("^y",)}),)).fingerprint
            != with_strip
        )
        assert _profile(cleaners=(strip,)).fingerprint == with_strip

    def test_cleaners_never_collide_with_the_metadata_part(self) -> None:
        with_metadata = _profile(metadata_fields=(MetadataField("a", "str", source="group:a"),))
        with_cleaner = _profile(cleaners=(CleanerSpec("a"),))
        assert with_metadata.fingerprint != with_cleaner.fingerprint


class TestStrategyVersions:
    """D2: the implementation version of each strategy that runs is part of the fingerprint."""

    LEGACY_DEFAULT = hashlib.sha256(
        json.dumps(
            ["simple", "structured", 512, [["include_heading", True]]], separators=(",", ":")
        ).encode()
    ).hexdigest()

    def test_baseline_versions_keep_the_legacy_hash(self) -> None:
        baseline = {"parser:simple": 1, "chunker:structured": 1}
        assert DEFAULT_PROFILE.fingerprint_for(baseline) == self.LEGACY_DEFAULT
        assert DEFAULT_PROFILE.fingerprint_for() == self.LEGACY_DEFAULT
        assert DEFAULT_PROFILE.fingerprint_for({}) == DEFAULT_PROFILE.fingerprint

    def test_baseline_versions_keep_the_hash_of_profiles_with_cleaners_and_metadata(self) -> None:
        profile = _profile(
            cleaners=(CleanerSpec("strip_lines", {"patterns": ("^x",)}),),
            metadata_fields=(MetadataField("a", "str", source="group:a"),),
        )
        versions = {"parser:simple": 1, "chunker:structured": 1, "cleaner:strip_lines": 1}
        assert profile.fingerprint_for(versions) == profile.fingerprint

    def test_a_bumped_strategy_changes_the_hash(self) -> None:
        base = DEFAULT_PROFILE.fingerprint
        bumped = DEFAULT_PROFILE.fingerprint_for({"chunker:structured": 2})
        assert bumped != base
        assert bumped != DEFAULT_PROFILE.fingerprint_for({"chunker:structured": 3})
        assert bumped != DEFAULT_PROFILE.fingerprint_for({"parser:simple": 2})

    def test_each_kind_and_key_is_pinned_separately(self) -> None:
        profile = _profile(cleaners=(CleanerSpec("strip_lines", {"patterns": ("^x",)}),))
        one = profile.fingerprint_for({"cleaner:strip_lines": 2})
        assert one != profile.fingerprint
        assert one != profile.fingerprint_for({"parser:simple": 2})

    def test_order_of_the_versions_mapping_is_irrelevant(self) -> None:
        a = DEFAULT_PROFILE.fingerprint_for({"parser:simple": 2, "chunker:structured": 3})
        b = DEFAULT_PROFILE.fingerprint_for({"chunker:structured": 3, "parser:simple": 2})
        assert a == b

    def test_the_versions_part_never_collides_with_cleaners_or_metadata(self) -> None:
        with_cleaner = _profile(cleaners=(CleanerSpec("a"),)).fingerprint
        assert _profile().fingerprint_for({"parser:simple": 2}) != with_cleaner

    def test_strategies_declare_versions_and_the_shipped_ones_are_bumped(self) -> None:
        from agintis.modules.knowledge.application import Chunker, Cleaner, Parser
        from agintis.modules.knowledge.infrastructure.chunkers import StructuredChunker
        from agintis.modules.knowledge.infrastructure.cleaners import CLEANER_CLASSES
        from agintis.modules.knowledge.infrastructure.parsers import SimpleParser

        assert (Parser.version, Chunker.version, Cleaner.version) == (1, 1, 1)
        assert StructuredChunker.version == 2  # page_end always stored, per-chunk re-split
        assert SimpleParser.version == 3  # Markdown heading attribute blocks
        versions = {cls.key: cls.version for cls in CLEANER_CLASSES}
        assert versions["html_main_content"] == versions["html_collect"] == 2
        assert versions["normalize_quotes"] == 2


class TestJoinedText:
    SECTIONS = (
        ParsedSection(text="alpha beta", page=1),
        ParsedSection(text="gamma", page=2),
        ParsedSection(text="delta epsilon", page=3),
    )

    def test_text_and_offsets(self) -> None:
        joined = join_sections(self.SECTIONS)
        assert joined.text == "alpha beta\n\ngamma\n\ndelta epsilon"
        assert [(s.start, s.end, s.page) for s in joined.spans] == [
            (0, 10, 1),
            (12, 17, 2),
            (19, 32, 3),
        ]
        for span, section in zip(joined.spans, self.SECTIONS, strict=True):
            assert joined.text[span.start : span.end] == section.text

    def test_pages_between(self) -> None:
        joined = join_sections(self.SECTIONS)
        assert joined.pages_between(0, 5) == (1, 1)
        assert joined.pages_between(6, 17) == (1, 2)
        assert joined.pages_between(0, len(joined.text)) == (1, 3)
        assert joined.pages_between(19, 32) == (3, 3)

    def test_separator_offsets_map_to_the_previous_section(self) -> None:
        joined = join_sections(self.SECTIONS)
        span = joined.span_at(10)
        assert span is not None
        assert span.page == 1
        assert joined.pages_between(10, 11) == (1, 1)

    def test_a_reversed_range_is_a_programming_error(self) -> None:
        joined = join_sections(self.SECTIONS)
        with pytest.raises(ValueError, match="start <= end"):
            joined.pages_between(5, 4)
        assert joined.pages_between(5, 5) == (1, 1)  # empty range is fine

    def test_out_of_range_and_empty(self) -> None:
        joined = join_sections(self.SECTIONS)
        assert joined.pages_between(-1, 3) is None
        assert joined.pages_between(0, 999) is None
        assert join_sections(()).pages_between(0, 0) is None
        assert join_sections(()).span_at(0) is None

    def test_pageless_sections_have_no_pages(self) -> None:
        joined = join_sections((ParsedSection(text="a"), ParsedSection(text="b")))
        assert joined.pages_between(0, 4) is None

    def test_a_single_page_section_inside_pageless_ones(self) -> None:
        joined = join_sections(
            (ParsedSection(text="a"), ParsedSection(text="b", page=7), ParsedSection(text="c"))
        )
        assert joined.pages_between(0, len(joined.text)) == (7, 7)

    def test_parsed_document_exposes_the_view(self) -> None:
        doc = ParsedDocument(title="t", sections=self.SECTIONS)
        assert doc.joined().text == join_sections(self.SECTIONS).text
        assert doc.joined(" ").text == "alpha beta gamma delta epsilon"
