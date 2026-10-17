"""Boot-time validation of metadata sources, extractors and transforms."""

from typing import Any

import pytest

from agintis.modules.knowledge.api import (
    InvalidProfileError,
    RagProfileRegistry,
    UnknownStrategyError,
    parse_profile,
    validate_profiles,
)
from agintis.modules.knowledge.infrastructure.registry import (
    build_chunkers,
    build_extractors,
    chunker_param_schemas,
    extractor_param_schemas,
    transform_param_schemas,
)
from tests.unit.knowledge.infrastructure.test_chunker import WordCounter

LEVELS = [{"name": "item", "pattern": r"^(?P<num>\d+)\.\s+(?P<head>\S.*)$"}]


def _profile(fields: list[dict[str, Any]], **overrides: Any) -> Any:
    raw: dict[str, Any] = {
        "key": "p",
        "top_k": 3,
        "chunk_max_tokens": 100,
        "parser_key": "simple",
        "chunker_key": "pattern_hierarchy",
        "chunker_params": {"levels": LEVELS, "split_level": "item"},
        "metadata_fields": fields,
    }
    raw.update(overrides)
    return parse_profile(raw, origin="p.yaml")


def _validate(profile: Any) -> None:
    validate_profiles(
        RagProfileRegistry([profile]),
        ["simple"],
        build_chunkers(WordCounter()),
        chunker_schemas=chunker_param_schemas(),
        extractors=build_extractors(),
        extractor_schemas=extractor_param_schemas(),
        transform_schemas=transform_param_schemas(),
    )


def _field(name: str = "f", **kwargs: Any) -> dict[str, Any]:
    return {"name": name, **kwargs}


class TestGroupSources:
    def test_a_group_of_the_chunker_is_accepted(self) -> None:
        _validate(_profile([_field(source="group:num"), _field("h", source="group:head")]))

    def test_an_unknown_group_is_reported_with_the_field(self) -> None:
        with pytest.raises(InvalidProfileError, match="field 'f' reads group 'nope'"):
            _validate(_profile([_field(source="group:nope")]))

    def test_a_chunker_without_groups_rejects_group_sources(self) -> None:
        profile = _profile(
            [_field(source="group:num")], chunker_key="structured", chunker_params={}
        )
        with pytest.raises(InvalidProfileError, match="does not produce"):
            _validate(profile)

    def test_other_sources_need_no_group(self) -> None:
        _validate(
            _profile(
                [
                    _field(source="document:number"),
                    _field("b", source="parsed:keywords"),
                    _field("c", source="const:x"),
                ]
            )
        )


class TestExtractors:
    def test_known_extractor_with_valid_params(self) -> None:
        _validate(_profile([_field(source="extractor:regex_all", params={"pattern": r"\d+"})]))

    def test_unknown_extractor_key(self) -> None:
        with pytest.raises(UnknownStrategyError, match="extractor 'nope'"):
            _validate(_profile([_field(source="extractor:nope")]))

    def test_invalid_params(self) -> None:
        with pytest.raises(InvalidProfileError, match="extractor 'regex_all' params"):
            _validate(_profile([_field(source="extractor:regex_all", params={"group": 1})]))

    def test_selectors_are_checked(self) -> None:
        params = {"keys": {"libro": {"pattern": r"Libro (\w+)"}}, "layers": [{"field": "title"}]}
        named = {"extractors": {"target": {"key": "layered_regex", "params": params}}}
        _validate(_profile([_field(source="extractor:target.libro")], **named))
        with pytest.raises(InvalidProfileError, match="no selector 'titulo'"):
            _validate(_profile([_field(source="extractor:target.titulo")], **named))
        with pytest.raises(InvalidProfileError, match="needs a selector"):
            _validate(_profile([_field(source="extractor:target")], **named))
        with pytest.raises(InvalidProfileError, match="no selector 'x'"):
            _validate(
                _profile(
                    [_field(source="extractor:regex_all.x", params={"pattern": r"\d"})],
                )
            )

    def test_named_extractors_are_validated_once_with_their_own_message(self) -> None:
        bad = {"extractors": {"target": {"key": "regex_all", "params": {"nope": 1}}}}
        with pytest.raises(InvalidProfileError, match="extractor 'target' \\('regex_all'\\)"):
            _validate(_profile([_field(source="extractor:target")], **bad))

    def test_a_named_extractor_with_an_unknown_key(self) -> None:
        bad = {"extractors": {"target": {"key": "nope"}}}
        with pytest.raises(UnknownStrategyError, match="extractor 'nope'"):
            _validate(_profile([_field(source="extractor:target")], **bad))

    def test_params_on_a_field_that_uses_a_named_extractor_are_rejected(self) -> None:
        named = {"extractors": {"t": {"key": "regex_all", "params": {"pattern": "x"}}}}
        with pytest.raises(InvalidProfileError, match="belong on the named extractor"):
            _validate(_profile([_field(source="extractor:t", params={"pattern": "y"})], **named))


class TestTemplateReferences:
    def _template(self, template: str, required: list[str] | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {"template": template}
        if required:
            params["required"] = required
        return _field("path", source="extractor:template", params=params)

    def test_groups_and_earlier_fields_are_readable(self) -> None:
        fields = [
            _field("system", source="const:SP"),
            self._template("{system}/{num}{head?/{head}}", required=["system"]),
        ]
        _validate(_profile(fields))

    def test_a_later_field_is_not_readable_yet(self) -> None:
        fields = [self._template("{system}/{num}"), _field("system", source="const:SP")]
        with pytest.raises(InvalidProfileError, match=r"reads \['system'\]"):
            _validate(_profile(fields))

    def test_an_unknown_name_is_reported_with_the_field(self) -> None:
        with pytest.raises(InvalidProfileError, match=r"field 'path'.*reads \['ghost'\]"):
            _validate(_profile([self._template("{ghost}")]))

    def test_required_names_are_checked_too(self) -> None:
        with pytest.raises(InvalidProfileError, match=r"reads \['ghost'\]"):
            _validate(_profile([self._template("{num}", required=["ghost"])]))

    def test_named_template_extractors_are_checked(self) -> None:
        named = {"extractors": {"t": {"key": "template", "params": {"template": "{ghost}"}}}}
        with pytest.raises(InvalidProfileError, match=r"reads \['ghost'\]"):
            _validate(_profile([_field("path", source="extractor:t")], **named))

    def test_a_malformed_template_is_an_invalid_param(self) -> None:
        with pytest.raises(InvalidProfileError, match="malformed placeholder"):
            _validate(_profile([self._template("{num:0}")]))


class TestMinimumChunkSize:
    def _validate_with_minimums(self, tokens: int, key: str = "pattern_hierarchy") -> None:
        chunkers = build_chunkers(WordCounter())
        profile = _profile([], chunk_max_tokens=tokens, chunker_key=key)
        if key != "pattern_hierarchy":
            profile = _profile([], chunk_max_tokens=tokens, chunker_key=key, chunker_params={})
        validate_profiles(
            RagProfileRegistry([profile]),
            ["simple"],
            chunkers,
            chunker_schemas=chunker_param_schemas(),
            chunker_min_tokens={k: c.min_chunk_max_tokens for k, c in chunkers.items()},
        )

    @pytest.mark.parametrize("key", ["pattern_hierarchy", "structured", "whole_document"])
    def test_a_budget_too_small_for_the_chunker_fails_at_boot(self, key: str) -> None:
        with pytest.raises(
            InvalidProfileError,
            match=f"chunk_max_tokens 23 is below the minimum 24 of chunker '{key}'",
        ):
            self._validate_with_minimums(23, key)

    def test_the_minimum_itself_is_accepted(self) -> None:
        self._validate_with_minimums(24)
        self._validate_with_minimums(24, "structured")

    def test_every_structural_chunker_declares_the_same_minimum(self) -> None:
        minimums = {c.min_chunk_max_tokens for c in build_chunkers(WordCounter()).values()}
        assert minimums == {24}


class TestTransforms:
    def test_valid_transform(self) -> None:
        _validate(
            _profile(
                [
                    _field(
                        source="document:area",
                        transforms=[{"key": "value_map", "params": {"map": {"a": "b"}}}],
                    )
                ]
            )
        )

    def test_unknown_transform(self) -> None:
        with pytest.raises(UnknownStrategyError, match="transform 'nope'"):
            _validate(_profile([_field(source="const:x", transforms=[{"key": "nope"}])]))

    def test_invalid_transform_params_name_the_position(self) -> None:
        profile = _profile(
            [_field(source="const:x", transforms=[{"key": "date_floor"}, {"key": "value_map"}])]
        )
        with pytest.raises(InvalidProfileError, match="transform #2 'value_map' params"):
            _validate(profile)

    def test_checks_are_skipped_when_no_registries_are_given(self) -> None:
        profile = _profile([_field(source="extractor:nope")])
        validate_profiles(
            RagProfileRegistry([profile]),
            ["simple"],
            build_chunkers(WordCounter()),
            chunker_schemas=chunker_param_schemas(),
        )
