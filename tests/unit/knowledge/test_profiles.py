import pytest

from agintis.modules.knowledge.api import (
    DEFAULT_PROFILE,
    RagProfile,
    RagProfileRegistry,
    UnknownStrategyError,
    validate_profiles,
)
from agintis.shared_kernel.errors import DomainError, NotFoundError


def _profile(**overrides: object) -> RagProfile:
    base: dict[str, object] = {
        "name": "p",
        "top_k": 3,
        "chunk_max_tokens": 100,
        "parser_key": "x",
        "chunker_key": "y",
    }
    return RagProfile(**{**base, **overrides})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "overrides",
    [
        {"top_k": 0},
        {"chunk_max_tokens": 0},
        {"chunker_params": {"": 1}},
        {"name": ""},
        {"parser_key": ""},
        {"reranker_key": ""},
    ],
)
def test_profile_invariants(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError, match=r"."):
        _profile(**overrides)


def test_registry_has_default_profile() -> None:
    profile = RagProfileRegistry().get("default")
    assert profile == DEFAULT_PROFILE
    assert (profile.top_k, profile.chunk_max_tokens) == (5, 512)
    assert (profile.parser_key, profile.chunker_key) == ("simple", "structured")
    assert profile.reranker_key == "none"
    assert dict(profile.chunker_params) == {"include_heading": True}


def test_registry_register_get_and_errors() -> None:
    registry = RagProfileRegistry()
    registry.register(_profile(name="legal"))
    assert registry.get("legal").name == "legal"
    with pytest.raises(NotFoundError):
        registry.get("missing")
    with pytest.raises(DomainError):
        registry.register(_profile(name="legal"))


def test_registry_all_lists_registered_profiles() -> None:
    registry = RagProfileRegistry()
    registry.register(_profile(name="legal"))
    assert [p.name for p in registry.all()] == ["default", "legal"]


class TestValidateProfiles:
    def test_passes_when_every_key_is_registered(self) -> None:
        validate_profiles(RagProfileRegistry(), ["simple"], ["structured"])

    def test_reports_every_missing_strategy(self) -> None:
        registry = RagProfileRegistry([_profile(name="a"), _profile(name="b", parser_key="ok")])
        with pytest.raises(UnknownStrategyError) as excinfo:
            validate_profiles(registry, ["ok"], [])
        message = str(excinfo.value)
        assert "profile 'a': parser 'x'" in message
        assert "profile 'a': chunker 'y'" in message
        assert "profile 'b': chunker 'y'" in message
        assert "profile 'b': parser" not in message

    def test_reports_unknown_reranker(self) -> None:
        registry = RagProfileRegistry([_profile(reranker_key="tei")])
        with pytest.raises(UnknownStrategyError, match="reranker 'tei'"):
            validate_profiles(registry, ["x"], ["y"])
        validate_profiles(registry, ["x"], ["y"], ["none", "tei"])


class TestFingerprint:
    def test_is_stable_and_ignores_param_order(self) -> None:
        a = _profile(chunker_params={"overlap": 10, "mode": "x"})
        b = _profile(chunker_params={"mode": "x", "overlap": 10})
        assert (
            a.fingerprint
            == b.fingerprint
            == _profile(chunker_params={"mode": "x", "overlap": 10}).fingerprint
        )

    @pytest.mark.parametrize(
        "overrides",
        [
            {"parser_key": "z"},
            {"chunker_key": "z"},
            {"chunk_max_tokens": 101},
            {"chunker_params": {"overlap": 1}},
        ],
    )
    def test_changes_with_each_ingestion_input(self, overrides: dict[str, object]) -> None:
        assert _profile(**overrides).fingerprint != _profile().fingerprint

    def test_ignores_retrieval_only_fields(self) -> None:
        assert _profile(top_k=9).fingerprint == _profile().fingerprint
        assert _profile(reranker_key="tei").fingerprint == _profile().fingerprint
