from pathlib import Path

import pytest
from pydantic import ValidationError

from agintis.bootstrap.settings import (
    EmbeddingsSettings,
    ParserSettings,
    RerankerSettings,
    Settings,
)
from agintis.modules.knowledge.api import (
    DEFAULT_PROFILE,
    DEFAULT_RERANK_CANDIDATES,
    Parser,
    RagProfile,
    RagProfileRegistry,
    Reranker,
    UnknownStrategyError,
    validate_profiles,
)
from agintis.modules.knowledge.application import NoopReranker
from agintis.modules.knowledge.infrastructure.chunkers import HfTokenCounter
from agintis.modules.knowledge.infrastructure.registry import (
    build_chunkers,
    build_parsers,
    build_rerankers,
)
from agintis.modules.knowledge.infrastructure.tei import FallbackReranker
from tests.support.settings import TEST_DATABASE_URL, make_settings


def _parsers() -> dict[str, Parser]:
    return build_parsers(
        max_bytes=1000, max_pages=10, max_uncompressed_bytes=5000, max_compression_ratio=50.0
    )


def _rerankers(url: str | None) -> dict[str, Reranker]:
    return build_rerankers(tei_url=url, timeout_seconds=5.0, deadline_seconds=10.0)


def test_registries_cover_the_default_profile() -> None:
    parsers = _parsers()
    chunkers = build_chunkers(HfTokenCounter("unused"))
    rerankers = _rerankers(None)
    validate_profiles(RagProfileRegistry(), parsers, chunkers, rerankers)
    assert (DEFAULT_PROFILE.parser_key, DEFAULT_PROFILE.chunker_key) == ("simple", "structured")


def test_tei_reranker_is_registered_only_with_a_url() -> None:
    disabled = _rerankers(None)
    assert set(disabled) == {"none"}
    assert isinstance(disabled["none"], NoopReranker)
    enabled = _rerankers("http://reranker:80")
    assert set(enabled) == {"none", "tei"}
    assert isinstance(enabled["tei"], FallbackReranker)  # degrades instead of failing retrieval


def test_unregistered_reranker_key_still_fails_fast() -> None:
    profile = RagProfile("p", 3, 100, "simple", "structured", reranker_key="tei")
    parsers = _parsers()
    chunkers = build_chunkers(HfTokenCounter("unused"))
    rerankers = _rerankers(None)
    with pytest.raises(UnknownStrategyError, match="tei"):
        validate_profiles(RagProfileRegistry([profile]), parsers, chunkers, rerankers)


class TestSettings:
    def test_defaults(self) -> None:
        settings = Settings.isolated(database_url=TEST_DATABASE_URL)
        assert settings.embeddings.url == "http://localhost:8081"
        assert settings.embeddings.batch_size == 16
        assert settings.embeddings.tokenizer_source == "BAAI/bge-m3"
        assert (settings.embeddings.query_prefix, settings.embeddings.document_prefix) == ("", "")
        assert settings.embeddings.expected_dim == 1024
        assert settings.embeddings.api_key is None
        assert settings.reranker.url is None
        assert settings.reranker.max_candidates == DEFAULT_RERANK_CANDIDATES
        assert settings.parser.max_bytes == 50 * 1024 * 1024
        assert settings.parser.max_pages == 2000
        assert not hasattr(settings.parser, "default_key")

    def test_knowledge_profiles_dir_defaults_to_none_and_reads_a_flat_env_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("KNOWLEDGE_PROFILES_DIR", raising=False)
        assert Settings.isolated(database_url=TEST_DATABASE_URL).knowledge.profiles_dir is None
        monkeypatch.setenv("KNOWLEDGE_PROFILES_DIR", "/etc/agintis/profiles")
        settings = Settings.isolated(database_url=TEST_DATABASE_URL)
        assert settings.knowledge.profiles_dir == Path("/etc/agintis/profiles")
        monkeypatch.setenv("KNOWLEDGE_PROFILES_DIR", "")
        assert Settings.isolated(database_url=TEST_DATABASE_URL).knowledge.profiles_dir is None

    def test_flat_env_names(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EMBEDDINGS_URL", "http://embeddings:80/")
        monkeypatch.setenv("EMBEDDINGS_BATCH_SIZE", "8")
        monkeypatch.setenv("EMBEDDINGS_QUERY_PREFIX", "query: ")
        monkeypatch.setenv("RERANKER_URL", "http://reranker:80")
        monkeypatch.setenv("PARSER_MAX_PAGES", "50")
        monkeypatch.setenv("EMBEDDINGS_TOKENIZER", "/models/tokenizer.json")
        settings = Settings.isolated(database_url=TEST_DATABASE_URL)
        assert settings.embeddings.url == "http://embeddings:80"
        assert settings.embeddings.batch_size == 8
        assert settings.embeddings.query_prefix == "query: "
        assert settings.embeddings.tokenizer_source == "/models/tokenizer.json"
        assert settings.reranker.url == "http://reranker:80"
        assert settings.parser.max_pages == 50

    def test_empty_reranker_url_means_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RERANKER_URL", "")
        assert RerankerSettings(_env_file=None).url is None

    def test_api_keys_are_secret(self) -> None:
        settings = Settings.isolated(
            database_url=TEST_DATABASE_URL, embeddings={"api_key": "top-secret-key"}
        )
        assert settings.embeddings.api_key is not None
        assert settings.embeddings.api_key.get_secret_value() == "top-secret-key"
        assert "top-secret-key" not in repr(settings)

    def test_group_overrides_accept_mappings_and_instances(self) -> None:
        from_mapping = Settings.isolated(
            database_url=TEST_DATABASE_URL, embeddings={"batch_size": 4}
        )
        from_instance = Settings.isolated(
            database_url=TEST_DATABASE_URL, parser=ParserSettings(_env_file=None, max_pages=7)
        )
        assert from_mapping.embeddings.batch_size == 4
        assert from_instance.parser.max_pages == 7

    @pytest.mark.parametrize(
        "factory",
        [
            lambda: EmbeddingsSettings(_env_file=None, url="localhost:8081"),
            lambda: EmbeddingsSettings(_env_file=None, url="ftp://host"),
            lambda: EmbeddingsSettings(_env_file=None, batch_size=0),
            lambda: EmbeddingsSettings(_env_file=None, expected_dim=0),
            lambda: RerankerSettings(_env_file=None, url="reranker"),
            lambda: RerankerSettings(_env_file=None, max_candidates=0),
            lambda: ParserSettings(_env_file=None, max_bytes=0),
            lambda: ParserSettings(_env_file=None, max_compression_ratio=1),
        ],
    )
    def test_invalid_values_are_rejected(self, factory: object) -> None:
        with pytest.raises(ValidationError):
            factory()  # type: ignore[operator]


class TestEnvFileIsolation:
    """A developer's local .env must never leak into tests (nested groups load it themselves)."""

    @staticmethod
    def _write_env(directory: Path) -> None:
        (directory / ".env").write_text(
            "EMBEDDINGS_URL=http://leaked-from-dotenv:9\n"
            "EMBEDDINGS_QUERY_PREFIX=leak: \n"
            "RERANKER_URL=http://leaked-reranker:9\n"
            "PARSER_MAX_PAGES=3\n"
            f"DATABASE_URL={TEST_DATABASE_URL}\n"
        )

    def test_plain_settings_do_read_the_dotenv(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Control: proves the simulation is faithful, otherwise the next test proves nothing."""
        self._write_env(tmp_path)
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("DATABASE_URL", raising=False)
        leaky = Settings(_env_file=None, database_url=TEST_DATABASE_URL)
        assert leaky.embeddings.url == "http://leaked-from-dotenv:9"

    def test_isolated_settings_ignore_the_dotenv(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._write_env(tmp_path)
        monkeypatch.chdir(tmp_path)
        for name in ("EMBEDDINGS_URL", "RERANKER_URL", "PARSER_MAX_PAGES", "DATABASE_URL"):
            monkeypatch.delenv(name, raising=False)
        isolated = Settings.isolated(database_url=TEST_DATABASE_URL)
        assert isolated.embeddings.url == "http://localhost:8081"
        assert isolated.embeddings.query_prefix == ""
        assert isolated.reranker.url is None
        assert isolated.parser.max_pages == 2000

    def test_make_settings_is_isolated_too(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._write_env(tmp_path)
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("EMBEDDINGS_URL", raising=False)
        assert make_settings().embeddings.url == "http://localhost:8081"
