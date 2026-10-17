"""The eval CLI around the metrics: `evaluate`, the report and its provenance (no database)."""

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from agintis.bootstrap.container import Container
from agintis.cli._common import git_revision
from agintis.cli.eval import evaluate, provenance, report_for
from agintis.cli.eval_metrics import EvalQuestion, QuestionOutcome
from agintis.modules.knowledge.api import (
    DENSE_RETRIEVAL_MODE,
    HYBRID_RETRIEVAL_MODE,
    IngestDocument,
    RagProfile,
    RagProfileRegistry,
    RetrieveForTopic,
    SourceFile,
)
from agintis.modules.knowledge.application import NoopReranker
from agintis.modules.knowledge.infrastructure.registry import build_retrieval_modes
from agintis.shared_kernel.ids import TenantId
from tests.fakes.clock import FixedClock
from tests.fakes.knowledge import (
    FakeChunker,
    FakeParser,
    InMemoryChunkRepository,
    InMemoryDocumentRepository,
    InMemoryIngestLock,
)
from tests.support.settings import make_settings

TENANT = TenantId("t")
TEXT = "\n\n".join(f"paragraph {i}" for i in range(10))


class _MarkerReranker(NoopReranker):
    """Registered under the profile's key: the spy shows whether the eval kept or dropped it."""


async def _container() -> Container:
    documents, chunks = InMemoryDocumentRepository(), InMemoryChunkRepository()
    embeddings = DeterministicFakeEmbedding(size=16)
    profile = RagProfile("p", 3, 100, "fake", "fake", reranker_key="rev")
    await IngestDocument(
        documents,
        chunks,
        {"fake": FakeParser()},
        {"fake": FakeChunker()},
        embeddings,
        FixedClock(),
        InMemoryIngestLock(),
    ).execute(TENANT, profile, SourceFile("file:///d.txt", "text/plain", TEXT.encode()))
    retrieve = RetrieveForTopic(
        documents,
        chunks,
        embeddings,
        rerankers={"none": NoopReranker(), "rev": _MarkerReranker()},
        retrieval_modes=build_retrieval_modes(branch_depth=20, rrf_k=60),
    )
    stub = SimpleNamespace(profiles=RagProfileRegistry([profile]), retrieve_for_topic=retrieve)
    return cast(Container, stub)


def _question(qid: str, snippet: str, **locator: object) -> EvalQuestion:
    return EvalQuestion(qid, snippet, "semantic", "p", snippet, locator)


class TestEvaluate:
    async def test_each_question_is_ranked_by_the_first_relevant_hit(self) -> None:
        container = await _container()
        questions = [
            _question("hit", "paragraph 3"),
            _question("miss", "no such text anywhere"),
        ]
        outcomes = await evaluate(
            container,
            questions,
            mode=DENSE_RETRIEVAL_MODE,
            ks=(1, 3, 5),
            tenant=TENANT,
            rerank=False,
        )
        by_id = {o.question_id: o for o in outcomes}
        assert by_id["hit"].rank == 1
        assert by_id["miss"].rank is None
        assert all(o.retrieved == 5 for o in outcomes)  # depth = the largest k

    async def test_the_mode_overrides_the_profile_and_rerank_is_off_by_default(self) -> None:
        container = await _container()
        seen: list[tuple[str, str, int]] = []
        real = container.retrieve_for_topic.execute

        async def spy(tenant: TenantId, profile: RagProfile, query: str, *args: object) -> object:
            seen.append((profile.retrieval_mode, profile.reranker_key, profile.top_k))
            return await real(tenant, profile, query)

        container.retrieve_for_topic.execute = spy  # type: ignore[method-assign,assignment]
        q = [_question("a", "paragraph 3")]
        await evaluate(
            container, q, mode=HYBRID_RETRIEVAL_MODE, ks=(1, 7), tenant=TENANT, rerank=False
        )
        await evaluate(container, q, mode=DENSE_RETRIEVAL_MODE, ks=(2,), tenant=TENANT, rerank=True)
        assert seen == [(HYBRID_RETRIEVAL_MODE, "none", 7), (DENSE_RETRIEVAL_MODE, "rev", 2)]

    async def test_the_locator_decides_relevance(self) -> None:
        container = await _container()
        wrong = _question("w", "paragraph 3", page=99)
        (outcome,) = await evaluate(
            container, [wrong], mode=DENSE_RETRIEVAL_MODE, ks=(5,), tenant=TENANT, rerank=False
        )
        assert outcome.rank is None


class TestReport:
    def test_report_has_overall_profiles_and_per_question_rows(self) -> None:
        outcomes = [
            QuestionOutcome("a", "x", "semantic", 1, 10),
            QuestionOutcome("b", "y", "exact_reference", None, 10),
        ]
        report = report_for(outcomes, (1, 10))
        assert report["overall"]["questions"] == 2
        assert report["overall"]["hit_at"] == {"1": 0.5, "10": 0.5}
        assert report["overall"]["mrr_at"] == 10  # MRR is MRR@max(k)
        assert set(report["profiles"]) == {"x", "y"}
        assert [q["id"] for q in report["questions"]] == ["a", "b"]
        assert report["questions"][1]["rank"] is None


class TestProvenance:
    def test_the_report_records_what_is_needed_to_reproduce_a_number(self, tmp_path: Path) -> None:
        questions = tmp_path / "q.jsonl"
        questions.write_text("{}\n", encoding="utf-8")
        settings = make_settings(
            embeddings={"model": "bge-m3"},
            knowledge={
                "hybrid_branch_depth": 33,
                "hybrid_rrf_k": 44,
                "text_search_config": "simple",
            },
            database={"hnsw_ef_search": 77},
        )
        profile = RagProfile("p", 3, 100, "fake", "fake")
        info = provenance(settings, [profile], questions, revision="abc1234")
        assert info["git"] == "abc1234"
        assert info["embeddings_model"] == "bge-m3"
        assert (info["hybrid_branch_depth"], info["hybrid_rrf_k"]) == (33, 44)
        assert (info["text_search_config"], info["hnsw_ef_search"]) == ("simple", 77)
        assert info["profile_fingerprints"] == {"p": profile.fingerprint}
        assert len(info["questions_sha256"]) == 64

    def test_the_questions_hash_changes_with_the_file(self, tmp_path: Path) -> None:
        settings = make_settings()
        questions = tmp_path / "q.jsonl"
        questions.write_text("a", encoding="utf-8")
        first = provenance(settings, [], questions, revision=None)["questions_sha256"]
        questions.write_text("b", encoding="utf-8")
        assert provenance(settings, [], questions, revision=None)["questions_sha256"] != first

    def test_git_revision_is_none_outside_a_checkout(self, tmp_path: Path) -> None:
        assert git_revision(tmp_path) is None

    def test_git_revision_survives_a_missing_git(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PATH", "")
        assert git_revision() is None
