"""Per-question W1 routing plus the hybrid hits of every selected topic (#1443). Read-only, no LLM.

usage (repo root): uv run --env-file .env python case/spensiones/tools/dump.py <out.json> <set.jsonl>...
Env: AGX_PROFILES (profiles dir), AGX_RERANK_N (rerank candidates, unset = off), AGX_PROFILE_MAP="a=b,c=d", AGX_AGENTS.
A set whose rows have no `profile` (bank_dev) is read with profile "none". Rows: {set, id, question, expected, routed: [{topic, score, hits: [{id, text, relevant}]}]}.
"""

import asyncio
import json
import os
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

from agintis.agent_kit.nodes import EvidenceTopic, EvidenceTopicRouter, RouteScope
from agintis.agent_kit.runtime import QueryVectors
from agintis.bootstrap.container import Container
from agintis.bootstrap.settings import get_settings
from agintis.cli.eval import eval_profile
from agintis.cli.eval_metrics import is_relevant, load_questions
from agintis.modules.agents.rag_chat import load_agent_specs
from agintis.shared_kernel.ids import TenantId

REPO = Path(__file__).resolve().parents[3]
# Rerank experiment (#1467): AGX_PROFILES = profiles dir, AGX_RERANK_N = candidates (unset = off).
PROFILES = Path(os.environ.get("AGX_PROFILES", REPO / "case/spensiones/profiles"))
RERANK_N = int(os.environ.get("AGX_RERANK_N", "0"))
# AGX_PROFILE_MAP="a=b,c=d": remap a question's gold profile (single-profile run of #1474).
PROFILE_MAP = dict(kv.split("=") for kv in os.environ.get("AGX_PROFILE_MAP", "").split(",") if kv)
AGENTS = Path(os.environ.get("AGX_AGENTS", REPO / "case/spensiones/agents"))
KS = [1, 3, 5, 10]
TENANT = TenantId("spensiones")


def questions(path: Path) -> list:
    """`load_questions` needs a `profile` per row; sets without one (bank_dev) get "none" via a temp copy."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if all("profile" in r for r in rows):
        return load_questions(path)
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / path.name
        copy.write_text("\n".join(json.dumps({"profile": "none", **r}, ensure_ascii=False) for r in rows),
                        encoding="utf-8")
        return load_questions(copy)


async def main() -> None:
    out, sets = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]
    settings = get_settings()
    settings = settings.model_copy(update={
        "knowledge": settings.knowledge.model_copy(update={"profiles_dir": PROFILES}),
        "database": settings.database.model_copy(update={"read_only": True}),
        "checkpointer": settings.checkpointer.model_copy(update={"enabled": False}),
        "agents": settings.agents.model_copy(update={"dir": None}),
        "reranker": settings.reranker.model_copy(update={
            "url": "http://localhost:8083" if RERANK_N else None,
            "max_candidates": RERANK_N or settings.reranker.max_candidates}),
    })
    spec = next(s for s in load_agent_specs(AGENTS) if len(s.topics) > 1)
    profile_of = {t.name: t.profile for t in spec.topics}
    topic_of = {t.profile: t.name for t in spec.topics}
    rows = []
    async with Container.build(settings) as c:
        router = EvidenceTopicRouter(
            c.probe_profiles,
            [EvidenceTopic(t.name, c.profiles.get(t.profile), t.evidence_bias) for t in spec.topics],
            spec.max_topics, spec.evidence_margin, spec.evidence_aggregate, spec.probe_k,
            settings.knowledge.max_query_chars,
        )
        for path in sets:
            for q in questions(path):
                q = replace(q, profile=PROFILE_MAP.get(q.profile, q.profile))
                scope = RouteScope(TENANT, QueryVectors())
                routed = await router.route(q.question, scope)
                vector = scope.vectors.get(q.question)
                topics = []
                for r in routed:
                    profile = eval_profile(c, profile_of[r["topic"]], "hybrid", KS, bool(RERANK_N))
                    result = await c.retrieve_for_topic.execute(
                        TENANT, profile, q.question, query_vector=vector)
                    topics.append({"topic": r["topic"], "score": r["confidence"], "hits": [
                        {"id": h.citation.chunk_id, "text": h.chunk.chunk.text,
                         "relevant": is_relevant(h.chunk.chunk.text, h.chunk.chunk.metadata, q)}
                        for h in result.hits]})
                rows.append({"set": path.stem, "id": q.id, "question": q.question,
                             "expected": topic_of.get(q.profile), "routed": topics})
    out.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    print(f"{len(rows)} rows -> {out}")


if __name__ == "__main__":
    asyncio.run(main())
