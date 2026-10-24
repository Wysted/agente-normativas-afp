"""Max rerank score per question over the evidence-routed topics' hits (#1514). Read-only, no LLM.

usage (repo root): uv run --env-file .env python case/spensiones/tools/rerank_scores.py OUT.json SET.jsonl...
Env: PROBE_TENANT (default spensiones), AGX_AGENTS (agents dir, default case/spensiones/agents).
A set without expected snippets (ood) only records S. Rows: {set, id, S, H, topics}.
"""

import asyncio
import json
import os
import sys
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
TENANT = TenantId(os.environ.get("PROBE_TENANT", "spensiones"))
KS = [1, 3, 5, 10]
RERANK_N = 40


def questions(path: Path) -> list[tuple[str, str, object | None]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if all("expected_snippet" in r for r in rows):
        return [(q.id, q.question, q) for q in load_questions(path)]
    return [(r["id"], r["question"], None) for r in rows]


async def main() -> None:
    out, sets = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]
    settings = get_settings()
    settings = settings.model_copy(update={
        "knowledge": settings.knowledge.model_copy(update={"profiles_dir": REPO / "case/spensiones/profiles"}),
        "database": settings.database.model_copy(update={"read_only": True}),
        "checkpointer": settings.checkpointer.model_copy(update={"enabled": False}),
        "agents": settings.agents.model_copy(update={"dir": None}),
        "reranker": settings.reranker.model_copy(update={"url": "http://localhost:8083", "max_candidates": RERANK_N}),
    })
    (spec,) = load_agent_specs(Path(os.environ.get("AGX_AGENTS", REPO / "case/spensiones/agents")))
    profile_of = {t.name: t.profile for t in spec.topics}
    rows = []
    async with Container.build(settings) as c:
        router = EvidenceTopicRouter(
            c.probe_profiles,
            [EvidenceTopic(t.name, c.profiles.get(t.profile), t.evidence_bias, t.supporting) for t in spec.topics],
            spec.max_topics, spec.evidence_margin, spec.evidence_aggregate, spec.probe_k,
            settings.knowledge.max_query_chars,
        )
        for path in sets:
            for qid, text, q in questions(path):
                scope = RouteScope(TENANT, QueryVectors())
                routed = await router.route(text, scope)
                vector = scope.vectors.get(text)
                scores, hit, topics, best = [], False, [], None
                for r in routed:
                    profile = eval_profile(c, profile_of[r["topic"]], "hybrid", KS, True)
                    result = await c.retrieve_for_topic.execute(TENANT, profile, text, query_vector=vector)
                    topics.append(r["topic"])
                    for h in result.hits:
                        if h.rerank_score is not None:
                            scores.append(h.rerank_score)
                            if best is None or h.rerank_score > best.rerank_score:
                                best = h
                        if q is not None and is_relevant(h.chunk.chunk.text, h.chunk.chunk.metadata, q):
                            hit = True
                rows.append({"set": path.stem, "id": qid, "S": max(scores, default=None),
                             "H": hit if q is not None else None, "topics": topics,
                             "conf": max((r.get("confidence") or 0.0 for r in routed), default=0.0),
                             "top_text": best.chunk.chunk.text if best else None})
    out.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    print(f"{len(rows)} rows -> {out}")


if __name__ == "__main__":
    asyncio.run(main())
