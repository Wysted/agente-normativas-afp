# Measurement tools (spensiones)

The scripts behind every experiment in [`../calibration/README.md`](../calibration/README.md), so a
result can be reproduced and audited from the repo. Run them from the repo root. Run outputs
(`case/spensiones/judge/`) are gitignored: they hold corpus text.

| Tool | Answers | Input | Output |
|---|---|---|---|
| `dump.py` | What does the router + hybrid (+ TEI rerank) retrieve offline? No LLM. | `<out.json> <set.jsonl>...` | rows `{set, id, expected, routed: [{topic, score, hits}]}` |
| `routed_report.py` | MRR@10 of the merged top 8 by gold system, SP/SC confusion | `dump.py` outputs | stdout |
| `ncg_mrr.py` | Pooled and per-set MRR@10 (imports `merged` from `routed_report`) | `dump.py` outputs | stdout |
| `rerank_scores.py` | Max rerank score per question (bands input) | `<out.json> <set.jsonl>...` | rows `{set, id, S, H, topics, conf, top_text}` (`top_text` is corpus text) |
| `live_guard.sh` | Live answers per set (+ judge for in_dev, ood, oficio_dev) | `<run> [set...]` | `judge/e2e/<run>/<set>_answers.jsonl`, reports; logs in `judge/logs/<run>` |
| `live_hit.py` | Live hit: the dev snippet appears in a source the model saw | `<set.jsonl> <answers.jsonl>...` | per-phrasing counts, flips between two dumps |
| `noninferiority.py` | Pre-registered paired non-inferiority decision | `B1,B2,B3 N1,N2,N3 [--sets] [--bank-relevance] [--new-family]` | PASS/FAIL, stable losses to adjudicate; `--self-test` |
| `tsearch.sh`, `q.sh` | Accent-insensitive chunk text search (verifies content-only `out` labels) | regex / SQL where-clause | stdout (via `docker exec` psql) |

## Environment

- `dump.py` and `rerank_scores.py` need the app env: `uv run --env-file .env python case/spensiones/tools/<tool>.py ...`.
  `live_hit.py`, `routed_report.py`, `ncg_mrr.py` and `noninferiority.py` are plain `python3`.
- `LLM_PROFILES_FILE=case/spensiones/llm_profiles.yaml`, `RERANKER_URL=http://localhost:8083`,
  `RERANKER_MAX_CANDIDATES=40`, `OBSERVABILITY_TRACING=none` (`live_guard.sh` exports them).
  `dump.py` and `rerank_scores.py` ignore the two `RERANKER_*` variables: they fix the reranker at
  `http://localhost:8083` in code (40 candidates in `rerank_scores.py`, `AGX_RERANK_N` in `dump.py`).
- `live_guard.sh` writes each dump to `*.partial` and renames it when it finishes; a set whose
  `<set>_answers.jsonl` exists is skipped, so delete it to redo the set.
- `live_guard.sh` maps `LLM_API_KEY` to `OPENAI_API_KEY` (named openai profiles read the latter) and
  drops an exported but empty key, which would otherwise win over `.env`. `LG_AGENTS=<dir>` runs
  another agents dir; `LG_JUDGE=0` skips the judge. A dump whose log shows "Reranker unavailable"
  fell back to vector order: it is moved to `*.degraded` and the run aborts.
- `dump.py`: `AGX_PROFILES`, `AGX_RERANK_N`, `AGX_PROFILE_MAP`, `AGX_AGENTS`. The current profiles
  declare the `tei` reranker, so `AGX_RERANK_N=40` is required (unset turns the reranker off and
  profile validation fails). Sets without a `profile`
  (bank_dev) are read as profile `none`. `rerank_scores.py`: `PROBE_TENANT`, `AGX_AGENTS`.
- `tsearch.sh`, `q.sh`: `TENANT` (default `spensiones`), `DB_CONTAINER`.

## Experiment recipe

1. Pre-register the rules (thresholds, sets, decision criteria) before measuring.
2. Baselines: B runs with `live_guard.sh` (3 runs for the noise range).
3. Offline stage: `dump.py` before/after, then `ncg_mrr.py` / `routed_report.py`.
4. Bands: `python -m agintis.cli.calibrate_evidence --analyze` (inputs from `rerank_scores.py`).
5. Live runs: `live_guard.sh <N run> <sets...>`, 3 runs.
6. Judges: `python -m agintis.cli.judge --judges ...` (e.g. relevance on bank_dev ->
   `<run>/bank_dev_relevance.json`).
7. Decide: `noninferiority.py B1,B2,B3 N1,N2,N3` (add `--bank-relevance` for decision (b),
   `--new-family` for the URI marking the new source family). Stable losses are adjudicated by content.

Answer dumps record per-model token usage, so the cost of a run is auditable from them.
