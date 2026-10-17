# Evidence calibration sets (fastapi_es)

Dev sets for `python -m agintis.cli.calibrate_evidence`. They calibrate evidence thresholds only.
They must NEVER be used to tune prompts, retrieval settings or anything else about the agent, and
they are not the held-out `profiles/eval_questions.jsonl` (never read while writing these).

- `in_dev.jsonl`: 30 `in` rows (`in-fa-01..30`): 23 tutorial, 7 guia.
- `ood.jsonl`: 40 `out` rows (`out-fa-01..40`): 10 unrelated domains, 6 chitchat, 6 generic
  programming, 12 near-domain traps, 6 off-topic technical.
- Label criterion: `in` = the corpus (64 Markdown files under `samples/tutorial` and
  `samples/guia`) contains the answer; `out` = it does not.
- The `in` questions were written while reading the documents, so recall and AUROC are optimistic.
- A threshold is only valid for the corpus snapshot, embedding model and profiles recorded in the
  report `meta`. Recalibrate after a re-ingest.

## How `in_dev` was built

Each question was written after opening the target file and finding the answer in it. Phrasing is
varied (short, long, task style, a few typos or missing accents). At most one question per file, except debugging (2)
(29 distinct files). Covered: tutorial: cors, debugging, static-files, middleware, metadata, encoder,
response-status-code, request-forms, background-tasks, testing, request-files, header-params,
server-sent-events, dependencies__dependencies-with-yield, security__oauth2-jwt, sql-databases,
handling-errors, path-operation-configuration, body-updates, path-params-numeric-validations,
query-param-models, bigger-applications. guia: fastapi-cli, async, virtual-environments,
benchmarks, project-generation, editor-support, python-types.

## Near-domain `out` verification

Method: for each trap, `rg --no-ignore -i` over `samples/` with the distinctive terms; a verdict is
`out` only if no passage answers the question (a passing mention does not count). Questions the
corpus answered were dropped.

| trap | rg term -> hits | verdict |
|---|---|---|
| Django ORM select_related | `select_related` -> 0 (Django appears only in guia/alternatives and history) | out |
| Flask Blueprint url_prefix/templates | `blueprint` -> 1 (tutorial/bigger-applications: "APIRouter is the equivalent of Flask Blueprints") | out, borderline: only an equivalence note, no Blueprint usage |
| Express middleware | `express` -> 0 real hits (only "expresiones regulares") | out |
| React useEffect | `useEffect` -> 0; `react` -> 4 (frontend frameworks list) | out |
| Kubernetes Deployment | `kubernetes` -> 0 | out |
| WebSocket endpoint | `websocket` -> 4 (feature bullets, no how-to) | out |
| Gunicorn + Uvicorn workers | `gunicorn` -> 0; `--workers` -> 1 (guia/alternatives, one sentence about Uvicorn) | out |
| Dockerfile | `dockerfile` -> 0; `docker` -> 2 (project-generation feature bullets) | out |
| Celery + Redis + retries | `celery` -> 1, `redis` -> 1 (tutorial/background-tasks "Advertencia": mentions them as alternatives, no setup) | out, borderline: retries/broker config absent |
| SQL tuning EXPLAIN ANALYZE | `EXPLAIN` -> 0 | out |
| Rate limiting | `rate.?limit` -> 0 | out |
| Prometheus metrics | `prometheus` -> 0 | out |

Dropped: a "lifespan pool at startup" question (`lifespan` -> 0, but tutorial/sql-databases shows
`@app.on_event("startup")`, which answers a close variant); replaced by the Prometheus trap.
Technical off-topic items (Excel, Linux, cron, ufw) were also checked: `BUSCARV|VLOOKUP`, `ufw`,
`usermod` -> 0 hits (`cron` only matches inside "sincron...").

The unrelated, chitchat and generic-programming items were not checked chunk by chunk.
