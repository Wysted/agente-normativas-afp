# spensiones: the Chilean pension regulations agent

`spensiones-rag` answers questions about the regulations of the Chilean pension system and
unemployment insurance (Superintendencia de Pensiones), citing the passages it used. Target user:
AFP operations and compliance staff. Agent config: [`agents/spensiones.yaml`](agents/spensiones.yaml);
measurements: [`calibration/README.md`](calibration/README.md).

## System

```mermaid
flowchart LR
    UI["Chat UI<br/>(agintis-spensiones-front, Vue)"]
    subgraph API["agintis API (FastAPI)"]
        TH["/v1/threads<br/>create, list, get, delete"]
        RUN["/v1/threads/{id}/runs/stream<br/>SSE: tokens, sources, scope, done"]
        MSG["/v1/threads/{id}/messages<br/>transcript + per-answer sources"]
    end
    G["spensiones-rag graph<br/>(LangGraph)"]
    PG[("Postgres + pgvector<br/>chunks: text, embedding, tsv<br/>threads, checkpoints")]
    EMB["TEI embeddings<br/>bge-m3"]
    RR["TEI reranker (GPU)<br/>bge-reranker-v2-m3"]
    LLM["OpenAI gpt-6-luna<br/>reasoning off"]
    PHX["Phoenix<br/>traces (optional)"]

    UI --> TH & RUN & MSG
    RUN --> G
    MSG --> PG
    G <--> PG
    G --> EMB & RR & LLM
    G -.-> PHX
```

- One run per thread at a time (a lease); the LangGraph checkpointer keeps the conversation
  (`messages`) and each answer's cited sources (`cited_sources`) in Postgres.
- The tenant comes from `X-Tenant-Id` (or a verified JWT claim); this agent's data is tenant
  `spensiones`.

## One turn

```mermaid
flowchart TD
    Q([user message]) --> AN["analyze (LLM)<br/>standalone query, request kind"]
    AN --> RT["route (no LLM)<br/>evidence router: probes every topic,<br/>picks up to 2 (3 in the low band)"]
    RT -- "out of scope, social<br/>or off-domain" --> RF["refuse<br/>fixed message, no retrieval"]
    RT -- "Send x N, in parallel" --> TP
    subgraph TP["topic subgraph (one per routed topic)"]
        EF["extract_filters (LLM)<br/>metadata filters"] --> RV["retrieve<br/>hybrid: full-text + dense, RRF,<br/>then cross-encoder rerank of 40"]
    end
    TP --> MG["merge<br/>rank interleave, primary topic x2;<br/>LLM tie-break when 2+ topics have sources"]
    MG --> SF{"sufficiency<br/>a source with rerank >= 0.92?"}
    SF -- "no, and rewrites left" --> RW["rewrite (LLM)<br/>a different query"]
    RW --> RT
    SF -- "yes, or no rewrites left" --> SY["synthesize (LLM, streamed)<br/>answer with [n] citations, up to 8 sources"]
    SY --> A([answer + sources])
    RF --> A
```

- The scope verdict (`in` / `low` / `out`) comes from the first route of the turn only, from the
  evidence router's cosine score: below 0.4779 is `out` (refused), below 0.6137 is `low` (one
  more topic searched). A rewrite never changes it.
- The rewrite loop keeps the evidence of earlier attempts: it is merged last into the next one.
  `max_rewrites` is 1 on `main`.
- The answer may only use the numbered sources; it keeps each rule's conditions and says it does
  not know when no source covers the question.

## Topics (one retrieval profile each)

| topic | content |
|---|---|
| `compendio` | Compendio de Normas, all Libros: the consolidated, current rules |
| `ncg` | Normas de Carácter General: the changes to the Compendio |
| `circular` | Circulares (1981 on) |
| `resolucion` | Resoluciones exentas |
| `oficio` | Oficios Ordinarios (2001-2024): how the SP resolved concrete cases |
| `jurisprudencia_nota` | Notas Internas (2021-2024) |
| `jurisprudencia_indice` | Index of oficios and dictámenes by subject |
| `articulo_faq` | Help articles for the public, in plain language |

The corpus is crawled from spensiones.cl ([`crawler/`](crawler/)), parsed by the profiles in
[`profiles/`](profiles/) and ingested with `agintis.cli.ingest`.

## How changes are measured

```mermaid
flowchart LR
    P["pre-register the criterion<br/>(engram)"] --> B["baseline runs<br/>live_guard.sh"]
    B --> C["change the prompt / config<br/>on a branch"]
    C --> N["new runs<br/>live_guard.sh"]
    N --> M["live hits (snippet in sources)<br/>+ LLM judge on in_dev, ood, oficio_dev"]
    M --> D{"floors from the<br/>run-to-run range"}
    D -- pass --> MG["merge to main"]
    D -- fail --> X["documented, not merged"]
```

A dump whose log shows a reranker fallback (`Reranker unavailable`) is invalid and redone.

## Status and known gaps (2026-10-24)

Works end to end: chat UI, threads with server-side history, streaming answers with citations,
scope refusal of off-domain questions, multi-topic retrieval with rerank, the rewrite loop.

Known gaps:
- **Sampled sources.** Circulares, NCG, resoluciones and help articles were ingested from a
  35-file sample, not the full listings (2,586 circulares and 483 NCG on spensiones.cl). Help
  articles at full coverage are measured but undecided (calibration README).
- **Colloquial and lay questions.** 77% of a colloquial question bank starts with a weak
  retrieval; most are register gaps (the norms phrase it differently), not missing content.
  A normative-register rewrite did not pass its measurement (see the calibration README).
- **Content outside the corpus.** AFP-internal procedures and systems (fixing a RUT, the
  "canje" in practice) and anything needing the member's own account (balance, projected
  pension, missing months) cannot be answered from the regulations.
- **No clarifying questions.** The thread API supports interrupts, but no production node asks
  the user; an unclear question is answered or refused.
- **Observability of the loop.** Answer dumps record the standalone query, not the rewritten
  queries or how many rewrites ran; cached tokens are not recorded.
- **Non-determinism.** The LLM profile sets no temperature (undocumented for gpt-6-luna):
  identical runs of a rewrite-heavy set varied by 8 hits out of 87.
- **Transcripts.** Threads answered before the transcript endpoint existed have no per-answer
  sources; a failed turn can leave a user message without an answer.
- **Front.** Not yet loading the transcript on reload.
