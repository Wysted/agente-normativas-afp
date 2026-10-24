# Evidence calibration sets (spensiones)

Dev sets for `python -m agintis.cli.calibrate_evidence` (never the held-out `eval_questions.jsonl`).

- `ood.jsonl`: 80 `out` queries. `in_dev.jsonl`: 40 `in` queries.
- Label criterion for `out` (fixed): the ingested corpus (tenant `default`) does NOT contain an
  answer to the question. `in`: it does.
- The `in` queries were written looking at the documents, so recall and AUROC are optimistic.
- A threshold is valid only for the corpus snapshot, tenant, embedding model and profiles in the
  report `meta`. Recalibrate after a re-ingest.

## Routing dev set 2 (`routing_dev2.jsonl`)

49 `in` questions in the eval format (`profile`, `expected_snippet`, `locator`, `type`) over the
34-document corpus of `samples/_manifest.json` (fetch it with `case/spensiones/fetch_samples.py`).
Use it for routing and retrieval experiments; it is a dev set, never a held-out one.

- Corpus: the original 8 samples plus a date-ordered selection made without looking at any score:
  10 historical CAFP circulars (rate tables excluded), 6 NCGs (NP 366-371), 7 FAQ-style articles
  and 3 normativa resolutions. Five articles whose body is loaded by JavaScript were dropped.
- Writing: three blind writers worked from the extracted document text, without access to the
  router, the probes or any score. At most 30% of the questions name a document.
- Checks: every snippet is in a ready chunk of its gold profile (mechanical check); 4 questions too
  close to `eval_questions.jsonl`, `in_dev.jsonl`, `retrieval_dev.jsonl` or `ood.jsonl` (token
  Jaccard >= 0.5) were dropped without reading those files into a prompt.
- Gold profiles: compendio 11, articulo_faq 9, ncg 9, circular 9, jurisprudencia_indice 8,
  resolucion 3.

Pre-registered results (2026-10-17), evidence router, tenant with this corpus:

| Router | hit@1 | any selected | compendio hit@1 | topics/question | old `retrieval_dev` hit@1 |
|---|---|---|---|---|---|
| max, no bias (current) | 34/49 | 39 | 11/11 | 1.22 | 34/39 |
| per-topic bias from the `ood.jsonl` medians | 38/49 | 42 | 9/11 | 1.20 | 32/39 |

The bias variant failed two of the five criteria (compendio and the old set may drop by at most
one question), so the router keeps `max` without biases. The set confirms the size bias: 10 of the
15 misses of the current router go to compendio.

## Afiliación dev set (`afiliacion_dev.jsonl`)

39 `in` questions in the same format, all answered by the compendio Libro I (Afiliación); `locator`
is empty on purpose (a hit is the snippet). It is a dev set for retrieval and answer experiments.

- Two slices, field `phrasing`: `v1_mixed` (af-01..19, about 60% citizen wording) and `operativa`
  (af-20..39, the target user: AFP operations or compliance staff who cite norms, Títulos and
  processes). `cites_identifier` exists only in `operativa` (12 of 20). Decisions use `operativa`.
- Writing: blind writers worked from the chunk text of each Título, without the router, the probes
  or any score. Some facts are asked once per slice (af-05/af-22, af-06/af-24, af-08/af-26).
- Checks: every snippet is in a Libro I chunk; no question is within token Jaccard 0.5 of
  `eval_questions.jsonl`, `in_dev.jsonl`, `routing_dev2.jsonl` or another row of this set.
- Fixed metadata issue (2026-10-20): some Títulos have no "Libro I, Título" breadcrumb, only a
  bare "Título XI ..." heading, so their bodies were tagged as the previous Título. In Libro I,
  V went from 4 to 70 chunks (66 had been tagged IV), XI from 0 to 79 and XII from 3 to 19. In
  Libro II, III went from 1 to 11. No other chunk changed Título.
  - Accepted as a documented exception: routed MRR@10 (router order, no tie-break) dropped on
    `routing_dev2` (.522 -> .508) and `afiliacion_dev` (.564 -> .548), below the floor of -0.01.
  - Two questions explain it (d2-ncg-02, af-37): compendio now ranks above ncg, and the
    matching source moves from 1st to 3rd of the 8 the model sees. Total hit@1 is unchanged.
- Live hit = the snippet, with all whitespace removed, appears in a source the model saw (the
  parser splits some words, e.g. "en lace").

Results (2026-10-18, tenant with the 34-document corpus, generator gpt-5.4-mini):

| Sources reach the model | `operativa` live hit | `v1_mixed` live hit |
|---|---|---|
| cut at 1500 chars (template default) | 16/20 | 11/19 |
| uncut (`max_source_chars` 3400) | 18/20 | 12/19 |

The two `operativa` gains (af-27, af-28) were retrieved before too: their answer sat past char
1500 of the chunk. In-dev judge results did not regress (faithfulness 37/37, refusal 38/40); ood
refusal went from 80/80 to 79/80.

## Cotizaciones dev set (`cotizaciones_dev.jsonl`)

20 `in` questions in the same format, all answered by the compendio Libro II (Cotizaciones
Previsionales), all in `operativa` voice (10 of 20 cite a real norm, Título or Capítulo). It is the
dev set of the cotizaciones area and the Libro II guard for routing changes: the other dev sets
only have Libro I gold, so a change that penalizes the compendio looks free on them.

- Writing: three blind writers, one question per window of 20 consecutive chunks, windows spread
  evenly over Títulos I, II, IV-X and XIII-XV (quota roughly by chunk count; III, XI and XII have
  1-9 chunks and were skipped).
- Checks: every snippet is in a Libro II chunk; no question is within token Jaccard 0.5 of
  `eval_questions.jsonl`, `in_dev.jsonl`, `routing_dev2.jsonl`, `retrieval_dev.jsonl`,
  `afiliacion_dev.jsonl` or another row of this set. Nothing was dropped.

Results (2026-10-19, tenant with Libro II, `evidence_margin` 0.05, generator gpt-5.4-mini): offline
routed hit@1 20/20 and MRR@10 .742; live hit 18/20. Ingesting Libro II did not move the afiliación
live hit (18/20, 12/19); the scope bands were recalibrated on this corpus (`report.json`).

## Merge tie-break (`merge_tiebreak: llm`)

The evidence router ranks topics by max probe cosine, and that score is biased towards the
compendio: right and wrong compendio wins look alike, so no score transform (margin, bias, lexical
overlap, a cross-encoder score per topic) fixed the primary choice offline. Most of the routed MRR
loss came from gold that was selected but merged as secondary. The tie-break lets the ROUTER model
read the top 2 passages of each group (labelled by letter, topic names hidden) and pick the primary
group of the merge.

Offline, routed MRR@10 (simulated from the hybrid hits, 2026-10-19):

| variant | `routing_dev2` | `retrieval_dev` | `afiliacion_dev` | `cotizaciones_dev` |
|---|---|---|---|---|
| router order (before) | .522 | .659 | .564 | .742 |
| LLM tie-break | .605 | .669 | .564 | .742 |
| cross-encoder rerank of the merged top 8 | .657 | .709 | .626 | .950 |

- The pre-registered floor also asked that compendio hit@1 drop by at most one question per set.
  The tie-break misses it on `afiliacion_dev` only (34 against 36), and was accepted as an
  exception: MRR does not move there.
- The cross-encoder rerank wins everywhere but took 21 s per question on CPU; it needs a GPU.
- Reruns (2026-10-20): two more tie-break runs gave identical picks, at .592 / .676 / .564 / .742
  with compendio hit@1 35 on `afiliacion_dev`, so the gain holds. They differ from the first run
  by a few picks: treat one LLM run as good to about one or two questions per set.
- Rejected variants:
  - a 3rd candidate (the next topic by router score): it recovers missed topics but loses
    cotizaciones, and with it every question costs a call;
  - `deepseek-flash` as the tie-break model (thinking off, temperature 0): below the floor on
    `retrieval_dev`, and not deterministic (51 of 56 picks agree between two runs).
- Live guard (same commands and models as the Libro II run): live hit 18/20, 12/19 and 18/20,
  unchanged; in-dev judge faithfulness 35/37, refusal 38/40, citation 39/39; ood refusal 78/80.
  The primary source changed in 10 of 179 questions. The three judge regressions had identical
  sources in both runs, so they come from the generator, not from the tie-break.
- Not measured: the `low` scope band, where the tie-break picks among 3 candidates.
  `cli.eval --route` ignores the tie-break.

## Cross-encoder rerank (`reranker_key: tei`)

Every profile reranks its candidates with `BAAI/bge-reranker-v2-m3` (TEI), at
`RERANKER_MAX_CANDIDATES=20`. On CPU it took about 20 s per question; on a GPU (RTX 5060 Ti, TEI
image `120-1.9.4`, `--max-batch-tokens 4096` so the GPU is not used flat out) 20 passages take
about 190 ms.

Offline (2026-10-20), MRR@10 on the 147 dev questions:

| | before | 20 candidates | 40 candidates |
|---|---|---|---|
| search on the gold topic (all sets) | .702 | .842 | .852 |
| — `afiliacion_dev` | .573 | .740 | .790 |
| — `cotizaciones_dev` | .750 | .967 | .963 |
| — `operativa` / `v1_mixed` | .711 / .469 | .946 / .546 | .969 / .595 |
| routed, router order (all sets) | .590 | .722 | .734 |
| retrieval p95 | 246 ms | 481 ms | 816 ms |

20 candidates was chosen: 40 adds less than the pre-registered +0.02.

- `max_rewrites` went to 0. With a reranker, `sufficiency: relevance` would start rewriting at
  the uncalibrated `min_rerank_score` 0.1; before the reranker the loop never fired.
- Live guard (against the tie-break run):
  - live hit went from 18/20, 12/19 and 18/20 to 19/20, 12/19 and 19/20;
  - in-dev faithfulness went from 35/37 to 37/38, and refusal stayed at 38/40;
  - ood refusal went from 78/80 to 79/80.
- Accepted as a documented exception: in-dev `citation_support` fell from 39/39 to 33/39, below
  the floor of 38.
  - In five of the six, the answer omits a condition its cited source states (in-cmp-08,
    in-cmp-09, in-ncg-17, in-cir-24, in-idx-37); in-cmp-12 is a quote that is not verbatim.
  - None states a fact the sources lack, and all six got different sources with the reranker.
  - The metric is noisy: between two earlier runs it moved from 33 to 39 with 36 of 40 source
    lists identical, so a floor of -1 was tighter than its run-to-run noise.
  - The operators' view: the cases that matter here concentrate in the compendio, where
    retrieval gained the most.
  - Next step: a synthesis prompt that keeps the conditions of the cited norm.
- The router probes are dense only, so the scope bands did not move.
- Deployment: set `RERANKER_MAX_CANDIDATES=40` (20 until the 10-Libro corpus; the app default is
  8). The TEI server must accept 40 texts per request: `--max-client-batch-size` of at least 40,
  and `--max-concurrent-requests` of at least 120, since TEI counts each text and up to three
  topics rerank in parallel (the `low` band). Otherwise every rerank fails and `FallbackReranker` silently keeps the vector
  order. The compose `reranker` service (CPU image, batch 16) does not meet this yet.

## New Libros dev set (`compendio_new_dev.jsonl`)

40 `in` questions on the compendio Libros added on 2026-10-21: SP III (10), SP IV (8), SP V (8)
and SC I-V (14). All are in `operativa` voice, and 21 of 40 cite a structural identifier.

- Writing: four blind writers (one per 10 windows) saw only chunk text, never the router, probes
  or scores. Each window is 20 consecutive chunks inside one Título, spread round-robin over the
  Títulos with at least 10 chunks.
- Checks: every snippet is relevant to its source chunk under the eval rule; no question is within
  token Jaccard 0.5 of another set or row. Nothing was dropped. 13 snippets also match a second
  chunk (repeated text), and either one counts as a hit.
- `profile` is `spensiones_compendio` for SP and `spensiones_compendio_sc` for SC, from a split
  that was measured and dropped. The single-profile runs map SC to `spensiones_compendio`.

Results (2026-10-21, reranker on, raw question). The routed column is MRR with the router
order and no tie-break:

| setup | new set, gold topic | new set, routed | existing 4 sets, routed |
|---|---|---|---|
| 1 compendio profile, 40 candidates | .891 | .891 | .607 / .699 / .710 / .888 |
| SP and SC split, 40 candidates | .913 | .896 | .607 / .691 / .693 / .863 |
| before the new Libros (R1) | n/a, not in the corpus | n/a | .626 / .724 / .715 / .967 |

- The split was dropped: it gained +0.005 routed on the new set, against a pre-registered +0.03,
  and lost 0.010 on the existing sets.
- Adding the Libros costs the existing sets up to about 0.08 routed MRR on cotizaciones. The new
  SP III-V chunks crowd Libro I and II answers out of the rerank candidates; 40 candidates
  recover part of it. It is an open trade-off, not a passed guard.
- The scope bands move with the corpus. On raw questions with all 10 Libros, at the current
  thresholds, `ood` items at or above `low_confidence_below` go from 4 to 8, and the p95 rule
  would put the threshold at 0.662 (0.610 now). Recalibrate with `--analyze` before shipping.

## Notas Internas (`guia_jurisprudencia_nota.yaml`, `nota_dev.jsonl`)

The 117 "Nota Interna N° ..." rows (index doc_type `guia_faq`) had never been downloaded, and
their profile was a `draft`, so the `jurisprudencia_nota` topic was empty.
- Activated after checking the 101 notes with text: 556 chunks, 0 failures.
- 16 are scans with no text layer. The profile uses `ocr_index_card`, which parses text-layer
  PDFs exactly like `simple` and runs RapidOCR on the scans. All 117 now ingest (636 chunks), and
  the OCR text has the same non-word share as the text-layer notes (median 0.14 against 0.12).
  - Ingest needs the `ocr` extra (`uv sync --inexact --extra ocr`; Docker: `UV_EXTRAS` with
    `ocr`). CI already installs every extra.
- `nota_dev.jsonl`: 10 blind questions. MRR@10 .850 on the topic and .850 routed. The router picks
  `jurisprudencia_nota` for 10 of 10, as primary for 9.
- Routed MRR@10 on the other 5 sets: pooled .737 against .739 without notes; routing_dev2 drops
  .007. The topic is selected for 13 of 187 other questions, as primary for 2.
- Unlike oficios, the notes are distinct enough for the evidence router.

## Live guard: 10 Libros + Notas Internas (2026-10-21)

Setup: all 10 compendio Libros in one profile, Notas Internas with OCR, 40 rerank candidates, and
the scope bands recalibrated (`low_confidence_below` 0.6621, `out_of_scope_below` 0.4732).
Baseline: the GPU rerank run. The floors were fixed before the run.

| measure | before | after | floor |
|---|---|---|---|
| `operativa` live hit | 19/20 | **17/20** | 18 |
| `v1_mixed` live hit | 12/19 | 11/19 | 11 |
| cotizaciones live hit | 19/20 | 18/20 | 18 |
| in-dev faithfulness | 37/38 | **34/38** | 36 |
| in-dev refusal | 38/40 | 38/40 | 37 |
| in-dev citation | 33/39 | 31/39 | 30 |
| ood refusal | 79/80 | 77/80 | 77 |
| `compendio_new_dev` live hit | not in the corpus | 39/40 | - |
| `nota_dev` live hit | not in the corpus | 10/10 | - |

- Accepted as a documented exception: two floors failed, `operativa` and faithfulness. In exchange,
  49 questions from areas that were missing are now answered.
- The `operativa` losses (af-01, af-28 and af-35, plus af-02 and af-36) match the offline crowding
  of Libro I by SP III-V. af-19 and af-34 were gained.
- Faithfulness: 2 of the 4 failures are judge quotes that are not verbatim (in-cmp-15, in-idx-38).
  The other 2 drop or merge a condition of the norm (in-ncg-21, in-cir-26); no answer states a fact
  the sources lack.
- Next: a synthesis prompt that keeps the conditions of the cited norm.
- Re-checked after the bare-Título fix (2026-10-21): every set's numbers are unchanged except one
  routed question of `compendio_new_dev` (nd-26 now routes to circular/ncg; .891 -> .866). The scope
  bands still hold: at 0.6621/0.4732, 12/40 in-domain questions fall in the `low` band, 4/80 ood
  questions pass as normal and none is refused.

## Synthesis prompt: keep the norm's conditions (2026-10-21)

`prompts.synthesize` gained two rules plus an invented example:
- Keep every condition a source puts on a rule, and apply the rule only to the subject, account
  or document the source names.
- Give each case (letras, numerales) with its own condition instead of chaining the cases, and say
  so when the source rule is about something nearby rather than transferring it.

It targets the 3 synthesis-side faithfulness failures seen so far (in-ncg-17, in-ncg-21,
in-cir-26). Each drops, swaps or transfers a condition. The criterion was fixed before the run.
Baseline: the 10 Libros + Notas run above, with the same retrieval. Live hits were not rerun: they
only read the retrieved sources, which a synthesis prompt cannot change.

| measure | before | after | rule |
|---|---|---|---|
| in-dev faithfulness | 34/38 | **36/38** | ≥ 36 (decides) |
| in-dev contradicted claims | 3 | **1** | ≤ 1 (decides; keeps judge noise from passing it alone) |
| in-dev refusal | 38/40 | 38/40 | ≥ 37 |
| in-dev relevance | 38/40 | 38/40 | ≥ 37 |
| in-dev citation | 31/39 | 35/38 | ≥ 28 |
| ood refusal | 77/80 | 77/80 | ≥ 76 |

- It passes, which closes the faithfulness part of the exception above. The `operativa` live hit
  (17/20) stays an exception.
- in-ncg-17 and in-ncg-21 now pass.
- in-cir-26 still fails: another source sets the valuation on the day the charges are recorded.
  It is a neighbouring rule, not a dropped condition.
- in-cmp-15 is still a judge quote that is not verbatim.
- Answers are about 21% longer (mean 467 -> 564 chars).

## Model: gpt-6-luna (2026-10-21)

Every LLM role (analyzer, merge tie-break, synthesis) moved from gpt-5.4-mini to OpenAI
gpt-6-luna with `reasoning_effort: none` (named profile `luna_none`). The list price is $0.10 /
$0.50 per million input / output tokens, against $0.75 / $4.50. The analyzer's `json_schema`
structured output works at both efforts on Chat Completions (0 dump errors).

The criterion was fixed before the runs. Baselines: the keep-conditions run (judge) and the
10 Libros + Notas run (live hits), with the same retrieval.

| measure | gpt-5.4-mini | luna `none` | luna `medium` | floor |
|---|---|---|---|---|
| in-dev faithfulness | 36/38 | **37/38** | 35/36 | ≥ 35 |
| in-dev contradicted claims | 1 | 0 | 0 | ≤ 2 |
| in-dev refusal | 38/40 | 38/40 | **36/40** | ≥ 37 |
| in-dev relevance | 38/40 | 38/40 | **36/40** | ≥ 37 |
| in-dev citation | 35/38 | 36/39 | 36/38 | ≥ 32 |
| ood refusal | 77/80 | 77/80 | 76/80 | ≥ 76 |
| `operativa` live hit | 17/20 | **19/20** | 19/20 | ≥ 16 |
| `v1_mixed` live hit | 11/19 | 12/19 | 12/19 | ≥ 10 |
| cotizaciones live hit | 18/20 | 18/20 | 18/20 | ≥ 17 |
| `compendio_new_dev` live hit | 39/40 | 39/40 | 39/40 | ≥ 38 |
| `nota_dev` live hit | 10/10 | 10/10 | 10/10 | ≥ 9 |

- `none` passes every floor and is kept. `medium` (the API default) fails refusal and relevance:
  reasoning tokens cost more and do not help here.
- `operativa` is back to 19/20, above the 18 floor that the 10 Libros + Notas exception failed.
- script_mix stays 40/40 and 80/80. Answers are a bit shorter (mean 564 -> 485 chars).
- Scope bands recalibrated with the Luna analyzer (`--analyze`, now `report.json`):
  `low_confidence_below` stays 0.6621, and `out_of_scope_below` goes 0.4732 -> 0.4795 (+0.0063,
  over the pre-registered 0.005 tolerance). AUROC .963. 4/80 ood questions pass as normal, 14/40
  in-domain questions fall in the `low` band, and none is refused.
- The in-dev and ood judge was rerun with the new bands against the same floors: faithfulness
  37/37, citation 35/38, ood refusal 76/80, script_mix 40/40 and 80/80. In-dev refusal and
  relevance came out at **36/40**, under the 37 floor.
- Accepted as a documented exception:
  - The in-dev scope split is identical in both Luna runs (28 `in`, 12 `low`, none refused), so
    for in-dev the two configurations are the same and 38 -> 36 is run-to-run noise. The judge
    moves by ±2 between identical runs, and a baseline −1 floor on 40 items cannot separate that.
  - To watch: in-cmp-08 (a mandate to a third party over the voluntary savings account) fails in
    2 of 3 Luna runs and never with gpt-5.4-mini. Luna says the sources do not describe it, then
    explains the neighbouring mandate.
- Running it needs `LLM_PROFILES_FILE=case/spensiones/llm_profiles.yaml` and `OPENAI_API_KEY`
  (a named profile cannot use the `LLM_` settings' key). In Docker: `APP_LLM_PROFILES_FILE` with
  the container path, and `OPENAI_API_KEY` in `.env`.

## Oficios (`oficio.yaml`, `oficio_dev.jsonl`)

489 Oficios Ordinarios (2001-2024), the Superintendencia's answers to consultations.
- Profile adapted from `circular.yaml`. All 489 ingest (3,324 chunks, median 5 per oficio): the
  4 scans with no text layer are OCR'd by `ocr_index_card` since the cleanup below (before it, 485
  and 3,312 with the `simple` parser).
- `oficio_dev.jsonl`: 20 blind questions from 20 oficios spread over the years. With 40 rerank
  candidates the search on the oficio topic scores MRR@10 .724.

Routing (MRR@10 with the router order and no tie-break) against the setup without oficios
(`C40`), on the 4 old dev sets plus `compendio_new_dev` (187 questions):

| oficio `evidence_bias` | oficio set | the 5 other sets, pooled | other-set questions that route to oficio |
|---|---|---|---|
| no oficio topic (C40) | n/a | .739 | 0 |
| 0 | .641 | .698 | 50 of 187 |
| -0.04 | .517 | .721 | 19 |
| -0.08 | .425 | .733 | 4 |

- Oficios cover every subject, so they win routing slots in other areas, and no bias keeps them
  reachable without that cost. With the router order alone the guard fails at every value; the
  tie-break below changes that.
- Keyword gating (route to oficios only when the question says "oficio", "criterio",
  "interpret..." and similar) was tried offline and fails: practitioners describe the situation
  without those words. It fires on 7 of the 20 oficio questions.

### With the merge tie-break (2026-10-22, gpt-6-luna)

The numbers above use the router order and never ran the LLM merge tie-break, which `main`
enables. The tie-break reads the top passages of the topics and picks the primary.

Live hits, measured with the criterion fixed before the run. All runs are from the same session.
B is the setup without oficios. The five sets are `afiliacion_dev` (both phrasings),
`cotizaciones_dev`, `compendio_new_dev` and `nota_dev` (109 questions).

| set | B | bias -0.04 | bias 0 |
|---|---|---|---|
| `operativa` | 19/20 | 19/20 | 19/20 |
| `v1_mixed` | 11/19 | 12/19 | 11/19 |
| cotizaciones | 18/20 | 18/20 | 18/20 |
| `compendio_new_dev` | 39/40 | 38/40 | 38/40 |
| `nota_dev` | 10/10 | 10/10 | 10/10 |
| the five pooled | 97/109 | 97/109 | 96/109 |
| `oficio_dev` | 0/20 | 14/20 | **18/20** |

- Both biases pass the rule (no set loses more than 1, the pool at most 2; the largest loss
  measured is 1). Bias 0 is kept because it scores highest overall.
- With bias 0, oficio passages are 79 of the 742 sources in the other five sets and reach 35 of
  109 answers. They are the first source in only 3: the tie-break keeps them secondary.
- Stage 2, scope bands: recalibrated with `--analyze` including the oficio topic (now
  `report.json`). `low_confidence_below` 0.6621 -> 0.6569, `out_of_scope_below` 0.4795 -> 0.4866,
  AUROC .966. 4/80 ood questions pass as normal, 10/40 in-domain questions fall in the `low`
  band, and none is refused.
- Stage 2, judge (floors fixed before the run, baseline: the gpt-6-luna run with the old bands
  and no oficios):

| measure | baseline | with oficios | floor |
|---|---|---|---|
| in-dev faithfulness | 37/37 | 38/39 | ≥ 35 |
| in-dev contradicted claims | 0 | 1 | ≤ 2 |
| in-dev refusal | 36/40 | 39/40 | ≥ 35 |
| in-dev relevance | 36/40 | 39/40 | ≥ 35 |
| in-dev citation | 35/38 | 38/40 | ≥ 32 |
| ood refusal | 76/80 | 76/80 | ≥ 75 |
| `oficio_dev` faithfulness / citation / relevance / refusal | - | 19/20, 16/20, 20/20, 20/20 | report only |

- Every floor passes. 19 of the 40 in-dev answers carry an oficio source; it is the first source
  in 2.
- Known gap: the near-domain `out` label verification below predates the oficios. Some oficios
  now cover two of those items: out-pens-02 (PGU) in 26 oficios and out-pens-03 (emergency
  withdrawal) in 43. Their `out` labels likely no longer hold. They inflate the p95 band
  (`low_confidence_below`) and count as refusal misses in the ood judge. Re-verifying the 36 items
  against the oficio chunks, then recalibrating, is pending.
- Personal data: the oficios are public documents of the Superintendencia, indexed as published.
  The heading chunk (`preamble: keep`) keeps the DE/A block, and 14 oficios name a person with a
  RUT. No redaction is applied.

## Rewrite loop: `min_rerank_score` and `max_rewrites` (2026-10-22)

After the merge, the agent rewrites the query when no source reaches `min_rerank_score`. Until now
the threshold was an uncalibrated 0.1 and the loop was off (`max_rewrites: 0`).

Stage 1 ran offline, with no LLM and a rule fixed before computing:
- 129 dev questions with expected snippets: `afiliacion_dev`, `cotizaciones_dev`,
  `compendio_new_dev`, `nota_dev` and `oficio_dev`.
- Each question is routed by the evidence router and searched hybrid with TEI rerank at 40
  candidates.
- S is the best rerank score over the routed hits, and H means the snippet is retrieved.
- The threshold is the 5th percentile of S over the H questions, rounded down: **0.92**.

| group | n | median S | below 0.92 |
|---|---|---|---|
| answer retrieved (H) | 116 | .997 | 4 |
| answer missed | 13 | .840 | 9 (69%) |
| ood | 76 | .005 | 74 (most never reach retrieval: scope check) |

The old 0.1 would have caught 2 of the 13 misses.

Stage 2 set `max_rewrites: 1`. Its live guard had a criterion fixed before the run; the baseline
is the oficios cleanup run.

| measure | baseline | rewrite | rule |
|---|---|---|---|
| live hits, 129 questions pooled | 116 | **119** | ≥ 117 (decides) |
| `v1_mixed` / `operativa` | 12/19, 20/20 | 14/19, 20/20 | each ≥ −1 |
| cotizaciones / `compendio_new_dev` / `nota_dev` / `oficio_dev` | 18, 38, 10, 18 | 18, 39, 10, 18 | each ≥ −1 |
| in-dev faithfulness / refusal / relevance / citation | 37/38, 37, 37, 33/39 | 37/38, 38, 38, 36/39 | ≥ 35, 36, 36, 31 |
| ood refusal misses | 1/76 | 0/76 | ≤ 3 |
| `oficio_dev` faithfulness / citation | 20/20, 18/20 | 20/20, 20/20 | report |

- It passes, with 0 dump or judge errors.
- Rescued: af-01 (lost earlier to the Libro I crowding), af-16 and nd-26. None was lost.
- Cost: a rewritten turn adds 1 model call and 1 retrieval. The dumps do not record how often it
  fired.

## Colloquial vocabulary (`colloquial_dev.jsonl`, 2026-10-23)

Users say "producto obligatorio"; the norms say "cuenta de capitalización individual de
cotizaciones obligatorias" (0 of 15,622 chunks contain the first). In a live chat the first turn
was saved by the rewrite, but the follow-ups ("¿y el paso a paso?", "dame el de chile") kept the
colloquial term and retrieved the Chile-Peru transfers.

The change is prompt-only:
- `analyze`: when the last message depends on the conversation, reuse the normative terms a
  previous answer gave for the same topic, next to the user's words.
- `rewrite`: put the normative term next to a colloquial one, with a 9-entry glossary. An entry
  needs the colloquial term in ~0 chunks and the normative one in many (for example "estado de
  cuenta" 8, "cartola" 469). APV, jubilación, sueldo, cambio de fondo, herencia, PGU and "retiro
  del 10" are literal in the corpus, so they are not in it.

`dump_answers` rows take an optional `history` (earlier turns), and records carry the
`standalone_query`. `colloquial_dev.jsonl` (20 rows, written before the glossary):
- 16 colloquial paraphrases of dev questions, keeping their snippets (`colloquial`);
- 4 follow-ups (`colloquial_followup`): turns 2 and 3 of the live chat, with its real history,
  and 2 built ones whose previous answer is taken from the source text. The live-chat rows hit
  when a source is in Libro I, Título III, Letra A, Capítulos XVIII-XXVIII (`expected_locators`).

The criterion was fixed before the runs. B is `main`, N the new prompts; B1 and N1 run every
set, B2 and N2 only `colloquial_dev`. The floors come from the documented run-to-run ranges.

| measure | B1, B2 | N1, N2 | rule |
|---|---|---|---|
| follow-ups | 2/4, 2/4 | **3/4, 4/4** | each N ≥ 3 (decides) |
| colloquial single-turn | 8/16, 9/16 | 10/16, 10/16 | min N ≥ 8 |
| live hits, 129 questions pooled | 120 | 119 | ≥ 118 |
| `v1_mixed` / `operativa` | 15/19, 20/20 | 14/19, 20/20 | each ≥ −1 |
| cotizaciones / `compendio_new_dev` / `nota_dev` / `oficio_dev` | 18, 39, 10, 18 | 18, 39, 10, 18 | each ≥ −1 |
| in-dev faithfulness / refusal / relevance / citation | 35/38, 39, 39, 35/40 | 36/37, 37, 37, 33/38 | ≥ 33, 37, 37, 28 |
| ood refusal misses | 0/76 | 0/76 | ≤ 3 |

- It passes, with 0 dump or judge errors and 0 reranker fallbacks.
- The CI 5-gram check then caught one glossary entry: "el saldo de la cuenta personal" shares a
  5-gram with in-cmp-10 and nd-33. It became "los fondos, el saldo", and `colloquial_dev` was
  rerun twice against the same floors. N3 got follow-ups 4/4 and single-turn 11/16; N4 got 4/4
  and 8/16. The guard was not rerun: "plata" is in 1 guard question, and the entry only acts
  inside `rewrite`.
- The follow-up gain is stable: 4/4 in three of the four new runs. The single-turn gain is not:
  N ranges from 8 to 11 and B from 8 to 9.
- On single-turn questions the `analyze` rule does nothing: 39 of 40 in-dev standalone queries
  are identical, and the other one is a paraphrase.
- The relevance and refusal losses (in-cmp-08, in-res-30) got identical sources in B1 and N1,
  so they come from synthesis variance.
- af-16 still answers right; only the snippet chunk changed.
- af-01 is likely a cost of the `rewrite` edit. Its standalone query is identical but its
  sources differ, and it now answers "No lo sé". The dumps do not record the rewritten query.
- Earlier runs were slightly degraded. Until this change, `rerank-gpu` ran with
  `--max-concurrent-requests 128`, and TEI takes one permit per text, so 40-candidate batches at
  `--concurrency 4` got HTTP 429. The fallback keeps the vector order silently. The `rewrite` run
  above had 6 fallen-back retrievals, and the oficio runs had 3 to 7. The container now runs with
  2048, and the live guard rejects a dump whose log shows a fallback. The clean B1 has 120 live
  hits, against 119 in the `rewrite` run.

## Colloquial question bank (`bank_dev.jsonl`) and the register rewrite (2026-10-24, not merged)

An offline probe (evidence router, hybrid search, rerank; no LLM) over a hand-written bank of 140
colloquial questions (89 in a member's voice, 51 in operations staff's voice):
- 108 (77%) start with a weak retrieval (best rerank below 0.92). Only 1 is refused by scope.
- Rewritten by hand in the norms' register (third person, "el afiliado", "la Administradora",
  their terms), 88 of the 108 reach 0.92. The other 20 are coverage gaps (e.g. seasonal workers,
  a wrong RUT, an employer's bankruptcy).
- Word swaps alone (AFP -> Administradora, imposiciones -> cotizaciones, plata -> fondos, ...)
  lift 15 of 78 weak questions; the full normative phrasing lifts 68. The gap is register, not
  rare words, so a bigger glossary would fix little.

`bank_dev.jsonl` keeps the 119 questions with a findable answer: 87 `bank_gap`, 32 `bank_strong`.
Each target is the top reranked chunk of the normative (gap) or own (strong) phrasing, as a
70-character snippet found in at most 3 chunks (chunks overlap their neighbours). These are
silver targets: a spot check of 14 found 11 clearly right and 2-3 doubtful.

Two prompt changes were measured against it, each with a criterion fixed before the runs
(B = `main`, 2 runs each):

| measure | B | register rewrite | + `max_rewrites: 2` | rule |
|---|---|---|---|---|
| `bank_gap` | 47, 44 | 55, 47 | 48, 52 | min N - max B >= 3 (decides) |
| `bank_strong` | 32, 32 | 32, 32 | 32, 32 | >= 31 |
| live hits, 129 pooled | 119 | 122 | 122 | >= 117 |
| colloquial single-turn / follow-ups | 9/16, 3/4 | 12/16, 3/4 | 14/16, 4/4 | >= 8, >= 3 |

- Both fail the deciding rule; every guard floor passes.
- Per question (exploratory, after the fact): over the 4 register runs against the 2 baseline
  runs, 24 rows do better and 21 worse (sign test p = 0.77). The mean gain comes from rows that
  flip both ways between runs, not from a consistent effect.
- The agent's run-to-run spread on this set is 4-8 hits, even on `main`. The LLM profile sets
  no temperature (undocumented for gpt-6-luna); reducing that noise comes before measuring
  smaller effects.

## `temperature: 0` on the agent's model (2026-10-24, not merged)

A single live call showed gpt-6-luna accepts `temperature: 0` with `reasoning_effort: none` (not
documented on its model page). Criterion fixed before the runs: the rows of `bank_dev` whose live
hit differs between two identical runs must drop from the baseline's 15 to at most 7.

| measure | B1, B2 (no temperature) | t0_1, t0_2 | rule |
|---|---|---|---|
| rows that flip between the two runs | 15 | **11** | <= 7 (decides) |
| identical source lists / identical answers | 29, 0 of 119 | 46, 5 of 119 | report |
| `bank_gap` | 47, 44 | 45, 46 | mean >= 43.5 |
| live hits, 129 pooled | 119 | 120 | >= 117 |
| in-dev faithfulness / citation / refusal / relevance | 37/38, 37/40, 39, 39 | 35/38, 30/38, 38, 38 | >= 35, 30, 37, 37 |

- It fails the noise rule and is not merged; every other floor passes (two at the floor).
- Temperature 0 lowers the variance only partly: the API is not deterministic at 0 and four LLM
  steps (analyze, rewrite, merge tie-break, synthesis) compound it. With this noise, future
  experiments need more runs per arm or a paired per-row rule fixed in advance.

## Help articles at full coverage (2026-10-24, undecided)

The `articulo_faq`, `circular`, `ncg` and `resolucion` profiles were ingested from the 35 sample
files only (8 help articles, 24 circulares, 14 NCG, 4 resoluciones in tenant `spensiones`), while
spensiones.cl lists 2,586 circulares and 483 NCG and the index has 210 help-article pages. Help
articles were scaled first: of the 210 pages, 69 are redirects to "Especiales" landing pages and
1 is empty, leaving 140; 126 route to `articulo_faq` and 6 to `resolucion`.

The criterion was fixed before the runs. It decides on the LLM judge's answer relevance on
`bank_dev` (its silver targets predate the articles, so live hits would count a correct
article-based answer as a miss): min of the new runs above the max of the baseline runs.

| measure (faq_1 against B1) | B1 | faq_1 | floor |
|---|---|---|---|
| `bank_dev` answers with a help-article source / citing one | 5 / 3 | 67 / 45 | report |
| live hits, 129 pooled | 119 | 119 | >= 117 |
| cotizaciones | 18 | 17 | >= 17 |
| ood refusal misses | 0 | 2 | <= 3 |
| colloquial single-turn / follow-ups | 9/16, 3/4 | 12/16, 3/4 | >= 8, >= 3 |

- Undecided: the DeepSeek judge account ran out of balance (HTTP 402) while judging the
  baseline, and the OpenAI credits ran out during faq_2 (57 of 119 answered). The deciding judge
  and faq_2 are pending; the articles were removed from the database until they pass.
- Some articles describe the former Pilar Solidario (PBSI, APSI), replaced by the PGU in 2022: a
  risk of outdated answers to check in the judged answers.

## Label verification (near-domain `out` items)

Method (content only, scores were NOT used or looked at to decide): for each of the 36
near-domain `out` items (`pens`, `labor`, `isapre`, `tax`, `bank`, `abroad`), (1) the top-3
retrieved chunks of every profile were read, and (2) the text of every READY chunk of the tenant
was searched (accent-insensitive regex) with the terms of the question. A verdict is `out` when no
chunk answers the question, even if it mentions the topic in passing.

Corpus at verification time (READY chunks): compendio 1413, ncg 17, circular 13, resolucion 1,
articulo_faq 1 (seguro de cesantia), jurisprudencia_indice 57, jurisprudencia_nota 0.

Result: 36/36 verified `out`, 0 relabelled. `indice` chunks are lists of oficio titles and
`resolucion` is a lobby-law resolution: none answers any of them.

| id | verdict | evidence |
|---|---|---|
| out-isapre-01 (change of Isapre) | out | only mentions ISAPRE/FONASA as the destination of the health contribution (compendio chunks 94d0e41b, 96b7dbd0); no switching procedure |
| out-isapre-02 (GES) | out | no mention of GES / explicit health guarantees |
| out-isapre-03 (plan price at 60) | out | no health-plan pricing text |
| out-isapre-04 (Superintendencia de Salud complaint) | out | "Superintendencia de Salud" absent; the complaint chapters of the compendio (Titulo VII Gestion de Reclamos, Titulo VIII Cap. V Reclamacion) concern AFP and pension-account claims |
| out-isapre-05 (Ley de Urgencia) | out | absent |
| out-isapre-06 (FONASA tramos) | out | FONASA appears only as a collection account (63fa0f66, d8aa3fc5); "tramo B/C" absent |
| out-tax-01 (Operacion Renta) | out | only "proceso anual de declaracion de impuesto a la renta" as the channel where independents pay contributions (compendio 00c82efb); no filing instructions |
| out-tax-02 (company deductions) | out | no text on deductible expenses |
| out-tax-03 (boleta de honorarios) | out | absent; SII appears only as a data source (058c02a9, 12ae3844) |
| out-tax-04 (IVA deadline) | out | IVA absent |
| out-tax-05 (inicio de actividades SII) | out | absent |
| out-tax-06 (impuesto territorial) | out | absent |
| out-labor-01 (aviso previo al renunciar) | out | no notice-period text |
| out-labor-02 (severance by years of service) | out | closest: compendio d87cafcf / 4505d90a, the "pacto de indemnizacion sustitutiva" (art. 164 CT replaced after year 6 by an employer contribution of at least 4.11%); it never explains how the legal severance is computed in a dismissal |
| out-labor-03 (vacation days) | out | only "bonos por vacaciones" as a taxable item (db866498) |
| out-labor-04 (weekly working hours) | out | only the employer's duty to report hours (798c7043); no legal maximum |
| out-labor-05 (overtime pay) | out | absent |
| out-labor-06 (fuero maternal) | out | absent |
| out-abroad-01 (Spain) | out | no foreign system text |
| out-abroad-02 (ANSES, Argentina) | out | absent |
| out-abroad-03 (US Social Security) | out | absent |
| out-abroad-04 (Mexico IMSS) | out | absent |
| out-abroad-05 (Germany) | out | absent |
| out-abroad-06 (AFORE, Mexico) | out | absent |
| out-bank-01 (mortgage) | out | "hipotecari", "credito", "prestamo" absent |
| out-bank-02 (credit vs debit card) | out | "tarjeta" absent |
| out-bank-03 (loan installment) | out | absent |
| out-bank-04 (CAE) | out | absent |
| out-bank-05 (open a checking account) | out | "cuenta corriente" appears only as AFP collection accounts (09633329) |
| out-bank-06 (DICOM) | out | absent |
| out-pens-01 (retirement age, men) | out | no chunk states the legal retirement age; "edad legal" is used without the number (3ded2f25, 69c7cc6b); 65 anos appears only for contribution-exemption rules (e59e9f9d, a3844eb0) and men 60 / women 55 for fund assignment (216ff5e0, a0ea91d5) |
| out-pens-02 (Pension Garantizada Universal) | out | "pension garantizada" / PGU absent |
| out-pens-03 (percentage withdrawal in an emergency) | out | closest: COVID withdrawals of the severance-savings account under Ley 21.227 / 21.247 (compendio 7445fa79, 4a8e41ae), not a withdrawal of pension funds |
| out-pens-04 (alimony amount) | out | alimony appears only as the procedure to collect debts from the member's funds (3ded2f25); no amount rule |
| out-pens-05 (AFP returns per analysts) | out | no ranking or analyst text; "rentabilidad" appears in definitions and in circular 2425's rate method |
| out-pens-06 (how to apply to the bono por hijo) | out | only the definition of the "bonificacion fiscal por hijo nacido vivo" (compendio 5ee9167b) and accrediting it in the account; no application procedure |

The other 44 `out` items (`gen`, `chat`, `code`, `other`) are unrelated to pensions, health,
labour, tax or banking and were not checked chunk by chunk.

### Re-verification against the grown corpus (2026-10-22)

The table above predates the 10 compendio Libros, the Notas Internas and the oficios. The 36 items
were checked again, before recalibrating, against the whole tenant: 11k compendio chunks, 636
notas, 3,324 oficio chunks after the OCR reingest, and the other profiles. Same method: read the
top-3 chunks of every profile, search every chunk's text with `unaccent`, never look at scores.
The rule, fixed beforehand: an item that some chunk answers is REMOVED from `ood.jsonl` (not
rewritten), which leaves 76 items.

| id | now answered by |
|---|---|
| out-pens-01 (retirement age, men) | compendio Libro III Título I Letra B Cap. II: "Tienen derecho a pensión de vejez los afiliados que hayan cumplido los 65 años si son hombres y 60 años si son mujeres" |
| out-pens-02 (PGU requirements) | compendio Libro III Título XVII Letra B Cap. III: 65 years, not in the richest 10%, 20 years of residence, base pension under the upper bound |
| out-pens-03 (emergency withdrawal) | oficios of 2020-2021: "retiro por única vez de hasta el 10% ... máximo 150 UF", the third 10% withdrawal under Ley 21.330 |
| out-pens-06 (applying for the bono por hijo) | compendio Libro III Título VII Cap. IV: "deberá ser solicitada ante el IPS ... a través de la suscripción del formulario" |

The other 32 stay `out`. Borderline cases kept `out`:
- out-labor-04: "45 horas semanales" appears only as a reference to scale Seguro de Cesantía amounts.
- out-isapre-06: one nota gives the coverage of FONASA groups C and D, but does not compare B and C.
- out-abroad-*: the oficios explain the bilateral agreements, not the foreign systems.

For out-pens-02 and out-pens-03 the chunk the verifier picked is in no profile's top-3 for the raw
question. It is not a gap of the agent, though. Offline, the PGU-requirements chunk ranks 11 of 40
after rerank (other Título XVII chunks score 0.99), and the 10% oficio ranks 5 (rerank 0.20).
Live, with 8 merged sources from 2 topics, the agent answered both: the four PGU requirements, and
the 10% withdrawal with its 150 UF / 35 UF bounds, noting it was exceptional. The same facts sit
in other oficios and chapters that do reach the sources. The rewrite loop does not fire here,
because other sources score above 0.92. Details: `~/scratch/agintis/labels/ood_reverify.md` (local).

Recalibrated on the 76 items (`--analyze`, gpt-6-luna, now `report.json`): `low_confidence_below`
0.6569 -> 0.6137, `out_of_scope_below` 0.4866 -> 0.4779, AUROC .966 -> .988. 4/76 ood questions
pass as normal, 3/40 in-domain questions fall in the `low` band, and none is refused. The covered
items were what inflated the p95 band.

### Oficios cleanup (2026-10-22)

- `oficio.yaml` now uses `ocr_index_card`, so all 489 oficios index, the 4 scans included.
- The internal-id line regexes cover every id seen: `OF-DDN-20-3753`, 5-digit numbers and
  `OF-DDNI-…`.
- The unmatched `titulo` and `capitulo` levels and the unused `materia` field are gone.
- An offline check over all 489 oficios, against the previous profile, found:
  - 0 line differences other than internal-id lines in the 485 text oficios;
  - 0 id-only lines left;
  - an OCR non-word share of 0.144, against 0.147 for text oficios.
- Only the oficio profile was reingested: 4 indexed, 485 replaced, 0 failed.

Guard, with floors fixed before the run. Baseline: the stage 2 judge run and the bias-0 live hits.

| measure | baseline | cleanup | floor |
|---|---|---|---|
| in-dev faithfulness | 38/39 | 37/38 | ≥ 36 |
| in-dev refusal / relevance | 39/40, 39/40 | 37/40, 37/40 | ≥ 37 |
| in-dev citation | 38/40 | **33/39** | ≥ 35 |
| ood refusal misses | 4/80 | 1/76 | ≤ 4 |
| judge errors | 0 | **1** | 0 |
| the five other sets' live hits, pooled | 96/109 | 98/109 | ≥ 94 |
| `oficio_dev` live hit / faithfulness | 18/20, 19/20 | 18/20, 20/20 | ≥ 17, ≥ 18 |

Accepted as a documented exception, because both failures are judge noise:
- Citation:
  - The 6 citation failures are disjoint from the baseline's 2.
  - None of them changed scope band.
  - in-cmp-05 and in-cir-24 cite the same sources in both runs, and the verdict flipped.
  - In-dev citation over the comparable runs was 31, 35, 36, 35, 38 and 33. The floor (the best
    run minus 3) was too tight. Future floors for this measure should come from that run-to-run
    range.
- The judge error: the faithfulness judge's output on out-abroad-03, a refusal, did not parse. No
  floored measure uses it.

## Edits to `in_dev.jsonl`

These are `in` items (they only report, they never set a threshold):

- `in-ncg-20`: rewritten as a standalone question (the original said "de este tipo", which needed
  the previous turn). Answer: NCG 372, "Procedimiento de Notificacion", within the third day of the
  sentence being final (chunk b681c5a4).
- `in-cir-26`: replaced by a different question. Answer: circular 100, the payment is credited to
  the individual account at the share value of the previous day's close (chunk e2645ced).
