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
- Deployment: set `RERANKER_MAX_CANDIDATES=20` (the app default is 8). The TEI server must
  accept 20 texts per request: `--max-client-batch-size` of at least 20, and
  `--max-concurrent-requests` of at least 40, since TEI counts each text and two topics rerank
  in parallel. Otherwise every rerank fails and `FallbackReranker` silently keeps the vector
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
- Activated after checking the 101 notes with text: 556 chunks, 0 failures. 16 are scans and
  fail until OCR is enabled. That is 86% with text, below the pre-registered 95%; accepting it is
  your call.
- `nota_dev.jsonl`: 10 blind questions. MRR@10 .850 on the topic and .850 routed. The router picks
  `jurisprudencia_nota` for 10 of 10, as primary for 9.
- Routed MRR@10 on the other 5 sets: pooled .737 against .739 without notes; routing_dev2 drops
  .007. The topic is selected for 13 of 187 other questions, as primary for 2.
- Unlike oficios, the notes are distinct enough for the evidence router.

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

## Edits to `in_dev.jsonl`

These are `in` items (they only report, they never set a threshold):

- `in-ncg-20`: rewritten as a standalone question (the original said "de este tipo", which needed
  the previous turn). Answer: NCG 372, "Procedimiento de Notificacion", within the third day of the
  sentence being final (chunk b681c5a4).
- `in-cir-26`: replaced by a different question. Answer: circular 100, the payment is credited to
  the individual account at the share value of the previous day's close (chunk e2645ced).
