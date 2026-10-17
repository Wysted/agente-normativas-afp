# Validation report: spensiones RagProfiles vs samples

Token counter: bge-m3 (HF tokenizers, add_special_tokens=False). Budgets subtract 8 special-token reserve.
Ingest date used for future-date rejection: 2026-10-15.

## compendio-sp-libro-i -> `spensiones_compendio` (compendio.yaml)

- cleaner drop_toc_pages: dropped 5 leading TOC pages
- cleaner strip_running_headers: removed 448 lines
- cleaner dedupe_heading_echo: removed 220 echo lines
- cleaner normalize_whitespace: ok
- Chars: raw 1,277,401 -> cleaned 1,216,525 (4.8% removed)

| level | matches | masked (inside quotes) | examples |
|---|---:|---:|---|
| seccion | 125 | 0 | Libro I, Título I, Letra A Afiliación e Incorporación a una Administradora / Libro I, Título I, Letra A Afiliación e Incorporación a una Administradora / Libro I, Título I, Letra A Afiliación e Incorporación a una Administradora |
| capitulo | 83 | 0 | Capítulo I. Introducción / Capítulo II. Normas generales de Afiliación / Capítulo III. Tipos de Afiliación e Incorporación a una AFP |
| anexo | 36 | 0 | Anexo Formulario Solicitud de Incorporación / Anexo Nº 1 Solicitud de Desafiliación / Anexo Nº 2 Declaración |
| numero | 1493 | 0 | 1. Todos aquellos trabajadores dependientes, trabajadores independientes y afiliados voluntarios, / 2. Una vez cumplido el período de permanencia a que se refiere el número precedente, los / 3. Para los trabajadores dependientes, el inicio de labores por primera vez generará la afiliación |
| subnumero | 57 | 0 | 1.1. El trabajador dependiente que se encuentre cotizando o haya cotizado en alguno de los / 1.2. El trabajador pensionado por alguno de los regímenes administrados por el IPS o en otras / 1.3. El trabajador que se encuentra en alguna de las situaciones mencionadas en los números |
| literal | 868 | 0 | a) Cuando la entidad adjudicataria no cumpla con los requisitos para constituirse como / b) Cuando no se realice o no se adjudique la licitación por alguna de las causales establecidas / a) Original para la Administradora adjudicataria; |

Checks against ground truth:
- capitulo: TOC lists 83 chapter entries; body matches 83 after echo dedupe. Sequence check (TOC order vs body order): IDENTICAL
- anexo: TOC lists 36 annex entries; body matches 36
- numero: 1493 in-sequence, 0 out-of-sequence, 5 numbers skipped inside runs (suspected FN). Out-of-sequence cases (review manually): []
- anexo (multiset vs TOC): TP=35, precision=0.972, recall=0.972
- anexo detail: TOC entries missing in body matches: ['anexo información de r']; body matches not in TOC: ['anexo']
- running header lines left after cleaning: 0; dotted-leader TOC lines left: 0

**Chunks: 1413** | tokens min 31 / p10 101 / p50 170 / p90 448 / max 690 | over budget (700): 0 | < 30 tokens: 0 | split parts: 116
Chunks by unit level: {'capitulo': 12, 'numero': 1371, 'anexo': 30}
Metadata coverage (chunks with value / total): source_id 1413/1413, doc_family 1413/1413, system 1413/1413, libro 1413/1413, titulo 1413/1413, letra 1087/1413, capitulo 1198/1413, capitulo_nombre 1198/1413, anexo 207/1413, numeros 1371/1413, compendio_path 1413/1413, amended_by_ncg 410/1413, cited_laws 219/1413, validity 1413/1413, page_end 1413/1413

Example compendio chunks (heading prefix | compendio_path | amended_by_ncg):

- `Compendio SP Libro I > Título I, Letra A Afiliación e Incorporación a una Administradora > Capítulo II. Normas generales` | SP/LI/TI/A/CII | [31] | 163 tok
- `Compendio SP Libro I > Título I, Letra A Afiliación e Incorporación a una Administradora > Capítulo II. Normas generales` | SP/LI/TI/A/CII | [31] | 205 tok
- `Compendio SP Libro I > Título I, Letra A Afiliación e Incorporación a una Administradora > Capítulo II. Normas generales` | SP/LI/TI/A/CII | [31] | 145 tok
- `Compendio SP Libro I > Título I, Letra A Afiliación e Incorporación a una Administradora > Capítulo II. Normas generales` | SP/LI/TI/A/CII | [31] | 169 tok

## circular-2425 -> `spensiones_circular` (circular.yaml)

- cleaner strip_repeated_page_lines: furniture=['superintendencia de pensiones ci-ddn-#-#'] removed 7 lines
- cleaner strip_lines: removed 11 lines
- cleaner normalize_whitespace: ok
- Chars: raw 7,636 -> cleaned 7,084 (7.2% removed)

| level | matches | masked (inside quotes) | examples |
|---|---:|---:|---|
| titulo | 0 | 0 |  |
| capitulo | 0 | 0 |  |
| seccion | 2 | 0 | VIGENCIA / ANEXO |
| numero | 4 | 0 | 1. En virtud de lo establecido en el inciso cuarto del artículo 64 del D.L. N° 3.500 de 1980 / 2. El vector de tasas de interés es el resultado de sumar a una estructura temporal de / 3. La tasa de retorno anual promedio de los Fondos de Pensiones C, D y E, se obtiene a |
| literal | 0 | 0 |  |

Checks against ground truth:
- expected numero 1,2,3,4 + seccion VIGENCIA, ANEXO -> got numero ['1', '2', '3', '4'], seccion ['VIGENCIA', 'ANEXO']
- footer lines left: 1 (one legit in IDENTIFICACIÓN INTERNA)

**Chunks: 9** | tokens min 74 / p10 143 / p50 215 / p90 487 / max 489 | over budget (600): 0 | < 30 tokens: 0 | split parts: 2
Chunks by unit level: {'preamble': 1, 'numero': 4, 'seccion': 4}
Metadata coverage (chunks with value / total): source_id 9/9, doc_family 9/9, circular_number 9/9, series 9/9, area 9/9, issued_date 9/9, date_precision 9/9, listing_tab 9/9, effective_from 9/9, modifies_circulars 0/9, numeros 4/9, seccion 4/9, cited_laws 3/9, page_end 9/9

## circular-cafp-100 -> `spensiones_circular` (circular.yaml)

- cleaner strip_repeated_page_lines: furniture=['de fondos de pensiones', 'superintendencia de administradoras'] removed 4 lines
- cleaner strip_lines: removed 0 lines
- cleaner normalize_whitespace: ok
- Chars: raw 2,361 -> cleaned 1,946 (17.6% removed)

| level | matches | masked (inside quotes) | examples |
|---|---:|---:|---|
| titulo | 0 | 0 |  |
| capitulo | 0 | 0 |  |
| seccion | 0 | 0 |  |
| numero | 3 | 0 | 1. Las Administradoras deberán remitir a esta Superintendencia los días / 2.- Las Administradoras depositarán el documento de pago del Bono de / 3.- En la letra b) del numeral 1 de la Ci rcular N° 7, de 1° de abril de 1981, y a |
| literal | 7 | 0 | a) Nombre del afiliado. / b) R.U.T. del afiliado. / c) Número del Bono de Reconocimiento. |

Checks against ground truth:
- expected numero 1,2,3 + literal a..g -> got numero ['1', '2', '3'], literal ['a', 'b', 'c', 'd', 'e', 'f', 'g']
- letterhead lines left: 0

**Chunks: 4** | tokens min 113 / p10 113 / p50 150 / p90 235 / max 235 | over budget (600): 0 | < 30 tokens: 0 | split parts: 0
Chunks by unit level: {'preamble': 1, 'numero': 3}
Metadata coverage (chunks with value / total): source_id 4/4, doc_family 4/4, circular_number 4/4, series 4/4, area 4/4, issued_date 4/4, date_precision 4/4, listing_tab 4/4, effective_from 0/4, modifies_circulars 4/4, numeros 3/4, seccion 0/4, cited_laws 0/4, page_end 4/4

## ncg-sp-372 -> `spensiones_ncg` (ncg.yaml)

- cleaner strip_repeated_page_lines: furniture=[] removed 0 lines
- cleaner strip_lines: removed 3 lines
- cleaner normalize_quotes: ok
- cleaner normalize_whitespace: ok
- Chars: raw 28,901 -> cleaned 27,672 (4.3% removed)

| level | matches | masked (inside quotes) | examples |
|---|---:|---:|---|
| parte | 2 | 0 | I. Introdúcense las siguientes modificaciones al Título I sobre PENSIONES, del Libro III: / II. VIGENCIA |
| enmienda | 6 | 0 | A. Reemplácese, en el número 6.1 del Capítulo II de la Letra D.1 Beneficios por / B. Modifíquese el Capítulo I, de la Letra E. Pensión de Sobrevivencia , de la siguiente / C. Modifíquese la Letra F. Modalidades de Pensión, de la siguiente forma: |
| operacion | 9 | 22 | a) Elimínase en el primer párrafo del número 1, a continuación de la frase “y los / b) Agrégase en el primer párrafo del número 1, a continuación de la expresión “se / c) Agrégase en el número 2, a continuación de la letra e), el siguiente párrafo |

Checks against ground truth:
- expected parte [I, II] -> ['I', 'II']
- expected enmienda [A, B, C, C, D, E] (source repeats 'C.') -> ['A', 'B', 'C', 'C', 'D', 'E']
- expected 9 operaciones outside quotes (B:a-c, C:a-c, C:a, E:a-b) -> 9 ['a', 'b', 'c', 'a', 'b', 'c', 'a', 'a', 'b']; masked inside quoted Compendio text: 22
- quote balance after normalize_quotes: open 18 / close 18

**Chunks: 17** | tokens min 117 / p10 213 / p50 437 / p90 567 / max 676 | over budget (700): 0 | < 30 tokens: 0 | split parts: 9
Chunks by unit level: {'preamble': 1, 'enmienda': 15, 'parte': 1}
Metadata coverage (chunks with value / total): source_id 17/17, doc_family 17/17, ncg_number 17/17, system 17/17, issued_date 17/17, effective_from 17/17, doc_role 17/17, op_verb 15/17, enmienda 15/17, target_libro 17/17, target_titulo 17/17, target_letra 14/17, target_capitulo 12/17, target_numero 1/17, target_anexo 1/17, compendio_path 17/17, cited_laws 14/17, page_end 17/17

NCG chunk -> Compendio target (one row per chunk):

| # | heading | op_verb | compendio_path | tokens |
|---:|---|---|---|---:|
| 0 | NCG N° 372 (2026-08-18) > Encabezado |  | SP/LIII/TI | 345 |
| 1 | NCG N° 372 (2026-08-18) > I. Introdúcense las siguientes modificaciones al Tít… > A. Reemp | replace | SP/LIII/TI/D.1/CII | 117 |
| 2 | NCG N° 372 (2026-08-18) > I. Introdúcense las siguientes modificaciones al Tít… > B. Modif | modify | SP/LIII/TI/E/CI | 284 |
| 3 | NCG N° 372 (2026-08-18) > I. Introdúcense las siguientes modificaciones al Tít… > C. Modif | modify | SP/LIII/TI/F | 434 |
| 4 | NCG N° 372 (2026-08-18) > I. Introdúcense las siguientes modificaciones al Tít… > C. Modif | modify | SP/LIII/TI/G | 213 |
| 5 | NCG N° 372 (2026-08-18) > I. Introdúcense las siguientes modificaciones al Tít… > D. Agrég | add | SP/LIII/TI/H/CVIII | 539 |
| 6 | NCG N° 372 (2026-08-18) > I. Introdúcense las siguientes modificaciones al Tít… > D. Agrég | add | SP/LIII/TI/H/CVIII | 663 |
| 7 | NCG N° 372 (2026-08-18) > I. Introdúcense las siguientes modificaciones al Tít… > D. Agrég | add | SP/LIII/TI/H/CVIII | 567 |
| 8 | NCG N° 372 (2026-08-18) > I. Introdúcense las siguientes modificaciones al Tít… > D. Agrég | add | SP/LIII/TI/H/CVIII | 508 |
| 9 | NCG N° 372 (2026-08-18) > I. Introdúcense las siguientes modificaciones al Tít… > D. Agrég | add | SP/LIII/TI/H/CVIII | 513 |
| 10 | NCG N° 372 (2026-08-18) > I. Introdúcense las siguientes modificaciones al Tít… > D. Agrég | add | SP/LIII/TI/H/CVIII | 354 |
| 11 | NCG N° 372 (2026-08-18) > I. Introdúcense las siguientes modificaciones al Tít… > D. Agrég | add | SP/LIII/TI/H/CVIII | 676 |
| 12 | NCG N° 372 (2026-08-18) > I. Introdúcense las siguientes modificaciones al Tít… > D. Agrég | add | SP/LIII/TI/H/CVIII | 535 |
| 13 | NCG N° 372 (2026-08-18) > I. Introdúcense las siguientes modificaciones al Tít… > D. Agrég | add | SP/LIII/TI/H/CVIII | 437 |
| 14 | NCG N° 372 (2026-08-18) > I. Introdúcense las siguientes modificaciones al Tít… > D. Agrég | add | SP/LIII/TI/H/CVIII | 534 |
| 15 | NCG N° 372 (2026-08-18) > I. Introdúcense las siguientes modificaciones al Tít… > E. Modif | modify | SP/LIII/TI/AX6 | 370 |
| 16 | NCG N° 372 (2026-08-18) > II. VIGENCIA |  | SP/LIII/TI | 133 |

## cms-aid-14843 -> `spensiones_jurisprudencia_indice` (guia_jurisprudencia.yaml)

- cleaner drop_pages_before: dropped 2 pages
- cleaner strip_lines: removed 10 lines
- cleaner join_continuation_lines: joined 62 lines
- cleaner normalize_whitespace: ok
- Chars: raw 30,747 -> cleaned 17,283 (43.8% removed)

| level | matches | masked (inside quotes) | examples |
|---|---:|---:|---|
| materia | 4 | 0 | I. ANTIGUO SISTEMA PREVISIONAL / II. SISTEMA DE CAPITALIZACIÓN INDIVIDUAL OBLIGATORIA CREADO POR EL DECRETO LEY N° 3.500 / III. SEGURO DE CESANTÍA |
| rama | 2 | 0 | A. SEGURIDAD SOCIAL / B. DERECHO CORPORATIVO |
| tema | 53 | 0 | 1. Accidentes del trabajo y enfermedades profesionales / 2. Acuerdo unión civil Of. Ord. N° 27902 de 06-12-17 / 3. AFP de retorno |
| record | 219 | 0 | 1.1. Pensión sustitutiva artículo 53 Ley N° 16.744 Of. Ord. N° 11391 de 30-05-14 Of. Ord. N° 7195 de 03-04-18 / 1.2. Pensión artículo 62 Ley N° 16.744 Of. Ord. N° 26837 de 07-12-18 Of. Ord. N° 29079 de 10-12-14 / 3.1. Desafectación Ley N° 18.458 Of. Ord. N° 16861 de 28-07-15 Of. Ord. N° 4882 de 29-02-16 |

Lines matching neither a record nor a group heading (appended to previous group): 0. Examples: []

Checks against ground truth:
- reference markers: raw 278, cleaned 278
- record lines (N.N / N.N.N) in raw text: 219; parsed records: 219

**Chunks: 57** | tokens min 27 / p10 31 / p50 89 / p90 266 / max 390 | over budget (400): 0 | < 30 tokens: 3 | split parts: 4
Chunks by unit level: {'records': 57}
Metadata coverage (chunks with value / total): source_id 57/57, doc_family 57/57, materia 57/57, tema 57/57, oficio_refs 57/57, book_date 57/57

## cms-aid-3590 -> `spensiones_articulo_faq` (articulo_faq.yaml)

- cleaner html_main_content: kept 10 blocks, title='¿Cuánto me deben descontar de mi sueldo para el seguro de cesantía?'
- cleaner html_collect: {keywords: 2, related_aids: 7}
- cleaner normalize_whitespace: ok
- Chars: raw 3,070 -> cleaned 1,350 (56.0% removed)

Checks against ground truth:
- boilerplate tokens left: menu=False, glossary=False, related=False, footer=False
- parsed keywords=['seguro de cesantía', 'remuneraciones'], related_aids=[3714, 3715, 3600, 3605, 3566, 2686, 7856]

**Chunks: 1** | tokens min 327 / p10 327 / p50 327 / p90 327 / max 327 | over budget (512): 0 | < 30 tokens: 0 | split parts: 0
Chunks by unit level: {'document': 1}
Metadata coverage (chunks with value / total): source_id 1/1, doc_family 1/1, aid 1/1, area 1/1, audiences 1/1, is_question 1/1, keywords 1/1, related_aids 1/1, last_modified 0/1

## cms-aid-17157 -> `spensiones_resolucion` (resolucion.yaml)

- Text layer: 4/4 pages textless -> simple parser raises ScannedDocumentError; profile `on_scanned: index_card`.
- Chars: raw 6 -> cleaned 316 (-5166.7% removed)

Checks against ground truth:

**Chunks: 1** | tokens min 74 / p10 74 / p50 74 / p90 74 / max 74 | over budget (600): 0 | < 30 tokens: 0 | split parts: 0
Chunks by unit level: {'index_card': 1}
Metadata coverage (chunks with value / total): source_id 1/1, doc_family 1/1, resolution_number 1/1, issued_date 1/1, audiences 1/1, content_mode 1/1, topic_class 1/1

## Summary

| sample | profile | chunks | p50 tok | p90 tok | max tok | over budget | <30 tok | total tok |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| compendio-sp-libro-i | spensiones_compendio | 1413 | 170 | 448 | 690 | 0 | 0 | 311555 |
| circular-2425 | spensiones_circular | 9 | 215 | 487 | 489 | 0 | 0 | 2179 |
| circular-cafp-100 | spensiones_circular | 4 | 150 | 235 | 235 | 0 | 0 | 638 |
| ncg-sp-372 | spensiones_ncg | 17 | 437 | 567 | 676 | 0 | 0 | 7222 |
| cms-aid-14843 | spensiones_jurisprudencia_indice | 57 | 89 | 266 | 390 | 0 | 3 | 7079 |
| cms-aid-3590 | spensiones_articulo_faq | 1 | 327 | 327 | 327 | 0 | 0 | 327 |
| cms-aid-17157 | spensiones_resolucion | 1 | 74 | 74 | 74 | 0 | 0 | 74 |
