# Corpus estimates per profile (index counts x sample ratios)

| family | docs | chunks/doc | est. chunks | est. tokens | embed time (chunk bound) | embed time (token bound) | basis |
|---|---:|---:|---:|---:|---:|---:|---|
| compendio | 10 | 1413 | 14,130 | 3,122,730 | 2.0 h | 52 min | Libro I sample: 1,413 chunks / 312,647 tokens |
| ncg | 485 | 17 | 8,245 | 3,545,350 | 69 min | 59 min | NCG 372 sample: 17 chunks / 7,305 tokens (PDF 175 KB vs index median 137 KB) |
| circular | 2520 | 9 | 22,680 | 5,488,560 | 3.1 h | 1.5 h | circular 2425 sample: 9 chunks / 2,179 tokens (185 KB); CAFP100: 4 chunks (53 KB) |
| articulo_faq | 204 | 1.2 | 245 | 80,784 | 2 min | 1 min | cms-aid-3590: 1 chunk / 327 tokens; guide pages assumed ~2 chunks |
| jurisprudencia_indice | 1 | 57 | 57 | 7,068 | 0 min | 0 min | cms-aid-14843: 57 chunks / 7,079 tokens |
| jurisprudencia_nota | 117 | 4 | 468 | 210,600 | 4 min | 4 min | NOT SAMPLED: assumption |
| resolucion_card | 287 | 1 | 287 | 21,238 | 2 min | 0 min | cms-aid-17157 index card: 1 chunk / 74 tokens |

Compendio caveat: the 10 Libros differ in size (PDF bytes: SP Libro I 8.4 MB, II 9.9 MB, IV 34 MB; SC Libro V 0.6 MB);
the per-Libro ratio of Libro I is a midpoint, real range ~0.4x-2x.

## Circulares breakdown

| series x listing tab | n |
|---|---:|
| CAFC / historica | 277 |
| CAFC / periodica | 163 |
| CAFP / historica | 1597 |
| CAFP / periodica | 417 |
| CCON / periodica | 15 |
| CIRC / historica | 31 |
| CIRC / periodica | 20 |

By decade: 1980s: 618, 1990s: 496, 2000s: 655, 2010s: 438, 2020s: 313

Circulares dated >= 2015: 521
NCG by system (subtype): {'NP': 371, 'NA': 2, 'NC': 112}; NCG >= 2020: 159

Future-dated rows (> crawl date 2026-10-15): 3
- cms-aid-17129 date=2027-04-01 source=title title='Régimen de Inversión de los Fondos de Pensiones vigente a partir del 1 de abril de 2027'
- cms-aid-14133 date=2030-03-31 source=title title='Comisiones pagadas por Fondos de Pensiones y de Cesantía, primer trimestre 2020'
- cms-aid-3498 date=2027-05-29 source=title title='Documento de Trabajo Nº 18: How to Integrate Disability Benefits into an Individual Account System: '

Resoluciones: 281; with index description (card text available): 247; section 'Normativa del ...' (OCR candidates): 185; lobby: 49
