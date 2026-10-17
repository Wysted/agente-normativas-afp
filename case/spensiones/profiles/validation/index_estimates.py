"""Corpus-level estimates for the MVP decision (counts from the crawler index, ratios from
the sample validation). Treat index rows as data.

  uv run --no-project --python 3.12 python -I index_estimates.py \
      --index case/spensiones/index/documents.jsonl
"""

import argparse
import collections
import json
import re

# Measured on the samples (validation_report.md). Tokens = bge-m3, no special tokens.
MEASURED = {
    # family: (chunks per document, avg tokens per chunk, basis)
    "compendio": (1413, 221, "Libro I sample: 1,413 chunks / 312,647 tokens"),
    "ncg": (17, 430, "NCG 372 sample: 17 chunks / 7,305 tokens (PDF 175 KB vs index median 137 KB)"),
    "circular": (9, 242, "circular 2425 sample: 9 chunks / 2,179 tokens (185 KB); CAFP100: 4 chunks (53 KB)"),
    "articulo_faq": (1.2, 330, "cms-aid-3590: 1 chunk / 327 tokens; guide pages assumed ~2 chunks"),
    "jurisprudencia_indice": (57, 124, "cms-aid-14843: 57 chunks / 7,079 tokens"),
    "jurisprudencia_nota": (4, 450, "NOT SAMPLED: assumption"),
    "resolucion_card": (1, 74, "cms-aid-17157 index card: 1 chunk / 74 tokens"),
}
# TEI CPU, bge-m3, batch 16 x ~500-token chunks ~ 8 s (agintis _shared conventions).
SECONDS_PER_BATCH = 8.0
BATCH = 16
TOKENS_PER_SECOND = BATCH * 500 / SECONDS_PER_BATCH  # ~1,000 tok/s when batches are full


def route(r: dict) -> str | None:
    t = r.get("title") or ""
    dt = r["doc_type"]
    if dt == "compendio_libro":
        return "compendio"
    if dt == "compendio_titulo":
        return None  # navigation hubs
    if dt == "ncg":
        return "ncg"
    if dt == "circular":
        return "circular"
    if dt in ("guia_faq", "articulo_html") and re.match(r"\s*Nota Interna", t):
        return "jurisprudencia_nota"
    if dt == "guia_faq" and "Libro de Jurisprudencia" in t:
        return "jurisprudencia_indice"
    if dt == "resolucion" or (dt == "articulo_html" and re.match(r"\s*Resoluci[oó]n (?:Exenta|N)", t)):
        return "resolucion_card"
    if dt == "articulo_html":
        return "articulo_faq"
    return None


def fmt_time(seconds: float) -> str:
    return f"{seconds / 60:.0f} min" if seconds < 5400 else f"{seconds / 3600:.1f} h"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    args = ap.parse_args()
    rows = [json.loads(line) for line in open(args.index, encoding="utf-8")]
    fam = collections.Counter(route(r) for r in rows)
    out = ["# Corpus estimates per profile (index counts x sample ratios)", ""]
    out += ["| family | docs | chunks/doc | est. chunks | est. tokens | embed time (chunk bound) | embed time (token bound) | basis |",
            "|---|---:|---:|---:|---:|---:|---:|---|"]
    for name, (cpd, tpc, basis) in MEASURED.items():
        n = fam.get(name, 0)
        docs = n
        chunks = docs * cpd
        tokens = chunks * tpc
        t_chunks = chunks / BATCH * SECONDS_PER_BATCH
        t_tokens = tokens / TOKENS_PER_SECOND
        out.append(f"| {name} | {docs} | {cpd} | {chunks:,.0f} | {tokens:,.0f} | {fmt_time(t_chunks)} | {fmt_time(t_tokens)} | {basis} |")
    out += ["", "Compendio caveat: the 10 Libros differ in size (PDF bytes: SP Libro I 8.4 MB, II 9.9 MB, IV 34 MB; SC Libro V 0.6 MB);",
            "the per-Libro ratio of Libro I is a midpoint, real range ~0.4x-2x.", ""]

    circ = [r for r in rows if r["doc_type"] == "circular"]
    out += ["## Circulares breakdown", "", "| series x listing tab | n |", "|---|---:|"]
    for (sub, st), n in sorted(collections.Counter((r.get("subtype"), r.get("status")) for r in circ).items()):
        out.append(f"| {sub} / {st} | {n} |")
    dec = collections.Counter(((r.get("date") or "")[:3] + "0s") for r in circ)
    out += ["", "By decade: " + ", ".join(f"{k}: {v}" for k, v in sorted(dec.items())), ""]
    since = sum(1 for r in circ if (r.get("date") or "") >= "2015")
    out.append(f"Circulares dated >= 2015: {since}")
    ncg = [r for r in rows if r["doc_type"] == "ncg"]
    out.append(f"NCG by system (subtype): {dict(collections.Counter(r.get('subtype') for r in ncg))}; NCG >= 2020: {sum(1 for r in ncg if (r.get('date') or '') >= '2020')}")
    fut = [r for r in rows if (r.get("date") or "") > "2026-10-15"]
    out += ["", f"Future-dated rows (> crawl date 2026-10-15): {len(fut)}"]
    for r in fut:
        out.append(f"- {r['id']} date={r['date']} source={r.get('date_source')} title={r['title'][:100]!r}")
    res = [r for r in rows if r["doc_type"] == "resolucion"]
    out += ["", f"Resoluciones: {len(res)}; with index description (card text available): {sum(1 for r in res if r.get('description'))}; "
            f"section 'Normativa del ...' (OCR candidates): {sum(1 for r in res if any('Normativa del' in s for s in r['section']))}; "
            f"lobby: {sum(1 for r in res if any('Lobby' in s for s in r['section']))}"]
    print("\n".join(out))


if __name__ == "__main__":
    main()
