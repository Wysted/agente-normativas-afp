"""Sanity check of eval_questions.jsonl against the profiles: for every question, the expected
snippet must land inside ONE chunk whose metadata matches the question's locator. This does
not measure retrieval; it proves each question is answerable from a single chunk and that the
locator metadata needed for filtered/exact-reference retrieval is actually extracted.

  uv run --no-project --python 3.12 --with tokenizers --with pyyaml python -I check_eval.py \
      --case case/spensiones
"""

import argparse
import bisect
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # own code dir only (not data)
import validate_profiles as v  # noqa: E402
import yaml  # noqa: E402


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def chunk_sample(case: Path, rel: str, sid: str, prof_file: str, rows: dict, count) -> tuple[list, list]:
    profile = yaml.safe_load((case / "profiles" / prof_file).read_text(encoding="utf-8"))
    path = case / "samples" / rel
    row = rows[sid]
    if path.suffix == ".html":
        doc = v.Doc(sid, row, [], html=path.read_text(encoding="utf-8", errors="replace"))
    else:
        doc = v.Doc(sid, row, v.load_pages(path))
    mode = "full_text"
    if path.suffix != ".html" and doc.pages and sum(1 for p in doc.pages if len(p.strip()) < 50) / len(doc.pages) > 0.5:
        mode = "index_card"
        card = v.render(profile["parser_params"].get("index_card_template", "{title}"),
                        {k: x for k, x in row.items() if isinstance(x, str)})
        doc.pages = [card.replace("\\n", "\n")]
    doc.parsed["content_mode"] = mode
    if mode == "full_text":
        for c in profile["parser_params"].get("cleaners", []):
            v.CLEANERS[c["name"]](doc, c.get("params", {}))
    text, page_starts = v.join_pages(doc)
    doc_vals = {"number": row.get("number"), "date": row.get("date"), "title_short": (row.get("title") or "")[:70],
                "libro": row.get("number"),
                "system": {"pensiones": "SP", "cesantia": "SC"}.get(row.get("area"), {"NP": "SP", "NC": "SC"}.get(row.get("subtype"), "SP"))}
    cp = dict(profile.get("chunker_params") or {})
    stats: dict = {}
    if mode == "index_card":
        drafts = [v.Draft(text, text, 0, len(text), {}, "", "index_card")]
    elif profile["chunker_key"] == "pattern_hierarchy":
        drafts = v.pattern_hierarchy(text, page_starts, doc_vals, cp, count, stats)
    elif profile["chunker_key"] == "records":
        drafts = v.records_chunker(text, cp, count, stats)
    else:
        drafts = v.structured_chunker(text, profile, count)
    issues: list[str] = []
    metas = []
    for d in drafts:
        md = v.resolve_metadata(profile, doc, d, text, doc_vals, page_starts, issues)
        if page_starts:
            md["page"] = bisect.bisect_right(page_starts, d.start)
        metas.append(md)
    return drafts, metas


def matches(md: dict, key: str, want) -> bool:
    have = md.get(key)
    if have is None:
        return False
    if isinstance(have, list):
        return want in have
    if key == "compendio_path":
        return str(have).startswith(str(want))
    return have == want


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    args = ap.parse_args()
    case = Path(args.case)
    rows = {}
    with open(case / "index" / "documents.jsonl", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            rows[r["id"]] = r
    count = v.TokenCounter()
    by_sample = {}
    for rel, sid, prof in v.SAMPLES:
        key = "samples/" + rel
        by_sample[key] = chunk_sample(case, rel, sid, prof, rows, count)
    by_sample["samples/resoluciones/cms-aid-17157.pdf"] = by_sample["samples/resoluciones/cms-aid-17157.txt"]
    qs = [json.loads(line) for line in open(case / "profiles" / "eval_questions.jsonl", encoding="utf-8")]
    lines = ["# Eval locator check", "", "| id | type | snippet in 1 chunk | locator match | chunk # | chunk tokens | detail |", "|---|---|---|---|---:|---:|---|"]
    ok_all = 0
    for q in qs:
        drafts, metas = by_sample[q["source_sample"]]
        snip = norm(q["expected_snippet"])
        hits = [i for i, d in enumerate(drafts) if snip in norm(d.body)]
        loc_ok = [i for i in hits if all(matches(metas[i], k, w) for k, w in q["locator"].items())]
        detail = ""
        if hits and not loc_ok:
            md = metas[hits[0]]
            detail = "; ".join(f"{k}: want {w!r} have {md.get(k)!r}" for k, w in q["locator"].items() if not matches(md, k, w))
        ok = bool(loc_ok)
        ok_all += ok
        idx = (loc_ok or hits or [None])[0]
        tok = count(drafts[idx].text) if idx is not None else "-"
        lines.append(f"| {q['id']} | {q['type']} | {'yes' if hits else 'NO'} | {'yes' if ok else 'NO'} | {idx if idx is not None else '-'} | {tok} | {detail} |")
    lines += ["", f"Questions whose answer lands in a single chunk with matching locator metadata: {ok_all}/{len(qs)}"]
    out = case / "profiles" / "validation" / "eval_locator_check.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
