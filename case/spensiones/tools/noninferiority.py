"""Pre-registered paired non-inferiority rule on live hits (and, optionally, bank_dev relevance).

usage: python3 case/spensiones/tools/noninferiority.py B1,B2,B3 N1,N2,N3 [--sets a,b,...]
           [--bank-relevance] [--new-family SUBSTR[,SUBSTR]] | --self-test
Runs are dirs under case/spensiones/judge/e2e. Rules:
- (a) mean pooled hits of N >= mean of B - range(B).
- (b, --bank-relevance) the same rule on bank_dev judge relevance (<run>/bank_dev_relevance.json).
- Stable loss: a row with >= 2/3 of the B runs hitting and no N hit; printed with N1's answer and
  sources for adjudication by content (target mismatch vs real loss). Stable gain: the mirror.
- The ood scope passes per run are reported; --new-family marks the source URIs of the family
  under test (default "NCG-"; the help-article run used "/w3-article-,/alt-article-").
"""

import argparse
import json
import math
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
E2E = ROOT / "case/spensiones/judge/e2e"
CAL = ROOT / "case/spensiones/calibration"
DEFAULT_SETS = "afiliacion_dev,cotizaciones_dev,compendio_new_dev,nota_dev,oficio_dev"
STABLE_FRACTION = 2 / 3


def squash(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFC", text)).lower()


def is_hit(q: dict, row: dict) -> bool:  # same rule as live_hit.py
    sources = row.get("sources") or []
    if "expected_locators" in q:
        return any((s.get("locator") or "").startswith(tuple(q["expected_locators"])) for s in sources)
    return any(squash(q["expected_snippet"]) in squash(s.get("text") or "") for s in sources)


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load(run: str, s: str) -> dict:
    return {r["id"]: r for r in read_rows(E2E / run / f"{s}_answers.jsonl")}


def hits(run: str, sets: list[str]) -> dict:
    out = {}
    for s in sets:
        rows = load(run, s)
        for q in read_rows(CAL / f"{s}.jsonl"):
            out[(s, q["id"])] = is_hit(q, rows[q["id"]])
    return out


def band(totals_b: list[int], totals_n: list[int]) -> tuple[float, float, int, bool]:
    """(mean B, mean N, range B, N passes): mean N >= mean B - range(B)."""
    mb, mn, rng = sum(totals_b) / len(totals_b), sum(totals_n) / len(totals_n), max(totals_b) - min(totals_b)
    return mb, mn, rng, mn >= mb - rng


def stable(hb: list[dict], hn: list[dict]) -> tuple[list, list]:
    """(stable losses, stable gains): >= 2/3 of the runs of one side hit, none of the other."""
    need = math.ceil(STABLE_FRACTION * len(hb))
    losses = [k for k in hb[0] if sum(h[k] for h in hb) >= need and not any(h[k] for h in hn)]
    gains = [k for k in hb[0] if not any(h[k] for h in hb) and sum(h[k] for h in hn) >= need]
    return losses, gains


def verdict(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def self_test() -> None:
    def runs(*rows: tuple[bool, ...]) -> list[dict]:
        return [{("s", f"q{i}"): v for i, v in enumerate(r)} for r in rows]

    # mean rule: B totals [3,3,2] (range 1, mean 2.67) -> threshold 1.67
    assert band([3, 3, 2], [2, 2, 2])[3] is True
    assert band([3, 3, 2], [1, 2, 1])[3] is False  # mean 1.33 < 1.67
    assert band([5, 5, 5], [4, 4, 4])[3] is False  # range 0: no slack
    # stable loss q0 (hit in 2/3 B, 0/3 N) and q1 (1/3 B: not stable); gain q2 (0 B, 2/3 N)
    hb = runs((True, True, False), (True, False, False), (False, False, False))
    hn = runs((False, False, True), (False, False, True), (False, False, False))
    assert stable(hb, hn) == ([("s", "q0")], [("s", "q2")])
    assert stable(hb, runs((True, False, False), (False, False, False), (False, False, False)))[0] == []
    q = {"expected_snippet": "a b"}
    assert is_hit(q, {"sources": [{"text": "xA\n B y"}]}) and not is_hit(q, {"sources": []})
    print("self-test ok")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base", nargs="?", help="comma-separated B runs")
    ap.add_argument("new", nargs="?", help="comma-separated N runs")
    ap.add_argument("--sets", default=DEFAULT_SETS, help="comma-separated live-hit sets")
    ap.add_argument("--bank-relevance", action="store_true", help="add decision (b): bank_dev judge relevance")
    ap.add_argument("--new-family", default="NCG-", help="comma-separated URI substrings of the new family")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not (args.base and args.new):
        ap.error("B runs and N runs are required")
    B, N, sets = args.base.split(","), args.new.split(","), args.sets.split(",")
    marks = tuple(args.new_family.split(","))

    def in_family(src: dict) -> bool:
        return any(m in (src.get("uri") or "") for m in marks)

    hb, hn = [hits(r, sets) for r in B], [hits(r, sets) for r in N]
    tb, tn = [sum(h.values()) for h in hb], [sum(h.values()) for h in hn]
    mb, mn, rng, ok = band(tb, tn)
    print(f"pooled hits B {tb} mean {mb:.2f} range {rng} | N {tn} mean {mn:.2f}")
    print(f"DECIDES (a): mean N {mn:.2f} >= mean B - range {mb - rng:.2f} -> {verdict(ok)}")
    for s in sets:
        keys = [k for k in hb[0] if k[0] == s]
        print(f"  {s}: B {[sum(h[k] for k in keys) for h in hb]} N {[sum(h[k] for k in keys) for h in hn]}")

    losses, gains = stable(hb, hn)
    print(f"\nstable losses {len(losses)} (adjudicate; real losses <= 2) | stable gains {len(gains)} (report)")
    for s, qid in losses:
        q = next(r for r in read_rows(CAL / f"{s}.jsonl") if r["id"] == qid)
        row = load(N[0], s)[qid]
        print(f"\n## {s} {qid}: {q['question']}\n   target: {q.get('expected_snippet') or q.get('expected_locators')}")
        print(f"   N1 answer: {row['answer'][:600]}")
        for src in row.get("sources") or []:
            print(f"   - [{src.get('n')}] {'NEW ' if in_family(src) else ''}{(src.get('title') or '')[:70]}"
                  f" | {(src.get('locator') or '')[:70]}")

    scope_in = lambda r: sum(1 for x in load(r, "ood").values() if (x.get("scope") or {}).get("level") in ("in", "low"))  # noqa: E731
    print("\nood scope passes:", {r: scope_in(r) for r in B + N})
    scanned = [*sets, "in_dev"]
    answers = [x for s in scanned for x in load(N[0], s).values()]
    first = sum(1 for x in answers if x.get("sources") and in_family(x["sources"][0]))
    any_ = sum(1 for x in answers if any(map(in_family, x.get("sources") or [])))
    print(f"N1 answers with a new-family source ({', '.join(marks)}): {any_}, first: {first} (report; outdated-text check on these)")

    if args.bank_relevance:
        if "bank_dev" not in sets:
            bank = load(N[0], "bank_dev").values()
            print("N1 bank_dev answers with / citing first a new-family source:",
                  sum(any(map(in_family, x.get("sources") or [])) for x in bank),
                  sum(bool(x.get("sources")) and in_family(x["sources"][0]) for x in bank))

        def rel(run: str) -> int:
            with (E2E / run / "bank_dev_relevance.json").open(encoding="utf-8") as f:
                return json.load(f)["aggregate"]["answer_relevance"]["passed"]

        rb, rn = [rel(r) for r in B], [rel(r) for r in N]
        mrb, mrn, rrng, rok = band(rb, rn)
        print(f"\nbank_dev relevance B {rb} mean {mrb:.2f} range {rrng} | N {rn} mean {mrn:.2f}")
        print(f"DECIDES (b): mean N {mrn:.2f} >= mean B - range {mrb - rrng:.2f} -> {verdict(rok)}")
        print(f"report only, superiority: min N {min(rn)} > max B {max(rb)} -> {min(rn) > max(rb)}")


if __name__ == "__main__":
    sys.exit(main())
