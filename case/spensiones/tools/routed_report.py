"""W1 routed report for a dump.py output: MRR@10 by gold system, primary hit@1, SP/SC confusion.

usage: python3 case/spensiones/tools/routed_report.py <dump.json>...
Confusion: the merged top 3 holds a chunk of the OTHER compendio topic (split runs), or, in a
single-profile run, a chunk whose text belongs to the other system (by the hit's source group).
"""

import json
import sys

CUT, WEIGHT = 8, 2
COMPENDIOS = {"compendio", "compendio_sc"}


def merged(row):
    groups = [(t["topic"], t["hits"]) for t in row["routed"]]
    weights = [WEIGHT] + [1] * (len(groups) - 1)
    out, seen, pos = [], set(), [0] * len(groups)
    while any(pos[i] < len(g) for i, (_, g) in enumerate(groups)):
        for i, (topic, g) in enumerate(groups):
            for h in g[pos[i]:pos[i] + weights[i]]:
                if h["id"] not in seen:
                    seen.add(h["id"])
                    out.append((topic, h))
            pos[i] += weights[i]
    return out[:CUT]


if __name__ == "__main__":
    for path in sys.argv[1:]:
        rows = json.load(open(path, encoding="utf-8"))
        rr = lambda row: next((1 / i for i, (_, h) in enumerate(merged(row), 1) if h["relevant"]), 0.0)
        out = [f"{path.rsplit('/', 1)[-1]}: W1 {sum(map(rr, rows)) / len(rows):.4f}"]
        for gold in ("compendio", "compendio_sc"):
            rs = [r for r in rows if r["expected"] == gold]
            if not rs:
                continue
            hit1 = sum(bool(r["routed"]) and r["routed"][0]["topic"] == gold for r in rs)
            other = (COMPENDIOS - {gold}).pop()
            confused = sum(any(t == other for t, _ in merged(r)[:3]) for r in rs)
            out.append(f"{gold}: n {len(rs)} W1 {sum(map(rr, rs)) / len(rs):.4f} primary {hit1}/{len(rs)}"
                       f" other-compendio in top3 {confused}")
        print("  ".join(out))
