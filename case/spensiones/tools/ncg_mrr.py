"""Pooled routed MRR@10 (merged top 8, router order) per set for dump.py outputs (stage 1 of the NCG pre-registration).
usage: python3 case/spensiones/tools/ncg_mrr.py <dump.json>...
"""
import collections, json, sys
from routed_report import merged

for path in sys.argv[1:]:
    rows = json.load(open(path, encoding="utf-8"))
    rr = lambda row: next((1 / i for i, (_, h) in enumerate(merged(row), 1) if h["relevant"]), 0.0)
    by = collections.defaultdict(list)
    for r in rows:
        by[r["set"]].append(rr(r))
    parts = " ".join(f"{s} {sum(v)/len(v):.3f}" for s, v in by.items())
    print(f"{path.rsplit('/',1)[-1]}: pooled {sum(map(rr, rows))/len(rows):.4f} (n {len(rows)}) | {parts}")
