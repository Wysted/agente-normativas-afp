"""Live hit: the dev snippet, whitespace removed, appears in a source the model saw.

usage: python3 case/spensiones/tools/live_hit.py <dev set jsonl> <answers dump> [<answers dump>...]
Prints per-phrasing counts per dump; with two dumps, also the questions that flip.
"""

import json
import re
import sys
import unicodedata


def squash(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", "", text).lower()


def is_hit(q: dict, row: dict) -> bool:
    sources = row.get("sources") or []
    if "expected_locators" in q:  # multi-chapter targets: any source under one of the prefixes
        return any((s.get("locator") or "").startswith(tuple(q["expected_locators"])) for s in sources)
    return any(squash(q["expected_snippet"]) in squash(s.get("text") or "") for s in sources)


def main() -> None:
    dev = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
    results = []
    for path in sys.argv[2:]:
        rows = {r["id"]: r for r in map(json.loads, open(path, encoding="utf-8"))}
        hit = {q["id"]: is_hit(q, rows[q["id"]]) for q in dev}
        counts = {}
        for q in dev:
            ph = q.get("phrasing", "all")
            got, total = counts.get(ph, (0, 0))
            counts[ph] = (got + hit[q["id"]], total + 1)
        print(path.rsplit("/", 2)[-2], {ph: f"{g}/{t}" for ph, (g, t) in counts.items()})
        results.append(hit)
    if len(results) == 2:
        for q in dev:
            a, b = results[0][q["id"]], results[1][q["id"]]
            if a != b:
                print(f"  {q['id']}: {a} -> {b}")


if __name__ == "__main__":
    main()
