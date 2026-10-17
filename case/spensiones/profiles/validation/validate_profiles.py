"""Validate the spensiones RagProfiles against the real sample texts.

Prototype implementations (reference behaviour for the template features) of:
  * parser-side cleaners declared in `parser_params.cleaners`
  * chunkers `pattern_hierarchy`, `records`, `structured` (markdown headings)
  * metadata sources `group:`, `document:`, `parsed:`, `const:`, `extractor:`
It reads the profile YAML files, so what is validated is exactly what is delivered.

Run (from any directory; sample/index files are data, never executed):
  uv run --no-project --with tokenizers --with pyyaml python -I validate_profiles.py \
      --case case/spensiones --out <dir>
"""

from __future__ import annotations

import argparse
import bisect
import collections
import json
import re
import statistics
import sys
import unicodedata
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable

import yaml

INGEST_DATE = "2026-10-15"
SPECIAL_TOKENS_RESERVE = 8
PAGE_MARK = re.compile(r"^=== PAGE (\d+) ===\s*$", re.M)

# ----------------------------------------------------------------------------- tokens


class TokenCounter:
    def __init__(self) -> None:
        self.kind = "chars/4"
        self._tok = None
        try:
            from tokenizers import Tokenizer

            self._tok = Tokenizer.from_pretrained("BAAI/bge-m3")
            self.kind = "bge-m3 (HF tokenizers, add_special_tokens=False)"
        except Exception as exc:  # offline -> approximation
            print(f"[warn] tokenizer unavailable ({exc}); using chars/4", file=sys.stderr)
        self._cache: dict[str, int] = {}

    def __call__(self, text: str) -> int:
        if text in self._cache:
            return self._cache[text]
        if self._tok is None:
            n = max(1, len(text) // 4)
        else:
            n = len(self._tok.encode(text, add_special_tokens=False).ids)
        if len(self._cache) < 200_000:
            self._cache[text] = n
        return n


# ----------------------------------------------------------------------------- loading


@dataclass
class Doc:
    sample: str
    row: dict[str, Any]
    pages: list[str]
    html: str | None = None
    parsed: dict[str, Any] = field(default_factory=dict)  # values exposed as `parsed:<key>`
    cleaner_log: list[str] = field(default_factory=list)


def load_pages(txt: Path) -> list[str]:
    raw = txt.read_text(encoding="utf-8")
    parts = PAGE_MARK.split(raw)
    if len(parts) == 1:
        return [raw]
    # parts = [pre, n1, text1, n2, text2, ...]
    return [parts[i + 1] for i in range(1, len(parts), 2)]


# ----------------------------------------------------------------------------- mini DOM


VOID = {"br", "img", "meta", "link", "input", "hr", "source", "area", "base", "col", "wbr"}


@dataclass
class Node:
    tag: str
    attrs: dict[str, str]
    children: list[Any] = field(default_factory=list)  # Node | str
    parent: "Node | None" = None


class DomBuilder(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = Node("#root", {})
        self.cur = self.root

    def handle_starttag(self, tag, attrs):
        node = Node(tag, {k: (v or "") for k, v in attrs}, parent=self.cur)
        self.cur.children.append(node)
        if tag not in VOID:
            self.cur = node

    def handle_endtag(self, tag):
        n = self.cur
        while n is not None and n.tag != tag:
            n = n.parent
        if n is not None and n.parent is not None:
            self.cur = n.parent

    def handle_data(self, data):
        self.cur.children.append(data)


def iter_nodes(node: Node):
    yield node
    for c in node.children:
        if isinstance(c, Node):
            yield from iter_nodes(c)


def select(root: Node, selector: str) -> list[Node]:
    """Tiny selector subset: '#id', '#id tag', 'tag', '[style*="x"]', '.class'."""
    parts = selector.split()
    scope = [root]
    for part in parts:
        found: list[Node] = []
        for s in scope:
            for n in iter_nodes(s):
                if n is s:
                    continue
                if _match(n, part):
                    found.append(n)
        scope = found
    return scope


def _match(n: Node, part: str) -> bool:
    if part.startswith("#"):
        return n.attrs.get("id") == part[1:]
    if part.startswith("."):
        return part[1:] in n.attrs.get("class", "").split()
    m = re.fullmatch(r"\[(\w+)\*=\"(.+)\"\]", part)
    if m:
        return m.group(2).replace(" ", "") in n.attrs.get(m.group(1), "").replace(" ", "")
    return n.tag == part


def text_of(n: Node) -> str:
    out = []
    for c in n.children:
        out.append(c if isinstance(c, str) else text_of(c))
    return "".join(out)


# ----------------------------------------------------------------------------- cleaners


def _nonempty_idx(lines: list[str]) -> list[int]:
    return [i for i, l in enumerate(lines) if l.strip()]


def c_drop_toc_pages(doc: Doc, p: dict) -> None:
    leader = re.compile(p["leader_pattern"])
    dropped = 0
    for page in doc.pages[: p.get("max_scan_pages", 15)]:
        lines = [l for l in page.splitlines() if l.strip()]
        ratio = sum(1 for l in lines if leader.search(l)) / max(1, len(lines))
        if ratio >= p.get("min_leader_ratio", 0.2):
            dropped += 1
        else:
            break
    doc.parsed["toc_pages"] = doc.pages[:dropped]
    doc.pages = doc.pages[dropped:]
    doc.cleaner_log.append(f"drop_toc_pages: dropped {dropped} leading TOC pages")


def c_strip_running_headers(doc: Doc, p: dict) -> None:
    pats = [re.compile(x) for x in p["patterns"]]
    n_lines = p.get("max_lines", 2)
    removed = 0
    new_pages = []
    for page in doc.pages:
        lines = page.splitlines()
        idx = _nonempty_idx(lines)
        zone = idx[:n_lines] if p.get("zone", "top") in ("top", "any") else []
        if p.get("zone") in ("bottom", "any"):
            zone += idx[-n_lines:]
        drop = {i for i in zone if any(r.search(lines[i]) for r in pats)}
        removed += len(drop)
        new_pages.append("\n".join(l for i, l in enumerate(lines) if i not in drop))
    doc.pages = new_pages
    doc.cleaner_log.append(f"strip_running_headers: removed {removed} lines")


def _norm_line(line: str, digits: bool) -> str:
    s = re.sub(r"\s+", " ", line.strip()).lower()
    return re.sub(r"\d+", "#", s) if digits else s


def c_strip_repeated_page_lines(doc: Doc, p: dict) -> None:
    pages = doc.pages
    if len(pages) < p.get("min_pages", 2):
        doc.cleaner_log.append("strip_repeated_page_lines: skipped (too few pages)")
        return
    top, bottom = p.get("zone_lines_top", 3), p.get("zone_lines_bottom", 3)
    digits = p.get("normalize_digits", True)
    zones = []
    counts: collections.Counter[str] = collections.Counter()
    for page in pages:
        lines = page.splitlines()
        idx = _nonempty_idx(lines)
        zone = set(idx[:top]) | set(idx[-bottom:] if bottom else [])
        zones.append((lines, zone))
        counts.update({_norm_line(lines[i], digits) for i in zone})
    threshold = max(2, p.get("min_page_ratio", 0.5) * len(pages))  # furniture repeats on >= 2 pages
    furniture = {k for k, v in counts.items() if v >= threshold and k not in ("#",)}
    removed = 0
    new_pages = []
    for lines, zone in zones:
        drop = {i for i in zone if _norm_line(lines[i], digits) in furniture}
        removed += len(drop)
        new_pages.append("\n".join(l for i, l in enumerate(lines) if i not in drop))
    doc.pages = new_pages
    doc.cleaner_log.append(
        f"strip_repeated_page_lines: furniture={sorted(furniture)[:6]} removed {removed} lines"
    )


def c_strip_lines(doc: Doc, p: dict) -> None:
    pats = [re.compile(x) for x in p["patterns"]]
    removed = 0
    new_pages = []
    for page in doc.pages:
        keep = []
        for l in page.splitlines():
            if any(r.search(l) for r in pats):
                removed += 1
            else:
                keep.append(l)
        new_pages.append("\n".join(keep))
    doc.pages = new_pages
    doc.cleaner_log.append(f"strip_lines: removed {removed} lines")


def c_dedupe_heading_echo(doc: Doc, p: dict) -> None:
    pat = re.compile(p["pattern"])
    window = p.get("window_lines", 4)

    def norm(k: str) -> str:
        k = k.lower()
        return re.sub(r"[\s.]+", "", k) if p.get("key_normalize") == "strip_dots_spaces" else re.sub(r"\s+", " ", k)
    removed = 0
    new_pages = []
    for page in doc.pages:
        lines = page.splitlines()
        drop: set[int] = set()
        i = 0
        while i < len(lines):
            m = pat.match(lines[i].strip())
            if m:
                key = norm(m.group("key"))
                for j in range(i + 1, min(len(lines), i + 1 + window)):
                    m2 = pat.match(lines[j].strip())
                    if m2 and norm(m2.group("key")) == key:
                        drop.update(range(i, j))
                        i = j
                        break
            i += 1
        removed += len(drop)
        new_pages.append("\n".join(l for k, l in enumerate(lines) if k not in drop))
    doc.pages = new_pages
    doc.cleaner_log.append(f"dedupe_heading_echo: removed {removed} echo lines")


def c_normalize_whitespace(doc: Doc, p: dict) -> None:
    new_pages = []
    for page in doc.pages:
        lines = page.splitlines()
        if p.get("strip_lines", True):
            lines = [l.strip() for l in lines]
        if p.get("collapse_inner_spaces", True):
            lines = [re.sub(r"[ \t ]{2,}", " ", l) for l in lines]
        out: list[str] = []
        blanks = 0
        for l in lines:
            if not l:
                blanks += 1
                if blanks > p.get("max_blank_lines", 1):
                    continue
            else:
                blanks = 0
            out.append(l)
        new_pages.append("\n".join(out).strip("\n"))
    doc.pages = new_pages
    doc.cleaner_log.append("normalize_whitespace: ok")


def c_normalize_quotes(doc: Doc, p: dict) -> None:
    mapping = p.get("map", {})
    new_pages = []
    for page in doc.pages:
        for a, b in mapping.items():
            page = page.replace(a, b)
        if p.get("ascii_double") == "alternate":
            state = {"open": True}

            def repl(_m, state=state):
                ch = "“" if state["open"] else "”"
                state["open"] = not state["open"]
                return ch

            page = re.sub('"', repl, page)
        new_pages.append(page)
    doc.pages = new_pages
    doc.cleaner_log.append("normalize_quotes: ok")


def c_drop_pages_before(doc: Doc, p: dict) -> None:
    pat = re.compile(p["pattern"], re.M)
    for i, page in enumerate(doc.pages):
        if pat.search(page):
            doc.pages = doc.pages[i if p.get("keep_matching_page", True) else i + 1 :]
            doc.cleaner_log.append(f"drop_pages_before: dropped {i} pages")
            return
    doc.cleaner_log.append("drop_pages_before: pattern not found (kept all)")


def c_join_continuation_lines(doc: Doc, p: dict) -> None:
    cont = re.compile(p["continuation"])
    joined = 0
    new_pages = []
    for page in doc.pages:
        out: list[str] = []
        for raw in page.splitlines():
            l = raw.strip()
            if not l:
                continue
            if out and cont.match(l):
                out[-1] = out[-1].rstrip() + " " + l
                joined += 1
            else:
                out.append(l)
        new_pages.append("\n".join(out))
    doc.pages = new_pages
    doc.cleaner_log.append(f"join_continuation_lines: joined {joined} lines")


def c_html_main_content(doc: Doc, p: dict) -> None:
    b = DomBuilder()
    b.feed(doc.html or "")
    root = b.root
    title_nodes = select(root, p["title_selector"])
    title = re.sub(r"\s+", " ", text_of(title_nodes[0])).strip() if title_nodes else doc.row["title"]
    excluded_ids: set[int] = set()
    for sel in p.get("exclude_selectors", []):
        for n in select(root, sel):
            for d in iter_nodes(n):
                excluded_ids.add(id(d))
    blocks: list[str] = [f"# {title}"]

    def walk(n: Node) -> None:
        if id(n) in excluded_ids:
            return
        if n.tag in ("p", "li") or n.tag in p.get("heading_selectors", []):
            txt = re.sub(r"\s+", " ", "".join(
                c if isinstance(c, str) else (text_of(c) if id(c) not in excluded_ids else "")
                for c in n.children)).strip()
            if not txt:
                return
            if n.tag in p.get("heading_selectors", []):
                blocks.append(f"## {txt}")
            elif n.tag == "p" and p.get("bold_paragraph_as_heading") and _whole_bold(n, txt):
                blocks.append(f"## {txt}")
            elif n.tag == "li":
                blocks.append(p.get("list_item_prefix", "- ") + txt)
            else:
                blocks.append(txt)
            return
        for c in n.children:
            if isinstance(c, Node):
                walk(c)

    for sel in p["include_selectors"]:
        for n in select(root, sel):
            walk(n)
    doc.pages = ["\n\n".join(blocks)]
    doc.parsed["_dom"] = root
    doc.cleaner_log.append(f"html_main_content: kept {len(blocks)} blocks, title={title!r}")


def _whole_bold(n: Node, txt: str) -> bool:
    bold = "".join(text_of(x) for x in iter_nodes(n) if x.tag in ("strong", "b") and x is not n)
    return bool(bold) and re.sub(r"\s+", " ", bold).strip() == txt


def c_html_collect(doc: Doc, p: dict) -> None:
    root = doc.parsed.get("_dom")
    if root is None:
        b = DomBuilder()
        b.feed(doc.html or "")
        root = b.root
    for name, spec in p["fields"].items():
        values: list[Any] = []
        for n in select(root, spec["selector"]):
            raw = n.attrs.get(spec["attr"], "") if "attr" in spec else text_of(n)
            if "regex" in spec:
                m = re.search(spec["regex"], raw)
                if m:
                    values.append(int(m.group(1)) if m.group(1).isdigit() else m.group(1))
            elif "split" in spec:
                values.extend(v.strip() for v in raw.split(spec["split"]) if v.strip())
            else:
                values.append(raw.strip())
        doc.parsed[name] = values
    doc.cleaner_log.append(f"html_collect: {{{', '.join(f'{k}: {len(doc.parsed[k])}' for k in p['fields'])}}}")


CLEANERS: dict[str, Callable[[Doc, dict], None]] = {
    "drop_toc_pages": c_drop_toc_pages,
    "strip_running_headers": c_strip_running_headers,
    "strip_repeated_page_lines": c_strip_repeated_page_lines,
    "strip_lines": c_strip_lines,
    "dedupe_heading_echo": c_dedupe_heading_echo,
    "normalize_whitespace": c_normalize_whitespace,
    "normalize_quotes": c_normalize_quotes,
    "drop_pages_before": c_drop_pages_before,
    "join_continuation_lines": c_join_continuation_lines,
    "html_main_content": c_html_main_content,
    "html_collect": c_html_collect,
}


def join_pages(doc: Doc) -> tuple[str, list[int]]:
    starts, buf, pos = [], [], 0
    for page in doc.pages:
        starts.append(pos)
        buf.append(page)
        pos += len(page) + 2
    return "\n\n".join(buf), starts


# ----------------------------------------------------------------------------- templates


_COND = re.compile(r"\{(\w+)\?((?:[^{}]|\{[\w:]+\})*)\}")
_VAR = re.compile(r"\{(\w+)(?::(\d+))?\}")


def render(template: str, values: dict[str, Any], top: bool = True) -> str:
    def cond(m):
        return render(m.group(2), values, top=False) if values.get(m.group(1)) else ""

    s = _COND.sub(cond, template)

    def var(m):
        v = values.get(m.group(1))
        v = "" if v is None else str(v)
        if m.group(2):
            n = int(m.group(2))
            v = v if len(v) <= n else v[: n - 1].rstrip() + "…"
        return v

    s = _VAR.sub(var, s)
    return re.sub(r"\s{2,}", " ", s).strip() if top else s


# ----------------------------------------------------------------------------- chunkers


@dataclass
class Draft:
    text: str
    body: str
    start: int
    end: int
    groups: dict[str, Any]
    heading: str
    level: str
    part: int | None = None


@dataclass
class LevelStat:
    matched: int = 0
    masked: int = 0
    examples: list[str] = field(default_factory=list)
    values: list[str] = field(default_factory=list)


def quote_spans(text: str, specs: list[dict]) -> list[tuple[int, int]]:
    spans = []
    for s in specs or []:
        o, c, mx = s["open"], s["close"], s.get("max_span_chars", 60000)
        if s.get("nesting"):
            depth, start = 0, -1
            for k, ch in enumerate(text):
                if ch == o:
                    if depth == 0:
                        start = k
                    depth += 1
                elif ch == c and depth:
                    depth -= 1
                    if depth == 0 and k - start <= mx:
                        spans.append((start, k + 1))
            continue
        i = 0
        while True:
            a = text.find(o, i)
            if a < 0:
                break
            b = text.find(c, a + 1)
            if b < 0 or b - a > mx:
                i = a + 1
                continue
            spans.append((a, b + 1))
            i = b + 1
    return spans


def in_spans(pos: int, spans: list[tuple[int, int]], starts: list[int]) -> bool:
    k = bisect.bisect_right(starts, pos) - 1
    return k >= 0 and spans[k][0] < pos < spans[k][1]


def recursive_split(text: str, budget: int, count: TokenCounter) -> list[str]:
    if count(text) <= budget:
        return [text]
    for sep in ("\n\n", r"(?<=[.;:])\n", "\n", r"(?<=[.;:])\s+", " "):
        pieces = re.split(sep, text) if sep.startswith("(") else text.split(sep)
        if len(pieces) > 1:
            joiner = "\n" if sep == r"(?<=[.;:])\n" else (" " if sep.startswith("(") or sep == " " else sep)
            out, cur = [], ""
            for piece in pieces:
                cand = piece if not cur else cur + joiner + piece
                if count(cand) <= budget:
                    cur = cand
                else:
                    if cur:
                        out.append(cur)
                    if count(piece) > budget:
                        out.extend(recursive_split(piece, budget, count))
                        cur = ""
                    else:
                        cur = piece
            if cur:
                out.append(cur)
            return out
    return [text]  # unreachable for normal text


def pattern_hierarchy(text: str, page_starts: list[int], doc_vars: dict, cp: dict,
                      count: TokenCounter, stats: dict[str, LevelStat]) -> list[Draft]:
    levels = cp["levels"]
    for i, lv in enumerate(levels):
        lv.setdefault("depth", i)
        lv["_re"] = re.compile(lv["pattern"], re.M)
        stats.setdefault(lv["name"], LevelStat())
    depth_of = {lv["name"]: lv["depth"] for lv in levels}
    split_depth = depth_of[cp["split_level"]]
    spans = quote_spans(text, cp.get("ignore_inside"))
    span_starts = [s[0] for s in spans]

    # Patterns are matched against ONE line at a time (never across a newline): a negated
    # class such as [^.;]* or a trailing \s* in re.M would otherwise swallow the next lines.
    line_starts, pos = [], 0
    for ln in text.split("\n"):
        line_starts.append((pos, ln))
        pos += len(ln) + 1
    matches = []  # (start, depth, order, level, groups, line)
    for order, lv in enumerate(levels):
        st = stats[lv["name"]]
        for lpos, line in line_starts:
            m = lv["_re"].match(line)
            if not m:
                continue
            if spans and in_spans(lpos, spans, span_starts):
                st.masked += 1
                continue
            matches.append((lpos, lv["depth"], order, lv, {k: v for k, v in m.groupdict().items() if v}, line))
    matches.sort(key=lambda x: (x[0], x[1], x[2]))
    # one match per position (shallowest wins)
    dedup, seen = [], set()
    for m in matches:
        if m[0] in seen:
            continue
        seen.add(m[0])
        dedup.append(m)
        st = stats[m[3]["name"]]
        st.matched += 1
        first = next(iter(m[4].values()), "")
        st.values.append(str(first))
        stats.setdefault("_seq", LevelStat()).values.append((m[3]["name"], str(first), m[5][:70]))
        if m[3]["name"] == "anexo":
            st.examples.append(m[5][:90]) if len(st.examples) >= 4 else None
        if len(st.examples) < 4:
            st.examples.append(m[5][:110])

    budget_total = cp.get("max_tokens", 700) - SPECIAL_TOKENS_RESERVE
    units = []  # (start, end, stack snapshot list[(lv, groups)])
    stack: dict[int, tuple[dict, dict]] = {}
    boundaries = []
    excise: list[tuple[int, int]] = []  # context-only heading lines removed from chunk bodies
    for start, depth, _o, lv, groups, line in dedup:
        if lv.get("keep_heading_line", True) is False:
            excise.append((start, start + len(line)))
        if lv.get("role") == "context":
            keys = lv.get("reset_on_change", [])
            cur = stack.get(depth)
            if cur and cur[0]["name"] == lv["name"] and all(cur[1].get(k) == groups.get(k) for k in keys):
                continue  # same context repeated on a new page: keep children open
        for d in [d for d in stack if d >= depth]:
            del stack[d]
        stack[depth] = (lv, groups)
        if depth <= split_depth:
            boundaries.append((start, dict(sorted(stack.items()))))
    deeper = [(s, d, lv, g) for s, d, _o, lv, g, _l in dedup if d > split_depth]

    drafts: list[Draft] = []
    first_start = boundaries[0][0] if boundaries else len(text)
    if first_start > 0 and cp.get("preamble", "keep") == "keep" and text[:first_start].strip():
        units.append((0, first_start, {}, True))
    for i, (start, snap) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
        units.append((start, end, snap, False))

    def heading_for(snap: dict) -> str:
        parts = [render(lv.get("heading_template", "{match}"), g) for _d, (lv, g) in snap.items()]
        return " > ".join(x for x in parts if x)

    def prefix_for(snap: dict, groups: dict, is_pre: bool) -> str:
        path = cp.get("preamble_heading", "Encabezado") if is_pre else heading_for(snap)
        vals = dict(doc_vars)
        vals.update(groups)
        vals["path"] = path
        return render(cp.get("heading_prefix_template", "{path}"), vals) if cp.get("include_heading", True) else ""

    raw_units = []
    for start, end, snap, is_pre in units:
        cut = [(a, b) for a, b in excise if start <= a < end]
        if cut:
            pieces, cur = [], start
            for a, b in cut:
                pieces.append(text[cur:a])
                cur = b
            pieces.append(text[cur:end])
            body = re.sub(r"\n{3,}", "\n\n", "".join(pieces)).strip()
        else:
            body = text[start:end].strip()
        if not body:
            continue
        groups: dict[str, Any] = {}
        for _d, (lv, g) in snap.items():
            groups.update(g)
        raw_units.append([start, end, snap, is_pre, body, groups])

    # fold small parent-intro units (a Capítulo heading + short text, "I. Introdúcense ...:")
    # into their first child: the child keeps its own metadata, the intro becomes context.
    fold_below = cp.get("fold_intro_below_tokens", 0)
    if fold_below:
        kept = []
        k = 0
        while k < len(raw_units):
            u = raw_units[k]
            nxt = raw_units[k + 1] if k + 1 < len(raw_units) else None
            if (nxt is not None and not u[3] and u[2] and nxt[2] and max(nxt[2]) > max(u[2])
                    and count(u[4]) < fold_below):
                nxt[0] = u[0]
                nxt[4] = u[4] + "\n" + nxt[4]
                k += 1
                continue
            kept.append(u)
            k += 1
        raw_units = kept

    # merge small units forward into the NEXT unit of the same level and parent; the merged
    # chunk keeps the first unit's heading and records every merged unit's groups.
    mb = cp.get("merge_below_tokens", 0)
    merged = []
    i = 0
    while i < len(raw_units):
        u = raw_units[i]
        u[5] = dict(u[5])
        u[5].setdefault("__all__", [dict(u[5])])
        while (mb and i + 1 < len(raw_units) and count(u[4]) < mb and not u[3]
               and _own(u[2]) == _own(raw_units[i + 1][2])
               and _parent(u[2], split_depth) == _parent(raw_units[i + 1][2], split_depth)
               and count(u[4] + "\n" + raw_units[i + 1][4]) <= budget_total):
            nxt = raw_units[i + 1]
            groups = dict(u[5])
            groups["__all__"] = u[5]["__all__"] + [dict(nxt[5])]
            u = [u[0], nxt[1], u[2], False, u[4] + "\n" + nxt[4], groups]
            i += 1
        merged.append(u)
        i += 1

    for start, end, snap, is_pre, body, groups in merged:
        prefix = prefix_for(snap, groups, is_pre)
        budget = budget_total - (count(prefix) + 1 if prefix else 0)
        level = "preamble" if is_pre else next(reversed(snap.values()))[0]["name"]
        heading = prefix
        if count(body) <= budget:
            drafts.append(Draft(_join(prefix, body), body, start, end, groups, heading, level))
            continue
        # oversize: split at deeper levels inside the unit, then recursive
        cuts = [s for s, d, lv, g in deeper if start < s < end and lv["name"] in cp.get("oversize", {}).get("levels", [])]
        for xp in cp.get("oversize", {}).get("extra_split_patterns", []):  # ignore quote masking
            cuts += [start + m.start() for m in re.finditer(xp, text[start:end], re.M) if m.start() > 0]
        cuts = sorted(set(cuts))
        pieces = []
        bounds = [start] + cuts + [end]
        segs = [text[bounds[k]:bounds[k + 1]].strip() for k in range(len(bounds) - 1)]
        cur = ""
        for seg in segs:
            if not seg:
                continue
            cand = seg if not cur else cur + "\n" + seg
            if count(cand) <= budget:
                cur = cand
            else:
                if cur:
                    pieces.append(cur)
                cur = seg
        if cur:
            pieces.append(cur)
        final = []
        for pc in pieces:
            final.extend(recursive_split(pc, budget, count) if count(pc) > budget else [pc])
        for k, pc in enumerate(final):
            drafts.append(Draft(_join(prefix, pc), pc, start, end, groups, heading, level, part=k))
    return drafts


def _own(snap: dict) -> str:
    return next(reversed(snap.values()))[0]["name"] if snap else ""


def _parent(snap: dict, split_depth: int) -> tuple:
    return tuple((d, lv["name"], tuple(sorted(g.items()))) for d, (lv, g) in snap.items() if d < split_depth)


def _join(prefix: str, body: str) -> str:
    return f"{prefix}\n{body}" if prefix else body


def records_chunker(text: str, cp: dict, count: TokenCounter, stats: dict[str, LevelStat]) -> list[Draft]:
    glevels = [(g["name"], re.compile(g["pattern"])) for g in cp["group_levels"]]
    rec = re.compile(cp["record_pattern"])
    for name, _ in glevels:
        stats.setdefault(name, LevelStat())
    stats.setdefault("record", LevelStat())
    budget = cp.get("max_tokens", 400) - SPECIAL_TOKENS_RESERVE
    ctx: dict[str, dict] = {}
    groups_out: list[tuple[dict, list[str]]] = []
    cur_key = None
    unmatched: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = rec.match(line)
        if m and re.match(r"^\d{1,2}\.\d", line):
            st = stats["record"]
            st.matched += 1
            if len(st.examples) < 4:
                st.examples.append(line[:110])
            if not groups_out:
                groups_out.append(({}, []))
            groups_out[-1][1].append(line)
            continue
        hit = False
        for gi, (name, gre) in enumerate(glevels):
            gm = gre.match(line)
            if gm:
                st = stats[name]
                st.matched += 1
                if len(st.examples) < 4:
                    st.examples.append(line[:110])
                ctx[name] = {k: v for k, v in gm.groupdict().items() if v}
                for deeper, _ in glevels[gi + 1:]:
                    ctx.pop(deeper, None)
                merged = {}
                for n, _ in glevels:
                    merged.update(ctx.get(n, {}))
                if name == cp["group_by"]:
                    groups_out.append((merged, [line]))
                hit = True
                break
        if not hit:
            unmatched.append(line)
            if groups_out:
                groups_out[-1][1].append(line)
    drafts = []
    for g, lines in groups_out:
        header = " > ".join(x for x in (g.get("materia_nombre"), g.get("rama_nombre"), g.get("tema_nombre")) if x)
        cur: list[str] = []
        part = 0
        for line in lines:
            cand = "\n".join(([header] if cp.get("repeat_group_header") else []) + cur + [line])
            if cur and count(cand) > budget:
                body = "\n".join(cur)
                drafts.append(Draft(_join(header, body), body, 0, 0, dict(g), header, "records", part))
                part += 1
                cur = [line]
            else:
                cur.append(line)
        if cur:
            body = "\n".join(cur)
            drafts.append(Draft(_join(header, body), body, 0, 0, dict(g), header, "records", part if part else None))
    stats["_unmatched_lines"] = LevelStat(matched=len(unmatched), examples=unmatched[:6])
    return drafts


def structured_chunker(text: str, profile: dict, count: TokenCounter) -> list[Draft]:
    budget = profile["chunk_max_tokens"] - SPECIAL_TOKENS_RESERVE
    sections: list[tuple[list[str], list[str]]] = []
    path: list[str] = []
    for block in text.split("\n\n"):
        hm = re.match(r"^(#{1,6})\s+(.*)$", block.strip())
        if hm:
            lvl = len(hm.group(1))
            path = path[: lvl - 1] + [hm.group(2)]
            sections.append((list(path), []))
        else:
            if not sections:
                sections.append(([], []))
            sections[-1][1].append(block.strip())
    drafts = []
    # merge sibling sections of the same top heading while within budget (whole FAQ = 1 chunk)
    title = sections[0][0][0] if sections and sections[0][0] else ""
    whole = "\n\n".join(
        ("\n".join([("## " + h[-1]) if len(h) > 1 else ""] + paras)).strip() for h, paras in sections
    ).strip()
    if count(title + "\n" + whole) <= budget:
        return [Draft(title + "\n" + whole, whole, 0, len(text), {}, title, "document")]
    for h, paras in sections:
        heading = " > ".join(h)
        for pc in recursive_split("\n\n".join(paras), budget - count(heading) - 1, count):
            drafts.append(Draft(heading + "\n" + pc, pc, 0, 0, {}, heading, "section"))
    return drafts


# ----------------------------------------------------------------------------- extractors

MONTHS = {m: i for i, m in enumerate(
    "enero febrero marzo abril mayo junio julio agosto septiembre octubre noviembre diciembre".split(), 1)}
MONTHS["setiembre"] = 9
DATE_RE = re.compile(r"(\d{1,2})[°º]?\s+de\s+(" + "|".join(MONTHS) + r")\s+(?:de|del)\s+(\d{4})", re.I)


def es_date(s: str) -> str | None:
    m = DATE_RE.search(s)
    if not m:
        return None
    return f"{int(m.group(3)):04d}-{MONTHS[m.group(2).lower()]:02d}-{int(m.group(1)):02d}"


def x_vigencia_date(doc_text: str, **_) -> str | None:
    m = re.search(r"^(?:[IVX]{1,4}\.\s*)?VIGENCIA\s*:?\s*$", doc_text, re.M)
    if m:
        return es_date(doc_text[m.end(): m.end() + 700])
    m = re.search(r"(?:regir[áa]|comenzar[áa]n? a regir|entrar[áa] en vigencia|vigencia a contar)[^.]{0,160}", doc_text, re.I)
    return es_date(m.group(0)) if m else None


LEGAL_PATTERNS = [
    (re.compile(r"\bD\.\s?L\.\s*N[°º]\s*([\d.]+)"), "DL {0}"),
    (re.compile(r"\bDecreto\s+Ley\s+N[°º]\s*([\d.]+)"), "DL {0}"),
    (re.compile(r"\bD\.\s?F\.\s?L\.\s*N[°º]\s*([\d.]+)"), "DFL {0}"),
    (re.compile(r"\bD\.\s?S\.(?:\s+conjunto)?\s+N[°º]\s*(\d+)"), "DS {0}"),
    (re.compile(r"\b[Ll]ey\s+N[°º]\s*([\d.]+)"), "Ley {0}"),
]


def x_legal_refs(chunk: str, **_) -> list[str]:
    out = []
    for r, fmt in LEGAL_PATTERNS:
        for m in r.finditer(chunk):
            num = m.group(1).replace(".", "").strip()
            if num:
                v = fmt.format(num)
                if v not in out:
                    out.append(v)
    return out


def x_regex_all(chunk: str, doc_text: str, params: dict, **_) -> list[Any]:
    flags = re.I if "i" in params.get("flags", "") else 0
    target = doc_text if params.get("scope") == "document" else chunk
    vals = []
    for m in re.finditer(params["pattern"], target, flags):
        v = m.group(params.get("group", 0))
        if params.get("normalize") == "compact_spaces":
            v = re.sub(r"\s+", " ", v).strip()
        if v.isdigit():
            v = int(v)
        if not params.get("unique") or v not in vals:
            vals.append(v)
    return vals


def x_compendio_path(groups: dict, doc_vals: dict, **_) -> str | None:
    if not groups.get("titulo"):
        return None
    p = f"{doc_vals.get('system')}/L{doc_vals.get('libro')}/T{groups['titulo']}"
    if groups.get("letra"):
        p += f"/{groups['letra']}"
    if groups.get("capitulo"):
        p += f"/C{groups['capitulo']}"
    elif groups.get("anexo"):
        p += f"/AX{groups['anexo']}"
    return p  # chapter-level join key; numbered items live in `numeros`


TARGET_RES = {
    "libro": re.compile(r"\bLibro\s+([IVX]+)\b", re.I),
    "titulo": re.compile(r"\bT[ií]tulo\s+([IVXL]+)\b", re.I),
    "letra": re.compile(r"\bLetra\s+([A-Z](?:\.\d)?)(?=[\s.,])"),
    "capitulo": re.compile(r"\bCap[ií]tulo\s+([IVXL]+)\b", re.I),
    "numero": re.compile(r"\bn[uú]mero\s+(\d{1,2}(?:\.\d{1,2})*)(?!\.\d)"),
    "anexo": re.compile(r"\bAnexo\s+N[°º]\s*(\d+)"),
}


def _strip_quotes(s: str) -> str:
    return re.sub(r"“[^”]*”", " ", s.split("“")[0] if s.count("“") > s.count("”") else s)


def x_compendio_target(draft: Draft, doc_vals: dict, doc_title: str, **_) -> dict[str, Any]:
    out: dict[str, Any] = {}
    # inherit: document title -> parte heading -> enmienda heading (outside quotes)
    layers = [doc_title, draft.groups.get("parte_head", ""), draft.groups.get("enmienda_head", "")]
    for layer in layers:
        s = _strip_quotes(layer)[:260]
        found = {k: r.search(s) for k, r in TARGET_RES.items()}
        for k, m in found.items():
            if m:
                out[k] = m.group(1).upper() if k in ("libro", "titulo", "capitulo") else m.group(1)
        if found["letra"] and not found["capitulo"]:
            out.pop("capitulo", None)
            out.pop("numero", None)
    if not out.get("libro"):
        return {}
    path = f"{doc_vals.get('system')}/L{out['libro']}/T{out.get('titulo', '?')}"
    if out.get("letra"):
        path += f"/{out['letra']}"
    if out.get("capitulo"):
        path += f"/C{out['capitulo']}"
    elif out.get("anexo"):
        path += f"/AX{out['anexo']}"
    out["path"] = path  # same chapter-level key as compendio.compendio_path
    return out


VERB_LEMMA = [("reempl", "replace"), ("sustit", "replace"), ("modif", "modify"), ("agreg", "add"),
              ("agrég", "add"), ("incorp", "add"), ("interc", "add"), ("elim", "delete"),
              ("supr", "delete"), ("derog", "repeal"), ("introd", "introduce"), ("rectif", "modify"),
              ("renum", "renumber"), ("reenum", "renumber")]


def verb_lemma(v: str) -> str | None:
    s = unicodedata.normalize("NFKD", v.lower()).encode("ascii", "ignore").decode()
    for stem, lemma in VERB_LEMMA:
        stem_ascii = unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode()
        if s.startswith(stem_ascii):
            return lemma
    return None


def apply_transform(value: Any, tr: dict | None, report: list[str]) -> Any:
    if value is None or not tr:
        return value
    if "value_map" in tr:
        value = tr["value_map"].get(value, value)
    if tr.get("date_floor") and isinstance(value, str):
        if re.fullmatch(r"\d{4}", value):
            value = value + "-01-01"
        elif re.fullmatch(r"\d{4}-\d{2}", value):
            value = value + "-01"
    if tr.get("reject_after") == "ingest_date" and isinstance(value, str) and value > INGEST_DATE:
        report.append(f"rejected future date {value}")
        return None
    if tr.get("path_segment") == "last" and isinstance(value, list):
        value = sorted({v.split(">")[-1].strip() for v in value})
    if tr.get("verb_lemma") and isinstance(value, str):
        value = verb_lemma(value)
    return value


TYPE_CHECK = {
    "str": lambda v: isinstance(v, str),
    "int": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "float": lambda v: isinstance(v, (int, float)),
    "date": lambda v: isinstance(v, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", v) is not None,
    "str_list": lambda v: isinstance(v, list) and all(isinstance(x, str) for x in v),
    "int_list": lambda v: isinstance(v, list) and all(isinstance(x, int) for x in v),
}


def coerce(v: Any, typ: str) -> Any:
    if v is None:
        return None
    if typ == "int" and isinstance(v, str) and v.isdigit():
        return int(v)
    if typ in ("str_list", "int_list") and not isinstance(v, list):
        v = [v]
    if typ == "int_list":
        return [int(x) for x in v if str(x).isdigit()]
    return v


def resolve_metadata(profile: dict, doc: Doc, draft: Draft, doc_text: str, doc_vals: dict,
                     page_starts: list[int], issues: list[str]) -> dict[str, Any]:
    md: dict[str, Any] = {}
    target_cache: dict[str, Any] | None = None
    for f in profile["metadata_fields"]:
        src: str = f["source"]
        kind, _, key = src.partition(":")
        v: Any = None
        if kind == "document":
            v = doc.row.get(key)
        elif kind == "const":
            v = key
        elif kind == "group":
            if f["type"].endswith("_list"):
                v = [g[key] for g in draft.groups.get("__all__", [draft.groups]) if g.get(key)]
            else:
                v = draft.groups.get(key)
        elif kind == "parsed":
            v = doc.parsed.get(key)
        elif kind == "extractor":
            name, _, sub = key.partition(".")
            if name == "regex_all":
                v = x_regex_all(draft.body, doc_text, f.get("params", {}))
            elif name == "legal_refs":
                v = x_legal_refs(draft.body)
            elif name == "vigencia_date":
                v = doc_vals.setdefault("_vig", x_vigencia_date(doc_text))
            elif name == "compendio_path":
                v = x_compendio_path(draft.groups, doc_vals)
            elif name == "compendio_target":
                if target_cache is None:
                    target_cache = x_compendio_target(draft, doc_vals, doc.row.get("title", ""))
                v = target_cache.get(sub)
            elif name == "page_end":
                v = bisect.bisect_right(page_starts, max(draft.start, draft.end - 1)) if page_starts else None
            elif name == "ncg_role":
                v = "amendment" if re.search(r"MODIFICA|Introd[uú]cense", doc.row.get("title", "") + doc_text[:4000], re.I) else "standalone"
            elif name == "title_is_question":
                v = "yes" if (doc.row.get("title") or "").strip().endswith("?") else "no"
            elif name == "resolution_topic":
                secs = " | ".join(doc.row.get("section") or [])
                v = "lobby" if "Lobby" in secs else "normativa" if "Normativa del" in secs else "institucional"
            else:
                issues.append(f"unknown extractor {name}")
        v = apply_transform(v, f.get("transform"), issues)
        v = coerce(v, f["type"])
        if v in (None, [], ""):
            continue
        if not TYPE_CHECK[f["type"]](v):
            issues.append(f"type mismatch {f['name']}={v!r} ({f['type']})")
            continue
        md[f["name"]] = v
    return md


# ----------------------------------------------------------------------------- run


SAMPLES = [
    ("compendio/compendio-sp-libro-i.txt", "compendio-sp-libro-i", "compendio.yaml"),
    ("circulares/circular-2425.txt", "circular-2425", "circular.yaml"),
    ("circulares/circular-cafp-100.txt", "circular-cafp-100", "circular.yaml"),
    ("ncg/ncg-sp-372.txt", "ncg-sp-372", "ncg.yaml"),
    ("guia_faq/cms-aid-14843.txt", "cms-aid-14843", "guia_jurisprudencia.yaml"),
    ("articulos_html/cms-aid-3590.html", "cms-aid-3590", "articulo_faq.yaml"),
    ("resoluciones/cms-aid-17157.txt", "cms-aid-17157", "resolucion.yaml"),
]


def pct(values: list[int], q: float) -> int:
    if not values:
        return 0
    s = sorted(values)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def roman_to_int(s: str) -> int:
    vals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
    total = 0
    for i, ch in enumerate(s):
        v = vals.get(ch, 0)
        total += -v if i + 1 < len(s) and vals.get(s[i + 1], 0) > v else v
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    case = Path(args.case)
    out = Path(args.out)
    profiles_dir = case / "profiles"
    rows = {}
    with open(case / "index" / "documents.jsonl", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            rows[r["id"]] = r
    count = TokenCounter()
    report: list[str] = [
        "# Validation report: spensiones RagProfiles vs samples",
        "",
        f"Token counter: {count.kind}. Budgets subtract {SPECIAL_TOKENS_RESERVE} special-token reserve.",
        f"Ingest date used for future-date rejection: {INGEST_DATE}.",
        "",
    ]
    previews = []
    summary_rows = []
    for rel, sid, prof_file in SAMPLES:
        profile = yaml.safe_load((profiles_dir / prof_file).read_text(encoding="utf-8"))
        path = case / "samples" / rel
        row = rows[sid]
        if path.suffix == ".html":
            doc = Doc(sid, row, [], html=path.read_text(encoding="utf-8", errors="replace"))
        else:
            doc = Doc(sid, row, load_pages(path))
        raw_text = "\n".join(doc.pages)
        if doc.html is not None and path.with_suffix(".txt").exists():
            raw_text = path.with_suffix(".txt").read_text(encoding="utf-8")  # naive tag-stripped text
        doc.parsed["_raw"] = raw_text
        report += [f"## {sid} -> `{profile['key']}` ({prof_file})", ""]
        issues: list[str] = []
        content_mode = "full_text"
        if path.suffix != ".html":
            textless = sum(1 for p in doc.pages if len(p.strip()) < 50)
            if doc.pages and textless / len(doc.pages) > 0.5:
                content_mode = "index_card"
                report.append(f"- Text layer: {textless}/{len(doc.pages)} pages textless -> simple parser raises ScannedDocumentError; profile `on_scanned: {profile['parser_params'].get('on_scanned')}`.")
                card = render(profile["parser_params"].get("index_card_template", "{title}"), {k: v for k, v in row.items() if isinstance(v, str)})
                doc.pages = [card.replace("\\n", "\n")]
        doc.parsed["content_mode"] = content_mode
        if content_mode == "full_text":
            for c in profile["parser_params"].get("cleaners", []):
                CLEANERS[c["name"]](doc, c.get("params", {}))
        for line in doc.cleaner_log:
            report.append(f"- cleaner {line}")
        text, page_starts = join_pages(doc)
        report.append(f"- Chars: raw {len(raw_text):,} -> cleaned {len(text):,} ({100 * (1 - len(text) / max(1, len(raw_text))):.1f}% removed)")
        doc_vals = {
            "number": row.get("number"), "date": row.get("date"), "title_short": (row.get("title") or "")[:70],
            "libro": row.get("number"),
            "system": {"pensiones": "SP", "cesantia": "SC"}.get(row.get("area"), {"NP": "SP", "NC": "SC"}.get(row.get("subtype"), "SP")),
        }
        stats: dict[str, LevelStat] = {}
        ck = profile["chunker_key"]
        cp = dict(profile.get("chunker_params") or {})
        if content_mode == "index_card":
            body = text.strip()
            drafts = [Draft(body, body, 0, len(body), {}, "", "index_card")]
        elif ck == "pattern_hierarchy":
            if profile["key"] == "spensiones_ncg":
                tgt = x_compendio_target(Draft("", "", 0, 0, {}, "", ""), doc_vals, row.get("title", ""))
                doc_vals["target_label"] = " ".join(f"{k.capitalize()} {v}" for k, v in tgt.items() if k != "path")
            drafts = pattern_hierarchy(text, page_starts, doc_vals, cp, count, stats)
        elif ck == "records":
            drafts = records_chunker(text, cp, count, stats)
        elif ck == "structured":
            drafts = structured_chunker(text, profile, count)
        else:
            raise SystemExit(f"unknown chunker {ck}")

        # level table
        if stats:
            report += ["", "| level | matches | masked (inside quotes) | examples |", "|---|---:|---:|---|"]
            for name, st in stats.items():
                if name.startswith("_"):
                    continue
                ex = " / ".join(e.replace("|", "\\|") for e in st.examples[:3])
                report.append(f"| {name} | {st.matched} | {st.masked} | {ex} |")
            if "_unmatched_lines" in stats:
                u = stats["_unmatched_lines"]
                report.append(f"\nLines matching neither a record nor a group heading (appended to previous group): {u.matched}. Examples: {u.examples}")
        report += sample_checks(sid, doc, text, stats, drafts)

        budget = profile["chunk_max_tokens"]
        toks = [count(d.text) for d in drafts]
        over = sum(1 for t in toks if t > budget - SPECIAL_TOKENS_RESERVE)
        tiny = sum(1 for t in toks if t < 30)
        metas = []
        for d in drafts:
            md = resolve_metadata(profile, doc, d, text, doc_vals, page_starts, issues)
            if page_starts:
                md["page"] = bisect.bisect_right(page_starts, d.start)
            metas.append(md)
        report += [
            "",
            f"**Chunks: {len(drafts)}** | tokens min {min(toks) if toks else 0} / p10 {pct(toks, .1)} / p50 {pct(toks, .5)} / p90 {pct(toks, .9)} / max {max(toks) if toks else 0} | over budget ({budget}): {over} | < 30 tokens: {tiny} | split parts: {sum(1 for d in drafts if d.part)}",
        ]
        by_level = collections.Counter(d.level for d in drafts)
        report.append(f"Chunks by unit level: {dict(by_level)}")
        cov = collections.Counter(k for md in metas for k in md)
        report.append("Metadata coverage (chunks with value / total): " + ", ".join(
            f"{f['name']} {cov.get(f['name'], 0)}/{len(drafts)}" for f in profile["metadata_fields"]))
        if issues:
            report.append(f"Issues: {collections.Counter(issues).most_common(8)}")
        report.append("")
        summary_rows.append((sid, profile["key"], len(drafts), pct(toks, .5), pct(toks, .9), max(toks) if toks else 0, over, tiny, sum(toks)))
        pick = drafts[:3] + drafts[len(drafts) // 2: len(drafts) // 2 + 1] + drafts[-1:]
        for d in pick:
            i = drafts.index(d)
            previews.append({"sample": sid, "profile": profile["key"], "ordinal": i, "tokens": toks[i],
                             "level": d.level, "part": d.part, "metadata": metas[i], "text": d.text[:600]})
        if sid == "compendio-sp-libro-i":
            dump_compendio_examples(drafts, metas, toks, report)
        if sid == "ncg-sp-372":
            report += ["NCG chunk -> Compendio target (one row per chunk):", "", "| # | heading | op_verb | compendio_path | tokens |", "|---:|---|---|---|---:|"]
            for i, (d, md) in enumerate(zip(drafts, metas)):
                report.append(f"| {i} | {d.heading[:90].replace('|', '/')} | {md.get('op_verb', '')} | {md.get('compendio_path', '')} | {toks[i]} |")
            report.append("")

    report += ["## Summary", "", "| sample | profile | chunks | p50 tok | p90 tok | max tok | over budget | <30 tok | total tok |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summary_rows:
        report.append("| " + " | ".join(str(x) for x in r) + " |")
    out.mkdir(parents=True, exist_ok=True)
    (out / "validation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    with open(out / "chunks_preview.jsonl", "w", encoding="utf-8") as fh:
        for p in previews:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    print("\n".join(report))


def dump_compendio_examples(drafts, metas, toks, report):
    report.append("Example compendio chunks (heading prefix | compendio_path | amended_by_ncg):")
    report.append("")
    seen = 0
    for d, md, t in zip(drafts, metas, toks):
        if md.get("amended_by_ncg") and seen < 4:
            report.append(f"- `{d.text.splitlines()[0][:120]}` | {md.get('compendio_path')} | {md.get('amended_by_ncg')} | {t} tok")
            seen += 1
    report.append("")


def sample_checks(sid: str, doc: Doc, text: str, stats: dict[str, LevelStat], drafts: list[Draft]) -> list[str]:
    out = ["", "Checks against ground truth:"]
    if sid == "compendio-sp-libro-i":
        toc = "\n".join(doc.parsed.get("toc_pages", []))
        toc_caps = re.findall(r"^• Cap[ií]tulo\.?\s*([IVXL]+)\b", toc, re.M)
        toc_anx = re.findall(r"^• Anexo\b", toc, re.M)
        body_caps = stats["capitulo"].values
        out.append(f"- capitulo: TOC lists {len(toc_caps)} chapter entries; body matches {len(body_caps)} after echo dedupe. Sequence check (TOC order vs body order): {'IDENTICAL' if toc_caps == body_caps else 'DIFFERENT'}")
        if toc_caps != body_caps:
            sm = __import__("difflib").SequenceMatcher(a=toc_caps, b=body_caps)
            tp = sum(b.size for b in sm.get_matching_blocks())
            out.append(f"  - aligned TP={tp}, precision={tp / max(1, len(body_caps)):.3f}, recall={tp / max(1, len(toc_caps)):.3f}")
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag != "equal":
                    out.append(f"  - {tag}: TOC{toc_caps[i1:i2]} vs body{body_caps[j1:j2]}")
        out.append(f"- anexo: TOC lists {len(toc_anx)} annex entries; body matches {stats['anexo'].matched}")
        # numero sequence consistency per parent (reset on seccion/capitulo/anexo matches)
        prev, ok, bad, gaps, badex = 0, 0, 0, 0, []
        for name, val, line in stats["_seq"].values:
            if name in ("capitulo", "anexo"):
                prev = 0
                continue
            if name != "numero":
                continue
            v = int(val)
            if v == prev + 1 or v == 1:
                ok += 1
                prev = v
            elif v > prev + 1 and v - prev <= 3:
                gaps += v - prev - 1
                ok += 1
                prev = v
            else:
                bad += 1
                badex.append(f"{prev}->{v} «{line[:60]}»")
                prev = v
        out.append(f"- numero: {ok} in-sequence, {bad} out-of-sequence, {gaps} numbers skipped inside runs (suspected FN). Out-of-sequence cases (review manually): {badex}")
        toc_anx_names = [re.sub(r"\W+", " ", x).strip().lower()[:22] for x in re.findall(r"^• (Anexo.*?)(?:\.{3,}|$)", toc, re.M)]
        body_anx = [re.sub(r"\W+", " ", line).strip().lower()[:22] for name, _v, line in stats["_seq"].values if name == "anexo"]
        ct, cb = collections.Counter(toc_anx_names), collections.Counter(body_anx)
        missing = list((ct - cb).elements())
        extra = list((cb - ct).elements())
        tp = sum((ct & cb).values())
        out.append(f"- anexo (multiset vs TOC): TP={tp}, precision={tp / max(1, len(body_anx)):.3f}, recall={tp / max(1, len(toc_anx_names)):.3f}")
        out.append(f"- anexo detail: TOC entries missing in body matches: {missing}; body matches not in TOC: {extra}")
        hdr = len(re.findall(r"^Compendio de Normas del Sistema de Pensiones - Libro I", text, re.M))
        leaders = len(re.findall(r"\.{4,}\s*\d+\s*$", text, re.M))
        out.append(f"- running header lines left after cleaning: {hdr}; dotted-leader TOC lines left: {leaders}")
    elif sid == "circular-2425":
        out.append(f"- expected numero 1,2,3,4 + seccion VIGENCIA, ANEXO -> got numero {stats['numero'].values}, seccion {stats['seccion'].values}")
        out.append(f"- footer lines left: {len(re.findall(r'CI-DDN', text))} (one legit in IDENTIFICACIÓN INTERNA)")
    elif sid == "circular-cafp-100":
        out.append(f"- expected numero 1,2,3 + literal a..g -> got numero {stats['numero'].values}, literal {stats['literal'].values}")
        out.append(f"- letterhead lines left: {len(re.findall(r'Superintendencia de Administradoras', text))}")
    elif sid == "ncg-sp-372":
        out.append(f"- expected parte [I, II] -> {stats['parte'].values}")
        out.append(f"- expected enmienda [A, B, C, C, D, E] (source repeats 'C.') -> {stats['enmienda'].values}")
        out.append(f"- expected 9 operaciones outside quotes (B:a-c, C:a-c, C:a, E:a-b) -> {stats['operacion'].matched} {stats['operacion'].values}; masked inside quoted Compendio text: {stats['operacion'].masked}")
        q_open, q_close = text.count("“"), text.count("”")
        out.append(f"- quote balance after normalize_quotes: open {q_open} / close {q_close}")
    elif sid == "cms-aid-14843":
        raw = doc.parsed.get("_raw", "")
        ref_re = r"(?:Of\.\s*Ord\.|NI)\s*N[°º]"
        raw_records = len(re.findall(r"^\s*\d{1,2}(?:\.\d{1,2}){1,2}\.?\s", raw, re.M))
        out.append(f"- reference markers: raw {len(re.findall(ref_re, raw))}, cleaned {len(re.findall(ref_re, text))}")
        out.append(f"- record lines (N.N / N.N.N) in raw text: {raw_records}; parsed records: {stats['record'].matched}")
    elif sid == "cms-aid-3590":
        out.append(f"- boilerplate tokens left: menu={'Quiénes Somos' in text}, glossary={'Agente de Venta' in text}, related={'Otras preguntas relacionadas' in text}, footer={'600-006-1013' in text}")
        out.append(f"- parsed keywords={doc.parsed.get('keywords')}, related_aids={doc.parsed.get('related_aids')}")
    return out


if __name__ == "__main__":
    main()
