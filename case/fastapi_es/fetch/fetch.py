"""Fetch the FastAPI Spanish docs at a pinned commit and write the case's samples and index.

Standard library only (needs `git`). Run it with:

    uv run python -I case/fastapi_es/fetch/fetch.py

Output (idempotent, the folders are rebuilt on every run):
    samples/<family>/<flat-name>.md   preprocessed Markdown (code includes inlined)
    samples/_manifest.json            id, url, family, path, content_type
    index/documents.jsonl             {"id", "doc_type"} rows; NO title on purpose (it comes from H1)
"""

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = "https://github.com/fastapi/fastapi.git"
COMMIT = "bb7f314287bf36e04539c572a2256d94ea45f0d6"  # tag 0.143.3
DOCS = Path("docs/es/docs")
SITE = "https://fastapi.tiangolo.com/es"
CASE_DIR = Path(__file__).resolve().parent.parent
GUIDE_PAGES = (
    "index", "async", "python-types", "features", "alternatives", "fastapi-cli",
    "history-design-future", "help-fastapi", "benchmarks", "editor-support",
    "environment-variables", "project-generation", "virtual-environments",
)  # fmt: skip
# `{* ../../docs_src/<path> hl[1:3] *}`. The `../../` prefix is the same at every depth: the docs
# build stages `docs/` and `docs_src/` side by side, so it always means <repo>/docs_src/.
INCLUDE_LINE = re.compile(r"^\{\*\s+(?:\.\./)+docs_src/(\S+)(?:\s+(.*?))?\s\*\}\s*$")
# `ln[1:9,29]`: 1-based inclusive line ranges (a bare number is one line). `hl[...]` only
# highlights, so it is ignored; other modifiers (`title[...]`) are cosmetic.
LN_MODIFIER = re.compile(r"\bln\[([^\]]*)\]")
LN_RANGE = re.compile(r"(\d+)(?::(\d+))?")
GAP_MARK = "# ..."
# shortcut: only whole-line includes are handled, there are no inline ones at the pinned commit;
# revisit if the commit is bumped (the unmatched count is printed).
ANY_INCLUDE = re.compile(r"^\s*\{\*")


def git(*args: str, cwd: Path) -> None:
    try:
        subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"git {' '.join(args)} failed:\n{exc.stderr}") from exc


def checkout(target: Path) -> None:
    git(
        "clone",
        "--quiet",
        "--filter=blob:none",
        "--no-checkout",
        REPO,
        str(target),
        cwd=target.parent,
    )
    git("sparse-checkout", "set", str(DOCS), "docs_src", "LICENSE", cwd=target)
    git("checkout", "--quiet", COMMIT, cwd=target)


def select_lines(code: str, ln: str) -> str | None:
    """The `ln[...]` lines of `code` (gaps marked), or None when a range is malformed/out of range."""
    lines = code.split("\n")
    parts: list[str] = []
    for item in ln.split(","):
        found = LN_RANGE.fullmatch(item.strip())
        if found is None:
            return None
        start = int(found.group(1))
        end = int(found.group(2) or start)
        if not 1 <= start <= end <= len(lines):
            return None
        if parts:
            parts.append(GAP_MARK)
        parts.extend(lines[start - 1 : end])
    return "\n".join(parts)


def inline_includes(text: str, repo: Path) -> tuple[str, int, int, int, int]:
    """Replace include lines by fenced python blocks.

    Returns (text, inlined, missing, unmatched, ln_honored). A path outside docs_src or a
    nonexistent file is `missing`; an include whose `ln[...]` cannot be applied is `unmatched`.
    """
    out: list[str] = []
    inlined = missing = unmatched = honored = 0
    root = (repo / "docs_src").resolve()
    for line in text.split("\n"):
        match = INCLUDE_LINE.match(line)
        if match is None:
            if ANY_INCLUDE.match(line):
                unmatched += 1
            out.append(line)
            continue
        source = (root / match.group(1)).resolve()
        if not source.is_relative_to(root) or not source.is_file():
            missing += 1  # the include line is dropped
            continue
        code = source.read_text(encoding="utf-8").rstrip("\n")
        ln = LN_MODIFIER.search(match.group(2) or "")
        if ln is not None:
            picked = select_lines(code, ln.group(1))
            if picked is None:
                unmatched += 1
                out.append(line)
                continue
            code = picked
            honored += 1
        out.append(f"```python\n{code}\n```")
        inlined += 1
    return "\n".join(out), inlined, missing, unmatched, honored


def selection(repo: Path) -> list[tuple[str, str, str, Path]]:
    """(id, family, flat file name, source) for every selected page, sorted."""
    docs = repo / DOCS
    pages: list[tuple[str, str, str, Path]] = []
    for path in sorted((docs / "tutorial").rglob("*.md")):
        rel = path.relative_to(docs / "tutorial").with_suffix("")
        pages.append(("fa-tut-" + "-".join(rel.parts), "tutorial", "__".join(rel.parts), path))
    for name in GUIDE_PAGES:
        pages.append((f"fa-guia-{name}", "guia", name, docs / f"{name}.md"))
    return pages


def page_url(path: Path, docs: Path) -> str:
    rel = path.relative_to(docs).with_suffix("")
    if rel.name == "index":  # mkdocs serves `x/index.md` at `x/`
        rel = rel.parent
    return f"{SITE}/" if rel.as_posix() == "." else f"{SITE}/{rel.as_posix()}/"


def main() -> int:
    samples, index_dir = CASE_DIR / "samples", CASE_DIR / "index"
    manifest: list[dict[str, str]] = []
    rows: list[dict[str, str]] = []
    totals = {"files": 0, "inlined": 0, "missing": 0, "unmatched": 0, "ln": 0, "bytes": 0}
    with tempfile.TemporaryDirectory(prefix="fastapi-docs-") as tmp:
        repo = Path(tmp) / "repo"
        checkout(repo)
        samples.mkdir(exist_ok=True)
        shutil.copyfile(repo / "LICENSE", samples / "LICENSE")  # MIT: keep the notice with copies
        for family in ("tutorial", "guia"):
            shutil.rmtree(samples / family, ignore_errors=True)
            (samples / family).mkdir(parents=True)
        for doc_id, family, flat, source in selection(repo):
            text, inlined, missing, unmatched, honored = inline_includes(
                source.read_text(encoding="utf-8"), repo
            )
            data = text.encode("utf-8")
            (samples / family / f"{flat}.md").write_bytes(data)
            manifest.append(
                {
                    "id": doc_id,
                    "url": page_url(source, repo / DOCS),
                    "family": family,
                    "path": f"{family}/{flat}.md",
                    "content_type": "text/markdown",
                }
            )
            rows.append({"id": doc_id, "doc_type": family})
            totals["files"] += 1
            totals["inlined"] += inlined
            totals["missing"] += missing
            totals["unmatched"] += unmatched
            totals["ln"] += honored
            totals["bytes"] += len(data)
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate ids")
    index_dir.mkdir(exist_ok=True)
    (samples / "_manifest.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (index_dir / "documents.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    print(
        f"files={totals['files']} includes_inlined={totals['inlined']} "
        f"ln_ranges_honored={totals['ln']} "
        f"includes_missing={totals['missing']} include_lines_unmatched={totals['unmatched']} "
        f"bytes={totals['bytes']}"
    )
    return 1 if totals["missing"] or totals["unmatched"] else 0  # both must be 0 at the pin


if __name__ == "__main__":
    sys.exit(main())
