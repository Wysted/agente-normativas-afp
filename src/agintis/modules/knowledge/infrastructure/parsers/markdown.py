"""Markdown and plain text: sections split at headings.

Supported (CommonMark subset): ATX headings (`#`..`######`, optional closing `#` sequence that
must be preceded by a space, so `# C#` keeps its `#`), setext headings (`===` / `---` under a
paragraph), fenced code blocks (a fence closes only with the same character and at least the
same length), and a leading YAML front matter block (skipped). A trailing attribute block of a
heading (Pandoc / attr_list: `# Title { #id .class key=val }`, kramdown `{: #id}`) is not part
of its text.
Not supported: indented code blocks, HTML blocks.
"""

import re

from agintis.modules.knowledge.application import ParsedDocument, ParsedSection
from agintis.modules.knowledge.infrastructure.parsers._title import title_from_uri
from agintis.modules.knowledge.infrastructure.parsers.limits import ParseContext
from agintis.modules.knowledge.infrastructure.text import decode_text

_ATX = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*))?[ \t]*$")
# Body of a trailing `{ #id }`, `{.class}`, `{key=val}` (`{:` is kramdown); plain braces such as
# `# Use {x}` stay. Applied only to the text after the last `{`, so it stays linear.
_ATTR_BODY = re.compile(r":?[ \t]*(?:#[^\s{}]|\.[^\W\d]|[A-Za-z_][\w:.-]*=[^\s{}])[^{}]*")
_SETEXT = re.compile(r"^ {0,3}(=+|-+)[ \t]*$")
_FENCE_OPEN = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_FRONT_MATTER_FENCE = "---"
_FRONT_MATTER_END = ("---", "...")
_SETEXT_LEVELS = {"=": 1, "-": 2}


def _skip_front_matter(lines: list[str]) -> list[str]:
    if not lines or lines[0].rstrip() != _FRONT_MATTER_FENCE:
        return lines
    for index in range(1, len(lines)):
        if lines[index].rstrip() in _FRONT_MATTER_END:
            return lines[index + 1 :]
    return lines  # never closed: it is just a thematic break followed by text


def _strip_closing(text: str) -> str:
    """Drop an ATX closing `#` run: it must follow a space/tab or be the whole content."""
    trimmed = text.rstrip(" \t")
    body = trimmed.rstrip("#")
    if body == trimmed:
        return trimmed
    return body.rstrip(" \t") if not body or body[-1] in " \t" else trimmed


def _strip_attrs(text: str) -> str:
    """Drop a trailing attribute block. rfind + fullmatch keep it linear on hostile input."""
    trimmed = text.rstrip(" \t")
    start = trimmed.rfind("{")
    if (
        start < 0
        or not trimmed.endswith("}")
        or not _ATTR_BODY.fullmatch(trimmed, start + 1, len(trimmed) - 1)
    ):
        return trimmed
    return trimmed[:start].rstrip(" \t")


def _atx_heading(line: str) -> tuple[int, str] | None:
    match = _ATX.match(line)
    if match is None:
        return None
    text = _strip_closing(match.group(2) or "")
    # Attributes may follow the closing run (`## T ##  {#id}`): a run left behind them is closing.
    stripped = _strip_attrs(text)
    text = (_strip_closing(stripped) if stripped != text else text).strip()
    return (len(match.group(1)), text) if text else None


def parse_markdown(content: bytes, uri: str, context: ParseContext) -> ParsedDocument:
    sections: list[ParsedSection] = []
    path: list[tuple[int, str]] = []  # (level, heading) stack
    body: list[str] = []
    title = ""
    fence: str | None = None  # the opening run, e.g. "```" or "~~~~"

    def flush() -> None:
        text = "\n".join(body).strip()
        body.clear()
        if text:
            sections.append(ParsedSection(text=text, headings=tuple(h for _, h in path)))

    def open_heading(level: int, heading: str) -> None:
        nonlocal title
        flush()
        while path and path[-1][0] >= level:
            path.pop()
        path.append((level, heading))
        title = title or (heading if level == 1 else "")

    def pop_paragraph() -> str:
        """Remove and return the paragraph (trailing non-blank lines) right above a setext line."""
        lines: list[str] = []
        while body and body[-1].strip():
            lines.append(body.pop().strip())
        return " ".join(reversed(lines))

    for line in _skip_front_matter(decode_text(content, context.charset).split("\n")):
        if fence is not None:
            body.append(line)
            stripped = line.strip()
            if stripped.startswith(fence[0] * len(fence)) and set(stripped) == {fence[0]}:
                fence = None
            continue
        opened = _FENCE_OPEN.match(line)
        if opened:
            fence = opened.group(1)
            body.append(line)
            continue
        atx = _atx_heading(line)
        if atx is not None:
            open_heading(*atx)
            continue
        setext = _SETEXT.match(line)
        if setext and body and body[-1].strip() and not _FENCE_OPEN.match(body[-1]):
            paragraph = pop_paragraph()
            heading = _strip_attrs(paragraph).strip() or paragraph
            open_heading(_SETEXT_LEVELS[setext.group(1)[0]], heading)
            continue
        body.append(line)
    flush()
    return ParsedDocument(title=title or title_from_uri(uri), sections=tuple(sections))


def parse_text(content: bytes, uri: str, context: ParseContext) -> ParsedDocument:
    text = decode_text(content, context.charset).strip()
    sections = (ParsedSection(text=text),) if text else ()
    return ParsedDocument(title=title_from_uri(uri), sections=sections)
