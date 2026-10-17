"""HTML subtree -> ordered text/heading blocks -> `ParsedSection`s (no Markdown round trip).

Going through Markdown text would need escaping every paragraph that happens to start with
`#`, `-` or a fence; typed blocks avoid that and feed the same section model as the Markdown
parser: a heading stack names each section's `headings`.

The walk is iterative (explicit stack), so nesting depth never reaches the interpreter stack.
"""

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Final, Literal

from agintis.modules.knowledge.application import ParsedSection
from agintis.modules.knowledge.infrastructure.html.dom import HtmlElement, text_content

# Never content, anywhere (the block walk also ignores the document head and its title).
HARD_SKIP: Final = frozenset({"svg", "iframe", "head", "title"})
# Page furniture dropped when no explicit content selector says otherwise.
BOILERPLATE: Final = frozenset({"nav", "footer", "aside"})
_HEADING_TAG = re.compile(r"h([1-6])")
# Elements whose whole text is one block.
_LEAF_BLOCKS: Final = frozenset({"p", "li", "dt", "dd", "pre", "tr", "caption", "figcaption"})
# Inside a list item or definition, these make the item a container: its own text becomes a
# block and each nested item/row is a block of its own (instead of one flattened blob).
_NESTED_STRUCTURES: Final = frozenset({"ul", "ol", "dl", "table"})
_INLINE: Final = frozenset(
    {"a", "abbr", "b", "bdi", "bdo", "cite", "code", "data", "del", "dfn", "em", "i", "ins", "kbd",
     "mark", "q", "s", "samp", "small", "span", "strong", "sub", "sup", "time", "u", "var", "wbr",
     "label", "font"}
)  # fmt: skip
_WHITESPACE = re.compile(r"\s+")


def normalize_space(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


@dataclass(frozen=True)
class Block:
    kind: Literal["heading", "text"]
    text: str
    level: int = 0


@dataclass(frozen=True)
class BlockOptions:
    skip_tags: frozenset[str] = HARD_SKIP
    min_heading_level: int = 1
    heading_tags: frozenset[str] | None = None  # None = h1..h6
    bold_paragraph_as_heading: bool = False
    bold_heading_level: int = 2
    list_item_prefix: str = "- "


def _is_whole_bold(element: HtmlElement, text: str, skip: frozenset[str]) -> bool:
    bold = [e for e in element.iterdescendants("strong", "b")]
    bold_set = set(bold)
    outermost = [e for e in bold if not any(a in bold_set for a in e.iterancestors())]
    joined = normalize_space(" ".join(text_content(e, skip) for e in outermost))
    return bool(joined) and joined == text


def _is_container_item(element: HtmlElement) -> bool:
    """A `<li>`/`<dd>`/`<dt>` holding a nested list or table."""
    return element.tag in ("li", "dd", "dt") and any(
        d.tag in _NESTED_STRUCTURES for d in element.iterdescendants()
    )


class _Frame:
    """A container being walked: its remaining text/children and the inline text pending."""

    __slots__ = ("flow", "pending", "prefix")

    def __init__(self, element: HtmlElement, prefix: str) -> None:
        self.flow = _flow(element)
        self.pending: list[str] = []
        self.prefix = prefix


def _flow(element: HtmlElement) -> Iterator[str | HtmlElement]:
    """Leading text, then each child followed by its tail text, in document order."""
    if element.text:
        yield element.text
    for child in element:
        yield child
        if child.tail:
            yield child.tail


def collect_blocks(elements: Iterable[HtmlElement], options: BlockOptions) -> list[Block]:
    blocks: list[Block] = []
    skip = options.skip_tags

    def leaf(element: HtmlElement) -> None:
        if element.tag == "tr":
            cells = [
                normalize_space(text_content(c, skip))
                for c in element
                if isinstance(c.tag, str) and c.tag in ("td", "th")
            ]
            text = " | ".join(cell for cell in cells if cell)
        else:
            text = normalize_space(text_content(element, skip))
        if not text:
            return
        heading = _HEADING_TAG.fullmatch(element.tag)
        if heading is not None and (
            options.heading_tags is None or element.tag in options.heading_tags
        ):
            level = max(options.min_heading_level, int(heading.group(1)))
            blocks.append(Block("heading", text, level))
        elif (
            element.tag == "p"
            and options.bold_paragraph_as_heading
            and _is_whole_bold(element, text, skip)
        ):
            blocks.append(Block("heading", text, options.bold_heading_level))
        elif element.tag == "li":
            blocks.append(Block("text", options.list_item_prefix + text))
        else:
            blocks.append(Block("text", text))

    def flush(frame: _Frame) -> None:
        text = normalize_space("".join(frame.pending))
        frame.pending.clear()
        if text:
            blocks.append(Block("text", frame.prefix + text))

    def enter(element: HtmlElement, stack: list[_Frame]) -> None:
        tag = element.tag
        if not isinstance(tag, str) or tag in skip:
            return
        if _HEADING_TAG.fullmatch(tag) or (tag in _LEAF_BLOCKS and not _is_container_item(element)):
            leaf(element)
        else:
            prefix = options.list_item_prefix if tag == "li" else ""
            stack.append(_Frame(element, prefix))

    for root in elements:
        stack: list[_Frame] = []
        enter(root, stack)
        while stack:
            frame = stack[-1]
            item = next(frame.flow, None)
            if item is None:
                flush(frame)
                stack.pop()
            elif isinstance(item, str):
                frame.pending.append(item)
            elif not isinstance(item.tag, str):
                continue
            elif item.tag in _INLINE and item.tag not in skip:
                frame.pending.append(text_content(item, skip))
            else:
                flush(frame)
                enter(item, stack)
    return blocks


def blocks_to_sections(blocks: Iterable[Block]) -> tuple[str, tuple[ParsedSection, ...]]:
    """Group blocks under their heading path; returns (first level-1 heading or "", sections)."""
    sections: list[ParsedSection] = []
    path: list[tuple[int, str]] = []
    body: list[str] = []
    title = ""

    def flush() -> None:
        text = "\n\n".join(body).strip()
        body.clear()
        if text:
            sections.append(ParsedSection(text=text, headings=tuple(h for _, h in path)))

    for block in blocks:
        if block.kind == "heading":
            flush()
            while path and path[-1][0] >= block.level:
                path.pop()
            path.append((block.level, block.text))
            title = title or (block.text if block.level == 1 else "")
        else:
            body.append(block.text)
    flush()
    return title, tuple(sections)
