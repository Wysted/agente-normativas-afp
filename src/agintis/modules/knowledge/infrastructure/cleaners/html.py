"""Cleaners for HTML sources (`ParsedDocument.source_html`, produced by the simple parser).

They work on the DOM, not on text: `html_main_content` keeps only the CSS-selected content
(menus, glossaries, footers fall away) and `html_collect` extracts document-level values that
chunking/metadata can use later (`ParsedDocument.collected`). Selectors are validated when the
profile loads.
"""

import logging
import re
from dataclasses import replace
from typing import ClassVar, Final, Literal, Self

from pydantic import Field, field_validator, model_validator

from agintis.modules.knowledge.application import (
    CollectedValue,
    ParsedDocument,
    UnsupportedDocumentTypeError,
)
from agintis.modules.knowledge.infrastructure.cleaners.base import ParamsCleaner, Pattern
from agintis.modules.knowledge.infrastructure.html import (
    HARD_SKIP,
    Block,
    BlockOptions,
    blocks_to_sections,
    collect_blocks,
    drop_subtrees,
    has_ancestor_in,
    normalize_space,
    parse_html_tree,
    parse_selector,
    select_all,
    text_content,
)
from agintis.modules.knowledge.infrastructure.text import strip_unstorable
from agintis.modules.knowledge.profiles import Flag, ParamsModel, compile_profile_regex

logger = logging.getLogger(__name__)

_HEADING_TAG = re.compile(r"h[1-6]")
# `int()` of a longer digit string is legal but no longer a sane identifier/metadata value.
MAX_COLLECTED_INT_DIGITS: Final = 18


def _selector(value: str) -> str:
    parse_selector(value)  # ValueError on unsupported syntax
    return value


def _require_html(parsed: ParsedDocument, key: str) -> str:
    if parsed.source_html is None:
        raise UnsupportedDocumentTypeError(f"Cleaner {key!r} needs an HTML source document")
    return parsed.source_html


class HtmlMainContentParams(ParamsModel):
    title_selector: str | None = None
    include_selectors: list[str] = Field(default_factory=lambda: ["body"], min_length=1)
    exclude_selectors: list[str] = Field(default_factory=list)
    bold_paragraph_as_heading: Flag = False
    heading_selectors: list[str] | None = None
    list_item_prefix: str = "- "
    output: Literal["markdown"] = "markdown"

    @field_validator("title_selector")
    @classmethod
    def _title_selector(cls, value: str | None) -> str | None:
        return None if value is None else _selector(value)

    @field_validator("include_selectors", "exclude_selectors")
    @classmethod
    def _selectors(cls, value: list[str]) -> list[str]:
        return [_selector(item) for item in value]

    @field_validator("heading_selectors")
    @classmethod
    def _heading_tags(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and any(not _HEADING_TAG.fullmatch(tag) for tag in value):
            raise ValueError("heading_selectors must be heading tag names (h1..h6)")
        return value


class HtmlMainContent(ParamsCleaner[HtmlMainContentParams]):
    """Rebuild the sections from the selected part of the page.

    * The title (first match of `title_selector`, else the parsed title) becomes the level-1
      heading; headings inside the content keep their natural level (h3 nests under the
      previous h2), never above level 2.
    * `exclude_selectors` drop whole subtrees (scripts, hidden boxes, buttons...) BEFORE
      anything else is selected: nothing inside an excluded subtree can match an include or
      title selector or contribute text.
    * A heading inside the content whose text equals the title is skipped (no "T > T" path).
    * `heading_selectors` limits which `h1..h6` tags are headings (default all);
      `bold_paragraph_as_heading` turns a paragraph made only of bold text into a level-2 heading.
    * `<li>` items become `list_item_prefix` + text (a nested list inside an item makes each
      nested item its own block); table rows become `cell | cell`; block boundaries inside a
      block are separated by a space.
    Selectors matching nested nodes are deduplicated (the outermost wins).
    """

    key: ClassVar[str] = "html_main_content"
    version: ClassVar[int] = 2
    rebuilds_text: ClassVar[bool] = True
    params_model = HtmlMainContentParams

    def _apply(self, parsed: ParsedDocument, params: HtmlMainContentParams) -> ParsedDocument:
        root = parse_html_tree(_require_html(parsed, self.key))
        drop_subtrees(select_all(root, [parse_selector(s) for s in params.exclude_selectors]))
        title = parsed.title
        if params.title_selector:
            found = select_all(root, [parse_selector(params.title_selector)])
            if found:
                title = normalize_space(text_content(found[0], HARD_SKIP)) or title
        matched = select_all(root, [parse_selector(s) for s in params.include_selectors])
        matched_set = set(matched)
        outermost = [e for e in matched if not has_ancestor_in(e, matched_set)]
        options = BlockOptions(
            skip_tags=HARD_SKIP,
            min_heading_level=2,
            heading_tags=frozenset(params.heading_selectors)
            if params.heading_selectors is not None
            else None,
            bold_paragraph_as_heading=params.bold_paragraph_as_heading,
            list_item_prefix=params.list_item_prefix,
        )
        title = title.strip()
        heading = [Block("heading", title, 1)] if title else []
        blocks = [
            block
            for block in collect_blocks(outermost, options)
            if not (block.kind == "heading" and title and block.text == normalize_space(title))
        ]
        _, sections = blocks_to_sections([*heading, *blocks])
        if not outermost:
            logger.warning("html_main_content: no include selector matched in the document")
        return replace(
            parsed,
            title=strip_unstorable(title),
            sections=tuple(replace(s, text=strip_unstorable(s.text)) for s in sections),
        )


class CollectFieldParams(ParamsModel):
    selector: str
    attr: str | None = None
    regex: Pattern | None = None
    split: str | None = Field(default=None, min_length=1)

    @field_validator("selector")
    @classmethod
    def _valid_selector(cls, value: str) -> str:
        return _selector(value)

    @model_validator(mode="after")
    def _regex_or_split(self) -> Self:
        if self.regex is not None and self.split is not None:
            raise ValueError("regex and split are mutually exclusive")
        if self.regex is not None and re.compile(self.regex).groups < 1:
            raise ValueError("regex must have a capture group")
        return self


class HtmlCollectParams(ParamsModel):
    fields: dict[str, CollectFieldParams] = Field(min_length=1)

    @field_validator("fields")
    @classmethod
    def _names(cls, value: dict[str, CollectFieldParams]) -> dict[str, CollectFieldParams]:
        if any(not re.fullmatch(r"[a-z][a-z0-9_]*", name) for name in value):
            raise ValueError("field names must match [a-z][a-z0-9_]*")
        return value


class HtmlCollect(ParamsCleaner[HtmlCollectParams]):
    """Extract named lists of values from the page into `ParsedDocument.collected`.

    For each matched node the raw value is the `attr` attribute or its text; then `regex` keeps
    the first capture group (ASCII digit-only captures of up to 18 digits become ints, longer
    or non-ASCII ones stay strings), `split` splits and trims, and without either the trimmed
    raw value is kept. A raw value over `MAX_REGEX_INPUT_CHARS` is not matched by `regex`.
    Reads the whole DOM, including content that `html_main_content` removes from the text
    (hidden keyword boxes, related links).
    """

    key: ClassVar[str] = "html_collect"
    version: ClassVar[int] = 2
    edits_text: ClassVar[bool] = False
    params_model = HtmlCollectParams

    def _apply(self, parsed: ParsedDocument, params: HtmlCollectParams) -> ParsedDocument:
        root = parse_html_tree(_require_html(parsed, self.key))
        collected = dict(parsed.collected)
        for name, spec in params.fields.items():
            regex = compile_profile_regex(spec.regex) if spec.regex is not None else None
            values: list[CollectedValue] = []
            for element in select_all(root, [parse_selector(spec.selector)]):
                raw = strip_unstorable(
                    element.get(spec.attr, "") if spec.attr else text_content(element)
                )
                if regex is not None:
                    match = regex.search(raw)
                    if match and match.group(1) is not None:
                        values.append(_collected_scalar(match.group(1)))
                elif spec.split is not None:
                    values.extend(v.strip() for v in raw.split(spec.split) if v.strip())
                elif raw.strip():
                    values.append(raw.strip())
            collected[name] = tuple(values)
        return replace(parsed, collected=collected)


def _collected_scalar(captured: str) -> CollectedValue:
    """Short ASCII digit strings become ints; anything else (long, unicode digits) stays text."""
    if captured.isascii() and captured.isdecimal() and len(captured) <= MAX_COLLECTED_INT_DIGITS:
        return int(captured)
    return captured
