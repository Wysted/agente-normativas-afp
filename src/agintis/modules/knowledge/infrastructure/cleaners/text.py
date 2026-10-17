"""Line/character level cleaners that apply to every section."""

import re
from dataclasses import replace
from typing import ClassVar, Literal

from pydantic import Field, field_validator

from agintis.modules.knowledge.application import ParsedDocument, ParsedSection
from agintis.modules.knowledge.infrastructure.cleaners.base import (
    ParamsCleaner,
    Pattern,
    map_lines,
    map_texts,
    with_sections,
)
from agintis.modules.knowledge.profiles import Flag, ParamsModel, compile_profile_regex

_INNER_SPACES = re.compile("[ \t\u00a0]{2,}")
_LEFT_DOUBLE, _RIGHT_DOUBLE = "“", "”"


class StripLinesParams(ParamsModel):
    patterns: list[Pattern] = Field(min_length=1)


class StripLines(ParamsCleaner[StripLinesParams]):
    """Remove every line matching any pattern (page numbers, URLs, internal ids, rules)."""

    key: ClassVar[str] = "strip_lines"
    params_model = StripLinesParams

    def _apply(self, parsed: ParsedDocument, params: StripLinesParams) -> ParsedDocument:
        compiled = [compile_profile_regex(p) for p in params.patterns]
        return map_lines(
            parsed, lambda lines: [ln for ln in lines if not any(r.search(ln) for r in compiled)]
        )


class NormalizeWhitespaceParams(ParamsModel):
    collapse_inner_spaces: Flag = True
    strip_lines: Flag = True
    max_blank_lines: int = Field(default=1, ge=0)


class NormalizeWhitespace(ParamsCleaner[NormalizeWhitespaceParams]):
    """Trim lines, collapse runs of spaces/tabs/NBSP and cap consecutive blank lines."""

    key: ClassVar[str] = "normalize_whitespace"
    params_model = NormalizeWhitespaceParams

    def _apply(self, parsed: ParsedDocument, params: NormalizeWhitespaceParams) -> ParsedDocument:
        def clean(lines: list[str]) -> list[str]:
            if params.strip_lines:
                lines = [ln.strip() for ln in lines]
            if params.collapse_inner_spaces:
                lines = [_INNER_SPACES.sub(" ", ln) for ln in lines]
            out: list[str] = []
            blanks = 0
            for ln in lines:
                blanks = blanks + 1 if not ln.strip() else 0
                if blanks <= params.max_blank_lines:
                    out.append(ln)
            return out

        return map_texts(parsed, lambda text: "\n".join(clean(text.splitlines())).strip("\n"))


class NormalizeQuotesParams(ParamsModel):
    map: dict[str, str] = Field(default_factory=dict)
    ascii_double: Literal["keep", "alternate"] = "keep"

    @field_validator("map")
    @classmethod
    def _non_empty_keys(cls, value: dict[str, str]) -> dict[str, str]:
        if any(not key for key in value):
            raise ValueError("map keys must not be empty")
        return value


class NormalizeQuotes(ParamsCleaner[NormalizeQuotesParams]):
    """Replace quote characters via `map`; optionally turn ASCII `"` into curly pairs.

    `map` is applied as ONE simultaneous replacement (a single alternation, longest key first):
    the output of one entry is never rewritten by another, so the order of the entries cannot
    change the result.

    `alternate` alternates open/close starting with an opening quote and keeps its state per
    document, restarting only on a page change: all the sections of a pageless source (HTML,
    Markdown, text) form one run, so a quote opened in one section closes in a later one, while
    each PDF page starts fresh and an odd quote on one page cannot flip every page after it
    (nothing better is knowable for an unbalanced page).
    """

    key: ClassVar[str] = "normalize_quotes"
    version: ClassVar[int] = 2
    params_model = NormalizeQuotesParams

    def _apply(self, parsed: ParsedDocument, params: NormalizeQuotesParams) -> ParsedDocument:
        mapping = params.map
        pattern = (
            re.compile("|".join(re.escape(k) for k in sorted(mapping, key=lambda k: (-len(k), k))))
            if mapping
            else None
        )
        sections: list[ParsedSection] = []
        opened = False  # parity of ASCII quotes seen in the current run
        previous_page: int | None = None
        for index, section in enumerate(parsed.sections):
            text = section.text
            if pattern is not None:
                text = pattern.sub(lambda m: mapping[m.group()], text)
            if params.ascii_double == "alternate":
                if index and section.page != previous_page:
                    opened = False
                text, opened = _alternate_quotes(text, opened)
            previous_page = section.page
            sections.append(replace(section, text=text))
        return with_sections(parsed, sections)


def _alternate_quotes(text: str, opened: bool) -> tuple[str, bool]:
    """Replace each ASCII `"` by an opening or closing curly quote; returns the new state."""
    parts = text.split('"')
    out = [parts[0]]
    for part in parts[1:]:
        out.append((_RIGHT_DOUBLE if opened else _LEFT_DOUBLE) + part)
        opened = not opened
    return "".join(out), opened


class JoinContinuationLinesParams(ParamsModel):
    continuation: Pattern
    record_start: Pattern | None = None


class JoinContinuationLines(ParamsCleaner[JoinContinuationLinesParams]):
    """Append wrapped lines to the previous line (records that span several lines).

    A line matching `continuation` is joined to the previous one with a single space, unless it
    also matches `record_start`, which always begins a new line. Blank lines are dropped.
    """

    key: ClassVar[str] = "join_continuation_lines"
    params_model = JoinContinuationLinesParams

    def _apply(self, parsed: ParsedDocument, params: JoinContinuationLinesParams) -> ParsedDocument:
        continuation = compile_profile_regex(params.continuation)
        start = compile_profile_regex(params.record_start) if params.record_start else None

        def join(lines: list[str]) -> list[str]:
            out: list[str] = []
            for raw in lines:
                line = raw.strip()
                if not line:
                    continue
                is_continuation = continuation.match(line) and not (start and start.match(line))
                if out and is_continuation:
                    out[-1] = f"{out[-1]} {line}"
                else:
                    out.append(line)
            return out

        return map_lines(parsed, join)


class DedupeHeadingEchoParams(ParamsModel):
    pattern: Pattern
    key_normalize: Literal["none", "strip_dots_spaces"] = "none"
    window_lines: int = Field(default=4, ge=1)

    @field_validator("pattern")
    @classmethod
    def _needs_key_group(cls, value: str) -> str:
        if "key" not in re.compile(value).groupindex:
            raise ValueError("pattern must define a named group (?P<key>...)")
        return value


class DedupeHeadingEcho(ParamsCleaner[DedupeHeadingEchoParams]):
    """Drop a heading printed twice (short marker, then the full title).

    When a line matches `pattern` and one of the next `window_lines` lines matches it with the
    same normalised `key`, the first line and everything between them (a marker wrapped over
    several lines) are removed; the later, complete occurrence stays.
    """

    key: ClassVar[str] = "dedupe_heading_echo"
    params_model = DedupeHeadingEchoParams

    def _apply(self, parsed: ParsedDocument, params: DedupeHeadingEchoParams) -> ParsedDocument:
        pattern = compile_profile_regex(params.pattern)

        def normalize(value: str) -> str:
            value = value.lower()
            if params.key_normalize == "strip_dots_spaces":
                return re.sub(r"[\s.]+", "", value)
            return re.sub(r"\s+", " ", value)

        def dedupe(lines: list[str]) -> list[str]:
            drop: set[int] = set()
            i = 0
            while i < len(lines):
                first = pattern.match(lines[i].strip())
                if first:
                    key = normalize(first.group("key"))
                    for j in range(i + 1, min(len(lines), i + 1 + params.window_lines)):
                        again = pattern.match(lines[j].strip())
                        if again and normalize(again.group("key")) == key:
                            drop.update(range(i, j))
                            i = j
                            break
                i += 1
            return [ln for index, ln in enumerate(lines) if index not in drop]

        return map_lines(parsed, dedupe)
