"""Cleaners that reason about PDF pages (sections carrying a page number)."""

import collections
import logging
import re
from dataclasses import replace
from typing import ClassVar, Literal

from pydantic import Field

from agintis.modules.knowledge.application import ParsedDocument
from agintis.modules.knowledge.infrastructure.cleaners.base import (
    ParamsCleaner,
    Pattern,
    nonblank_indexes,
    paged_indexes,
    with_sections,
)
from agintis.modules.knowledge.profiles import Flag, ParamsModel, compile_profile_regex

logger = logging.getLogger(__name__)

# A "furniture" key made only of digits ("#" after digit normalisation) is a page number: that
# is `strip_lines` territory, never treated as a repeated header.
_DIGITS_ONLY_KEY = "#"


class DropTocPagesParams(ParamsModel):
    leader_pattern: Pattern
    min_leader_ratio: float = Field(default=0.2, gt=0, le=1)
    max_scan_pages: int = Field(default=15, ge=1)


class DropTocPages(ParamsCleaner[DropTocPagesParams]):
    """Drop the leading run of table-of-contents pages (lines with dotted leaders).

    Scans at most `max_scan_pages` pages from the start and stops at the first page whose share
    of matching non-blank lines is below `min_leader_ratio`. Never drops every page.
    """

    key: ClassVar[str] = "drop_toc_pages"
    params_model = DropTocPagesParams

    def _apply(self, parsed: ParsedDocument, params: DropTocPagesParams) -> ParsedDocument:
        leader = compile_profile_regex(params.leader_pattern)
        indexes = paged_indexes(parsed)
        dropped: list[int] = []
        for index in indexes[: params.max_scan_pages]:
            lines = [ln for ln in parsed.sections[index].text.splitlines() if ln.strip()]
            ratio = sum(1 for ln in lines if leader.search(ln)) / max(1, len(lines))
            if ratio < params.min_leader_ratio:
                break
            dropped.append(index)
        if len(dropped) == len(indexes):
            return parsed  # a document made only of "TOC" pages is not a TOC
        gone = set(dropped)
        return with_sections(parsed, [s for i, s in enumerate(parsed.sections) if i not in gone])


class StripRunningHeadersParams(ParamsModel):
    patterns: list[Pattern] = Field(min_length=1)
    zone: Literal["top", "bottom", "any"] = "top"
    max_lines: int = Field(default=2, ge=1)


class StripRunningHeaders(ParamsCleaner[StripRunningHeadersParams]):
    """Remove known running headers/footers: lines matching a pattern within the page zone."""

    key: ClassVar[str] = "strip_running_headers"
    params_model = StripRunningHeadersParams

    def _apply(self, parsed: ParsedDocument, params: StripRunningHeadersParams) -> ParsedDocument:
        patterns = [compile_profile_regex(p) for p in params.patterns]
        sections = list(parsed.sections)
        for index in paged_indexes(parsed):
            lines = sections[index].text.splitlines()
            nonblank = nonblank_indexes(lines)
            zone: set[int] = set()
            if params.zone in ("top", "any"):
                zone.update(nonblank[: params.max_lines])
            if params.zone in ("bottom", "any"):
                zone.update(nonblank[-params.max_lines :])
            drop = {i for i in zone if any(p.search(lines[i]) for p in patterns)}
            kept = "\n".join(ln for i, ln in enumerate(lines) if i not in drop)
            sections[index] = replace(sections[index], text=kept)
        return with_sections(parsed, sections)


class StripRepeatedPageLinesParams(ParamsModel):
    zone_lines_top: int = Field(default=3, ge=0)
    zone_lines_bottom: int = Field(default=3, ge=0)
    min_page_ratio: float = Field(default=0.5, gt=0, le=1)
    # Documents with fewer pages are left untouched: with 1-2 pages "repeated" proves nothing.
    min_pages: int = Field(default=3, ge=2)
    # Off by default: with it, "Article 5" / "Article 6" at the top of many pages look like
    # one repeated line. Turn it on for "Page 3 of 9" style footers.
    normalize_digits: Flag = False


def _normalize_line(line: str, digits: bool) -> str:
    text = re.sub(r"\s+", " ", line.strip()).lower()
    return re.sub(r"\d+", "#", text) if digits else text


class StripRepeatedPageLines(ParamsCleaner[StripRepeatedPageLinesParams]):
    """Detect letterhead/footer lines that repeat across pages and remove them.

    Only lines in the first `zone_lines_top` / last `zone_lines_bottom` non-blank lines of a page
    count. A (whitespace/case/digit-normalised) line is furniture when it shows up in the zone of
    at least `max(2, min_page_ratio * pages)` different pages. Guards against deleting real
    content on short documents: nothing happens below `min_pages` pages (default 3), and a line
    must repeat on at least two pages even when the ratio asks for less.
    """

    key: ClassVar[str] = "strip_repeated_page_lines"
    params_model = StripRepeatedPageLinesParams

    def _apply(
        self, parsed: ParsedDocument, params: StripRepeatedPageLinesParams
    ) -> ParsedDocument:
        indexes = paged_indexes(parsed)
        if len(indexes) < params.min_pages:
            return parsed
        digits = params.normalize_digits
        zones: dict[int, tuple[list[str], set[int]]] = {}
        counts: collections.Counter[str] = collections.Counter()
        for index in indexes:
            lines = parsed.sections[index].text.splitlines()
            nonblank = nonblank_indexes(lines)
            zone = set(nonblank[: params.zone_lines_top])
            if params.zone_lines_bottom:
                zone.update(nonblank[-params.zone_lines_bottom :])
            zones[index] = (lines, zone)
            counts.update({_normalize_line(lines[i], digits) for i in zone})
        threshold = max(2, params.min_page_ratio * len(indexes))
        furniture = {k for k, n in counts.items() if n >= threshold and k != _DIGITS_ONLY_KEY}
        if not furniture:
            return parsed
        sections = list(parsed.sections)
        for index, (lines, zone) in zones.items():
            drop = {i for i in zone if _normalize_line(lines[i], digits) in furniture}
            kept = "\n".join(ln for i, ln in enumerate(lines) if i not in drop)
            sections[index] = replace(sections[index], text=kept)
        return with_sections(parsed, sections)


class DropPagesBeforeParams(ParamsModel):
    pattern: Pattern
    keep_matching_page: Flag = True


class DropPagesBefore(ParamsCleaner[DropPagesBeforeParams]):
    """Drop the pages before the first one containing a line matching `pattern` (cover, prologue).

    The pattern is tested line by line. When no page matches, nothing is dropped.
    """

    key: ClassVar[str] = "drop_pages_before"
    params_model = DropPagesBeforeParams

    def _apply(self, parsed: ParsedDocument, params: DropPagesBeforeParams) -> ParsedDocument:
        pattern = compile_profile_regex(params.pattern)
        indexes = paged_indexes(parsed)
        match = next(
            (
                i
                for i in indexes
                if any(pattern.search(ln) for ln in parsed.sections[i].text.splitlines())
            ),
            None,
        )
        if match is None:
            logger.warning(
                "drop_pages_before: pattern %r not found, kept every page", pattern.pattern
            )
            return parsed
        cut = match if params.keep_matching_page else match + 1
        gone = {i for i in indexes if i < cut}
        return with_sections(parsed, [s for i, s in enumerate(parsed.sections) if i not in gone])
