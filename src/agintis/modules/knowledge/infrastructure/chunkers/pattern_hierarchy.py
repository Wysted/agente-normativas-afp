"""Structural chunker driven by regex levels (key "pattern_hierarchy").

For documents whose structure is typographic, not markup: numbered items, "Chapter X" headings,
lettered amendments. The profile declares a hierarchy of `levels`; each is a regex applied to
ONE LINE at a time (never `re.M` over the whole text: `[^.;]*` or `\\s*$` would swallow the
next heading). A matching line opens a unit at its `depth`; a deeper match nests inside it, a
match at the same or a shallower depth closes it. `split_level` picks the depth at which
chunks are cut: everything from one split-level (or shallower) heading to the next is one
chunk. The chunker works over the JOINED text of all sections, so a unit may cross pages; each
chunk records `page` / `page_end` through the offset map.

Per chunk
* Text = optional heading prefix (`heading_prefix_template`, rendered with the document
  metadata, `title` and the chunk's groups, plus `{path}`: the `heading_template` of every open
  level joined with " > ") + the unit body. The prefix counts against the token budget.
* `groups`: the named regex groups of the open levels (one value per merged unit); profile
  metadata reads them as `group:<name>`. `heading` metadata = the path; with `parent_level`
  the heading of that open level is also stored as `parent_heading` (plain metadata, no
  parent-child structure).

Structure options
* `role: context` + `reset_on_change`: a level that only SETS context (a breadcrumb repeated on
  every page). A repeat with the same values for the `reset_on_change` groups is ignored (it
  must not close the open children); `keep_heading_line: false` cuts the line out of the text.
* `ignore_inside`: spans (typographic quotes, `nesting: true` to count depth) whose lines are
  never headings: amendment texts quote the document they modify, with its own numbering. Spans
  of several specs are merged into disjoint intervals; an opener with no closer within
  `max_span_chars` is abandoned, so a stray quote never disables masking for the rest of the
  document.
* `preamble: keep|drop`: text before the first heading. `preamble_heading` is its path (no
  default: absent means no heading line for it).
* `fold_intro_below_tokens`: a unit shorter than this whose next unit is deeper (a chapter
  title and one line) is prepended to that next unit instead of standing alone.
* `merge_below_tokens`: a unit shorter than this is merged into the NEXT unit when both have
  the same level and parent and the result fits the budget NET of the first unit's heading
  prefix; the merged chunk keeps the first unit's heading and `groups` lists the values of
  every merged unit in order.
* `oversize`: a unit over budget is cut at the `levels` below the split level, then at lines
  matching `extra_split_patterns` (not masked by `ignore_inside`), the pieces are re-packed,
  and any piece still too large goes through `BudgetSplitter` (paragraphs, sentences, words,
  tokens). Each piece records its own `page` / `page_end` (through the offsets of the source
  text) and the `groups` of the merged units it overlaps.
"""

import asyncio
import bisect
import itertools
import re
from array import array
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final, Literal, Self

from pydantic import Field, field_validator, model_validator

from agintis.modules.knowledge.application import ChunkDraft, Chunker, JoinedText, ParsedDocument
from agintis.modules.knowledge.domain import MetadataValue, RagProfile
from agintis.modules.knowledge.infrastructure.chunkers.splitting import (
    MAX_HEADING_BUDGET_SHARE,
    MIN_CHUNK_MAX_TOKENS,
    BudgetSplitter,
    Span,
    chunk_budget,
)
from agintis.modules.knowledge.infrastructure.chunkers.tokens import TokenCounter
from agintis.modules.knowledge.infrastructure.templates import (
    check_template_syntax,
    render_template,
    template_names,
)
from agintis.modules.knowledge.profiles import (
    Flag,
    ParamsModel,
    Pattern,
    ProfileRegex,
    compile_profile_regex,
    validate_params,
)

PATTERN_HIERARCHY_CHUNKER_KEY: Final = "pattern_hierarchy"
HEADING_SEPARATOR: Final = " > "
_NAME: Final = re.compile(r"[a-z][a-z0-9_]*")
_BLANK_RUNS: Final = re.compile(r"\n{3,}")


class LevelParams(ParamsModel):
    name: str
    depth: int | None = Field(default=None, ge=0)
    pattern: Pattern
    heading_template: str | None = None
    role: Literal["structure", "context"] = "structure"
    reset_on_change: list[str] = Field(default_factory=list)
    keep_heading_line: Flag = True

    @field_validator("name")
    @classmethod
    def _identifier(cls, value: str) -> str:
        if not _NAME.fullmatch(value):
            raise ValueError("level names must match [a-z][a-z0-9_]*")
        return value

    @model_validator(mode="after")
    def _template_reads_own_groups(self) -> Self:
        if self.heading_template is None:
            return self
        check_template_syntax(self.heading_template)
        groups = re.compile(self.pattern).groupindex
        if unknown := sorted(template_names(self.heading_template) - set(groups)):
            raise ValueError(
                f"level {self.name!r}: heading_template reads {unknown}, which are not named "
                "groups of the level's pattern"
            )
        return self

    @model_validator(mode="after")
    def _reset_keys_are_groups(self) -> Self:
        groups = re.compile(self.pattern).groupindex
        if unknown := [key for key in self.reset_on_change if key not in groups]:
            raise ValueError(f"level {self.name!r}: reset_on_change names unknown groups {unknown}")
        if self.reset_on_change and self.role != "context":
            raise ValueError(f"level {self.name!r}: reset_on_change only applies to role: context")
        return self


class QuoteSpanParams(ParamsModel):
    open: str = Field(min_length=1, max_length=1)
    close: str = Field(min_length=1, max_length=1)
    nesting: Flag = False
    max_span_chars: int = Field(default=60_000, ge=1)

    @model_validator(mode="after")
    def _distinct(self) -> Self:
        if self.nesting and self.open == self.close:
            raise ValueError("nesting spans need different open and close characters")
        return self


class OversizeParams(ParamsModel):
    strategy: Literal["deeper_levels_then_recursive"] = "deeper_levels_then_recursive"
    levels: list[str] = Field(default_factory=list)
    extra_split_patterns: list[Pattern] = Field(default_factory=list)


class PatternHierarchyParams(ParamsModel):
    levels: list[LevelParams] = Field(min_length=1)
    split_level: str
    oversize: OversizeParams = Field(default_factory=OversizeParams)
    ignore_inside: list[QuoteSpanParams] = Field(default_factory=list)
    merge_below_tokens: int = Field(default=0, ge=0)
    fold_intro_below_tokens: int = Field(default=0, ge=0)
    include_heading: Flag = True
    heading_prefix_template: str = "{path}"
    preamble: Literal["keep", "drop"] = "keep"
    preamble_heading: str | None = None
    parent_level: str | None = None

    @field_validator("heading_prefix_template")
    @classmethod
    def _prefix_template_syntax(cls, value: str) -> str:
        # Its variables are the document metadata (known only at ingest), `title`, `path` and
        # the groups: only the syntax can be checked at boot.
        return check_template_syntax(value)

    @model_validator(mode="after")
    def _levels_are_consistent(self) -> Self:
        names = [level.name for level in self.levels]
        if len(set(names)) != len(names):
            raise ValueError("level names must be unique")
        depths = self.depths()
        if self.split_level not in depths:
            raise ValueError(f"split_level {self.split_level!r} is not a declared level")
        for name in self.oversize.levels:
            if name not in depths:
                raise ValueError(f"oversize.levels names unknown level {name!r}")
            if depths[name] <= depths[self.split_level]:
                raise ValueError(f"oversize level {name!r} must be deeper than the split level")
        if self.parent_level is not None:
            if self.parent_level not in depths:
                raise ValueError(f"parent_level {self.parent_level!r} is not a declared level")
            if depths[self.parent_level] >= depths[self.split_level]:
                raise ValueError("parent_level must be shallower than the split level")
        return self

    def depths(self) -> dict[str, int]:
        return {
            level.name: level.depth if level.depth is not None else index
            for index, level in enumerate(self.levels)
        }

    def group_names(self) -> frozenset[str]:
        """Every named regex group of every level (what `group:<name>` sources may read)."""
        return frozenset(
            name for level in self.levels for name in re.compile(level.pattern).groupindex
        )


@dataclass(frozen=True)
class _Level:
    name: str
    depth: int
    order: int
    regex: ProfileRegex
    template: str
    context: bool
    reset_on_change: tuple[str, ...]
    keep_line: bool


type _Groups = dict[str, str]
type _Frame = tuple[int, _Level, _Groups]  # (depth, level, groups) of one open level


@dataclass(frozen=True)
class _Match:
    start: int
    line_end: int
    level: _Level
    groups: _Groups


@dataclass
class _Unit:
    start: int
    end: int
    frames: tuple[_Frame, ...]  # open levels, shallowest first; empty for the preamble
    preamble: bool
    body: str
    members: list[_Groups] = field(default_factory=list)  # groups of every merged unit
    member_starts: list[int] = field(default_factory=list)  # where each member begins

    @property
    def depth(self) -> int:
        return self.frames[-1][0] if self.frames else -1

    @property
    def own(self) -> str:
        return self.frames[-1][1].name if self.frames else ""


def _quote_spans(text: str, specs: Sequence[QuoteSpanParams]) -> list[tuple[int, int]]:
    """Disjoint, sorted `(open, close + 1)` intervals covering every span of every spec.

    Spans of different specs may overlap or interleave; they are merged so that a binary search
    over the interval starts is exact. Unclosed or over-long spans are skipped.
    """
    spans: list[tuple[int, int]] = []
    for spec in specs:
        spans.extend(_nesting_spans(text, spec) if spec.nesting else _flat_spans(text, spec))
    return _union(spans)


def _union(spans: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _flat_spans(text: str, spec: QuoteSpanParams) -> list[tuple[int, int]]:
    """Non-nesting spans, one pass: the closer found for an opener is reused by the openers
    before it (rescanning for every stray opener would be quadratic)."""
    spans: list[tuple[int, int]] = []
    position, closing = 0, -1
    while (opening := text.find(spec.open, position)) >= 0:
        if closing <= opening:
            closing = text.find(spec.close, opening + 1)
            if closing < 0:
                break  # no closer left: no later opener can form a span either
        if closing - opening > spec.max_span_chars:
            position = opening + 1
            continue
        spans.append((opening, closing + 1))
        position = closing + 1
    return spans


def _nesting_spans(text: str, spec: QuoteSpanParams) -> list[tuple[int, int]]:
    """Nesting spans with a stack of openers.

    A closer pairs with the most recent opener. An opener that has no closer within
    `max_span_chars` is abandoned (dropped from the bottom of the stack), so one stray quote
    cannot keep every later pair "nested" and unmasked. Linear in the number of quote characters.
    """
    spans: list[tuple[int, int]] = []
    openers: deque[int] = deque()
    quotes = re.compile(f"[{re.escape(spec.open)}{re.escape(spec.close)}]")
    for match in quotes.finditer(text):
        index = match.start()
        while openers and index - openers[0] > spec.max_span_chars:
            openers.popleft()
        if match.group() == spec.open:
            openers.append(index)
        elif openers:
            spans.append((openers.pop(), index + 1))
    return spans


class PatternHierarchyChunker(Chunker):
    # 2: oversize pieces carry their own pages, merging respects the heading prefix, quote spans
    # are merged/abandoned (stray quotes), no default preamble heading, sanitized text.
    version = 2
    min_chunk_max_tokens = MIN_CHUNK_MAX_TOKENS
    params_model = PatternHierarchyParams

    def __init__(self, counter: TokenCounter) -> None:
        self._counter = counter

    async def chunk(self, parsed: ParsedDocument, profile: RagProfile) -> list[ChunkDraft]:
        budget = chunk_budget(profile.chunk_max_tokens)
        params = validate_params(
            PatternHierarchyParams, profile.chunker_params, where="pattern_hierarchy chunker params"
        )
        return await asyncio.to_thread(_Run(self._counter, parsed, params, budget).chunk)


class _Run:
    """One chunking pass over one document (holds the per-document state)."""

    def __init__(
        self,
        counter: TokenCounter,
        parsed: ParsedDocument,
        params: PatternHierarchyParams,
        budget: int,
    ) -> None:
        self._joined: JoinedText = parsed.joined()
        self._text = self._joined.text
        self._params = params
        self._budget = budget
        self._splitter = BudgetSplitter(counter)
        self._counter = counter
        depths = params.depths()
        self._levels = [
            _Level(
                name=spec.name,
                depth=depths[spec.name],
                order=order,
                regex=compile_profile_regex(spec.pattern),
                template=spec.heading_template or _default_template(spec.pattern),
                context=spec.role == "context",
                reset_on_change=tuple(spec.reset_on_change),
                keep_line=spec.keep_heading_line,
            )
            for order, spec in enumerate(params.levels)
        ]
        self._split_depth = depths[params.split_level]
        self._document_vars: dict[str, object] = {
            **{key: _plain(value) for key, value in parsed.document_metadata.items()},
        }
        self._document_vars.setdefault("title", parsed.title)
        self._excised: list[tuple[int, int]] = []

    # -- pipeline ---------------------------------------------------------------------------

    def chunk(self) -> list[ChunkDraft]:
        matches = self._matches()
        boundaries, deeper = self._structure(matches)
        units = self._units(boundaries)
        units = self._fold(units)
        units = self._merge(units)
        drafts: list[ChunkDraft] = []
        for unit in units:
            drafts.extend(self._emit(unit, deeper))
        return drafts

    def _matches(self) -> list[_Match]:
        spans = _quote_spans(self._text, self._params.ignore_inside)
        span_starts = [span[0] for span in spans]
        found: list[_Match] = []
        position = 0
        for line in self._text.split("\n"):
            for level in self._levels:
                match = level.regex.match(line)
                if match is None:
                    continue
                if spans and _inside(position, spans, span_starts):
                    continue
                groups = {k: v for k, v in match.groupdict().items() if v}
                found.append(_Match(position, position + len(line), level, groups))
            position += len(line) + 1
        # One match per line: the shallowest level wins, ties go to the declaration order.
        found.sort(key=lambda m: (m.start, m.level.depth, m.level.order))
        unique: list[_Match] = []
        for candidate in found:
            if not unique or unique[-1].start != candidate.start:
                unique.append(candidate)
        return unique

    def _structure(
        self, matches: Sequence[_Match]
    ) -> tuple[list[tuple[int, tuple[_Frame, ...]]], list[_Match]]:
        """Open-level stack walk: the unit boundaries (at or above the split depth) and the
        matches deeper than the split depth (oversize cut points)."""
        stack: dict[int, tuple[_Level, _Groups]] = {}
        boundaries: list[tuple[int, tuple[_Frame, ...]]] = []
        for match in matches:
            level, depth = match.level, match.level.depth
            if not level.keep_line:
                self._excised.append((match.start, match.line_end))
            if level.context:
                current = stack.get(depth)
                if (
                    current is not None
                    and current[0].name == level.name
                    and all(current[1].get(k) == match.groups.get(k) for k in level.reset_on_change)
                ):
                    continue  # the same context printed again: keep the children open
            for open_depth in [d for d in stack if d >= depth]:
                del stack[open_depth]
            stack[depth] = (level, match.groups)
            if depth <= self._split_depth:
                boundaries.append((match.start, _snapshot(stack)))
        deeper = [m for m in matches if m.level.depth > self._split_depth]
        return boundaries, deeper

    def _units(self, boundaries: Sequence[tuple[int, tuple[_Frame, ...]]]) -> list[_Unit]:
        units: list[_Unit] = []
        first = boundaries[0][0] if boundaries else len(self._text)
        if first > 0 and self._params.preamble == "keep":
            self._add_unit(units, 0, first, (), preamble=True)
        for index, (start, frames) in enumerate(boundaries):
            end = boundaries[index + 1][0] if index + 1 < len(boundaries) else len(self._text)
            self._add_unit(units, start, end, frames, preamble=False)
        return units

    def _add_unit(
        self,
        units: list[_Unit],
        start: int,
        end: int,
        frames: tuple[_Frame, ...],
        *,
        preamble: bool,
    ) -> None:
        body = self._clean_text(start, end)
        if not body:
            return
        groups: _Groups = {}
        for _depth, _level, level_groups in frames:
            groups.update(level_groups)
        units.append(_Unit(start, end, frames, preamble, body, [groups], [start]))

    def _kept_ranges(self, start: int, end: int) -> list[tuple[int, int]]:
        """`[start, end)` minus the lines of `keep_heading_line: false` levels, as source ranges."""
        ranges: list[tuple[int, int]] = []
        position = start
        first = bisect.bisect_left(self._excised, (start, -1))  # sorted by position
        for cut_start, cut_end in self._excised[first:]:
            if cut_start >= end:
                break
            ranges.append((position, cut_start))
            position = cut_end
        ranges.append((position, end))
        return ranges

    def _clean_text(self, start: int, end: int) -> str:
        """`text[start:end]` without the lines of `keep_heading_line: false` levels."""
        raw = "".join(self._text[a:b] for a, b in self._kept_ranges(start, end))
        return _BLANK_RUNS.sub("\n\n", raw).strip()

    def _clean_mapped(self, start: int, end: int) -> tuple[str, array[int]]:
        """Like `_clean_text`, plus the source offset of every character of the result."""
        ranges = self._kept_ranges(start, end)
        raw = "".join(self._text[a:b] for a, b in ranges)
        origins: array[int] = array("q")
        for a, b in ranges:
            origins.extend(range(a, b))
        pieces: list[str] = []
        kept: array[int] = array("q")
        position = 0
        for run in _BLANK_RUNS.finditer(raw):
            pieces.append(raw[position : run.start() + 2])
            kept.extend(origins[position : run.start() + 2])
            position = run.end()
        pieces.append(raw[position:])
        kept.extend(origins[position:])
        text = "".join(pieces)
        lead = len(text) - len(text.lstrip())
        trail = len(text.rstrip())
        return text[lead:trail], kept[lead:trail]

    def _fold(self, units: list[_Unit]) -> list[_Unit]:
        limit = self._params.fold_intro_below_tokens
        if not limit:
            return units
        kept: list[_Unit] = []
        index = 0
        while index < len(units):
            unit = units[index]
            following = units[index + 1] if index + 1 < len(units) else None
            if (
                following is not None
                and not unit.preamble
                and unit.frames
                and following.frames
                and following.depth > unit.depth
                and self._count(unit.body) < limit
            ):
                following.start = unit.start
                following.body = f"{unit.body}\n{following.body}"
                index += 1
                continue
            kept.append(unit)
            index += 1
        return kept

    def _merge(self, units: list[_Unit]) -> list[_Unit]:
        limit = self._params.merge_below_tokens
        merged: list[_Unit] = []
        index = 0
        while index < len(units):
            unit = units[index]
            # The merged chunk keeps the first unit's heading prefix: that is what the body
            # competes with for the budget.
            room = self._net_budget(self._heading(unit)[1]) if limit and not unit.preamble else 0
            while (
                limit
                and index + 1 < len(units)
                and not unit.preamble
                and self._count(unit.body) < limit
                and unit.own == units[index + 1].own
                and self._parent(unit) == self._parent(units[index + 1])
                and self._count(f"{unit.body}\n{units[index + 1].body}") <= room
            ):
                following = units[index + 1]
                unit = _Unit(
                    unit.start,
                    following.end,
                    unit.frames,
                    False,
                    f"{unit.body}\n{following.body}",
                    [*unit.members, *following.members],
                    [*unit.member_starts, *following.member_starts],
                )
                index += 1
            merged.append(unit)
            index += 1
        return merged

    def _parent(self, unit: _Unit) -> tuple[object, ...]:
        return tuple(
            (depth, level.name, tuple(sorted(groups.items())))
            for depth, level, groups in unit.frames
            if depth < self._split_depth
        )

    # -- chunk assembly ---------------------------------------------------------------------

    def _heading(self, unit: _Unit) -> tuple[str, str]:
        """(path, heading prefix) of a unit; the prefix is empty without `include_heading`."""
        path = (self._params.preamble_heading or "") if unit.preamble else self._path(unit)
        prefix = self._prefix(path, unit.members[0]) if self._params.include_heading else ""
        return path, prefix

    def _net_budget(self, prefix: str) -> int:
        """Tokens left for a body once its heading prefix (and the newline after it) is counted."""
        return max(self._budget - (self._count(prefix) + 1 if prefix else 0), 1)

    def _emit(self, unit: _Unit, deeper: Sequence[_Match]) -> list[ChunkDraft]:
        path, prefix = self._heading(unit)
        budget = self._net_budget(prefix)
        if self._count(unit.body) <= budget:
            pieces = [_Piece(unit.body, unit.start, unit.end)]
        else:
            pieces = self._oversize(unit, deeper, budget)
        return [
            ChunkDraft(
                text=f"{prefix}\n{piece.text}" if prefix else piece.text,
                metadata=self._metadata(unit, path, piece),
                groups=_collect_groups(self._members_of(unit, piece)),
                body=piece.text,
            )
            for piece in pieces
        ]

    @staticmethod
    def _members_of(unit: _Unit, piece: "_Piece") -> list[_Groups]:
        """The merged units a piece overlaps (all of them for a piece that is the whole unit)."""
        if len(unit.members) == 1:
            return unit.members
        ends = [*unit.member_starts[1:], unit.end]
        chosen = [
            groups
            for groups, begin, finish in zip(unit.members, unit.member_starts, ends, strict=True)
            if begin < piece.end and finish > piece.start
        ]
        return chosen or unit.members[:1]

    def _path(self, unit: _Unit) -> str:
        parts = [render_template(level.template, groups) for _depth, level, groups in unit.frames]
        return HEADING_SEPARATOR.join(part for part in parts if part)

    def _prefix(self, path: str, groups: Mapping[str, str]) -> str:
        values: dict[str, object] = {**self._document_vars, **groups, "path": path}
        prefix = render_template(self._params.heading_prefix_template, values)
        cap = self._budget // MAX_HEADING_BUDGET_SHARE
        if prefix and self._count(prefix) > cap:
            prefix = self._counter.split(prefix, max(cap, 1))[0]
        return prefix

    def _metadata(self, unit: _Unit, path: str, piece: "_Piece") -> dict[str, MetadataValue]:
        metadata: dict[str, MetadataValue] = {}
        pages = self._joined.pages_between(piece.start, piece.end)
        if pages is not None:
            metadata["page"], metadata["page_end"] = pages
        if path:
            metadata["heading"] = path
        parent = self._params.parent_level
        if parent is not None:
            for _depth, level, groups in unit.frames:
                if level.name == parent and (heading := render_template(level.template, groups)):
                    metadata["parent_heading"] = heading
        return metadata

    def _oversize(self, unit: _Unit, deeper: Sequence[_Match], budget: int) -> list["_Piece"]:
        """Cut an over-budget unit and report each piece with the source range it came from."""
        wanted = set(self._params.oversize.levels)
        cuts = {
            m.start for m in deeper if m.level.name in wanted and unit.start < m.start < unit.end
        }
        cuts |= self._extra_cuts(unit)
        bounds = [unit.start, *sorted(cuts), unit.end]
        # `joined` = the cleaned segments one per line; `origins[i]` = source offset of joined[i].
        parts: list[str] = []
        origins: array[int] = array("q")
        segments: list[Span] = []
        length = 0
        for first, last in itertools.pairwise(bounds):
            text, offsets = self._clean_mapped(first, last)
            if not text:
                continue
            if parts:
                parts.append("\n")
                origins.append(offsets[0])
                length += 1
            segments.append(Span(length, length + len(text)))
            parts.append(text)
            origins.extend(offsets)
            length += len(text)
        joined = "".join(parts)
        packed: list[Span] = []
        current: Span | None = None
        for segment in segments:
            if current is not None and self._count(joined[current.start : segment.end]) <= budget:
                current = Span(current.start, segment.end)
                continue
            if current is not None:
                packed.append(current)
            current = segment
        if current is not None:
            packed.append(current)
        pieces: list[_Piece] = []
        for block in packed:
            for sub in self._splitter.split_spans(joined[block.start : block.end], budget):
                begin, finish = block.start + sub.start, block.start + sub.end
                pieces.append(_Piece(joined[begin:finish], origins[begin], origins[finish - 1] + 1))
        return pieces

    def _extra_cuts(self, unit: _Unit) -> set[int]:
        patterns = [compile_profile_regex(p) for p in self._params.oversize.extra_split_patterns]
        if not patterns:
            return set()
        cuts: set[int] = set()
        position = unit.start
        for line in self._text[unit.start : unit.end].split("\n"):
            if position > unit.start and any(p.match(line) for p in patterns):
                cuts.add(position)
            position += len(line) + 1
        return cuts

    def _count(self, text: str) -> int:
        return self._splitter.count(text)


@dataclass(frozen=True)
class _Piece:
    """A chunk body and the `[start, end)` range of the joined text it was read from."""

    text: str
    start: int
    end: int


def _snapshot(stack: Mapping[int, tuple[_Level, _Groups]]) -> tuple[_Frame, ...]:
    return tuple((depth, level, dict(groups)) for depth, (level, groups) in sorted(stack.items()))


def _inside(position: int, spans: Sequence[tuple[int, int]], starts: Sequence[int]) -> bool:
    index = bisect.bisect_right(starts, position) - 1
    return index >= 0 and spans[index][0] < position < spans[index][1]


def _default_template(pattern: str) -> str:
    groups = sorted(re.compile(pattern).groupindex.items(), key=lambda item: item[1])
    return "{" + groups[0][0] + "}" if groups else ""


def _plain(value: object) -> object:
    return ", ".join(str(item) for item in value) if isinstance(value, tuple) else value


def _collect_groups(members: Sequence[Mapping[str, str]]) -> dict[str, tuple[str, ...]]:
    collected: dict[str, list[str]] = {}
    for groups in members:
        for key, value in groups.items():
            collected.setdefault(key, []).append(value)
    return {key: tuple(values) for key, values in collected.items()}
