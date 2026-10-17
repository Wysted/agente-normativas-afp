"""Structure-aware chunker (key "structured").

* Never crosses a section boundary (heading, or PDF page): headings/pages stay citable.
  With `cross_pages` it may cross PAGE boundaries inside the same heading path (a numbered
  item that continues on the next page stays in one chunk). Every chunk from a paged source
  records the pages it covers (`page` = first, `page_end` = last; equal for a single page),
  found through the joined-text offset map.
* Inside a section, paragraphs (blank-line separated) are merged up to the token budget.
* A paragraph over budget is packed by sentences; a sentence over budget is cut at token
  boundaries. Every returned chunk has at most `chunk_max_tokens - SPECIAL_TOKENS_RESERVE`
  tokens, leaving room for the model's special tokens (bge-m3 adds 2: <s> and </s>).
* No overlap.

Chunker params (`StructuredParams`; part of the profile fingerprint, so changing them re-indexes)
* `include_heading` (default True): the section heading path ("A > B") is prepended to the
  text of every chunk of the section, so the embedded and stored text carries its context.
  Its tokens count against the budget (a heading longer than half the budget is truncated).
  The `heading` metadata is set either way.
* `cross_pages` (default False): see above. Off = chunks never span pages.

`version` 2: `page_end` is always stored (it used to exist only with `cross_pages`) and a chunk
that overflows the budget once the heading prefix is added re-splits only that chunk.
"""

import asyncio
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Final

from agintis.modules.knowledge.application import (
    ChunkDraft,
    Chunker,
    JoinedText,
    ParsedDocument,
    ParsedSection,
    join_sections,
)
from agintis.modules.knowledge.domain import MetadataValue, RagProfile
from agintis.modules.knowledge.infrastructure.chunkers.splitting import (
    MAX_HEADING_BUDGET_SHARE,
    MIN_CHUNK_MAX_TOKENS,
    chunk_budget,
)
from agintis.modules.knowledge.infrastructure.chunkers.tokens import TokenCounter
from agintis.modules.knowledge.profiles import Flag, ParamsModel, validate_params

STRUCTURED_CHUNKER_KEY: Final = "structured"
HEADING_SEPARATOR: Final = " > "
PARAGRAPH_SEPARATOR: Final = "\n\n"

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")


class StructuredParams(ParamsModel):
    include_heading: Flag = True
    cross_pages: Flag = False


@dataclass(frozen=True)
class _Unit:
    """A piece of text with the pages (first, last) it came from; None = pageless source."""

    text: str
    pages: tuple[int, int] | None


def _hull(a: tuple[int, int] | None, b: tuple[int, int] | None) -> tuple[int, int] | None:
    if a is None or b is None:
        return a or b
    return (min(a[0], b[0]), max(a[1], b[1]))


class StructuredChunker(Chunker):
    version = 2
    min_chunk_max_tokens = MIN_CHUNK_MAX_TOKENS
    params_model = StructuredParams

    def __init__(self, counter: TokenCounter) -> None:
        self._counter = counter

    async def chunk(self, parsed: ParsedDocument, profile: RagProfile) -> list[ChunkDraft]:
        budget = chunk_budget(profile.chunk_max_tokens)
        params = validate_params(
            StructuredParams, profile.chunker_params, where="structured chunker params"
        )
        return await asyncio.to_thread(self._chunk_all, parsed, budget, params)

    def _chunk_all(
        self, parsed: ParsedDocument, budget: int, params: StructuredParams
    ) -> list[ChunkDraft]:
        drafts: list[ChunkDraft] = []
        for group in self._groups(parsed.sections, params.cross_pages):
            headings = group[0].headings
            joined = join_sections(group)
            for unit in self._chunk_group(joined, headings, budget, params.include_heading):
                drafts.append(
                    ChunkDraft(text=unit.text, metadata=self._metadata(headings, unit.pages))
                )
        return drafts

    @staticmethod
    def _groups(
        sections: Sequence[ParsedSection], cross_pages: bool
    ) -> Iterator[list[ParsedSection]]:
        group: list[ParsedSection] = []
        for section in sections:
            if group and not (cross_pages and group[-1].headings == section.headings):
                yield group
                group = []
            group.append(section)
        if group:
            yield group

    @staticmethod
    def _metadata(
        headings: tuple[str, ...], pages: tuple[int, int] | None
    ) -> dict[str, MetadataValue]:
        metadata: dict[str, MetadataValue] = {}
        if pages is not None:
            metadata["page"] = pages[0]
            metadata["page_end"] = pages[1]
        if headings:
            metadata["heading"] = HEADING_SEPARATOR.join(headings)
        return metadata

    def _heading_prefix(self, headings: tuple[str, ...], budget: int) -> str:
        if not headings:
            return ""
        heading = HEADING_SEPARATOR.join(headings)
        cap = budget // MAX_HEADING_BUDGET_SHARE
        if cap < 1:
            return ""
        if self._counter.count(heading) > cap:
            heading = self._counter.split(heading, cap)[0]
        return f"{heading}{PARAGRAPH_SEPARATOR}"

    def _chunk_group(
        self, joined: JoinedText, headings: tuple[str, ...], budget: int, include_heading: bool
    ) -> list[_Unit]:
        prefix = self._heading_prefix(headings, budget) if include_heading else ""
        if not prefix:
            return self._split_text(joined, budget)
        body_budget = budget - self._counter.count(prefix)
        return [
            fitted
            for unit in self._split_text(joined, body_budget)
            for fitted in self._compose(prefix, unit, body_budget, budget)
        ]

    def _compose(self, prefix: str, unit: _Unit, body_budget: int, budget: int) -> list[_Unit]:
        """`prefix + unit` as chunks within `budget`.

        Tokenizers may merge tokens across the prefix/body boundary, so the composed text can
        exceed the budget although both parts fit. Only the offending unit is re-split, with a
        body budget smaller by the overflow; the rest of the group is untouched. Iterative: a
        pathological tokenizer cannot deepen the call stack.
        """
        pending = [(unit, body_budget)]
        chunks: list[_Unit] = []
        while pending:
            current, allowed = pending.pop(0)
            composed = _Unit(prefix + current.text, current.pages)
            overflow = self._counter.count(composed.text) - budget
            if overflow <= 0 or allowed <= 1:
                chunks.append(composed)
                continue
            smaller = max(1, allowed - overflow)
            pending[0:0] = [(piece, smaller) for piece in self._resplit(current, smaller)]
        return chunks

    def _resplit(self, unit: _Unit, budget: int) -> list[_Unit]:
        """Split an already merged unit again under a smaller budget (pages stay its hull)."""
        pieces = [
            _Unit(paragraph, unit.pages)
            for paragraph in (p.strip() for p in _PARAGRAPH_SPLIT.split(unit.text))
            if paragraph
        ]
        return self._merge(
            [fitted for piece in pieces for fitted in self._fit_paragraph(piece, budget)],
            PARAGRAPH_SEPARATOR,
            budget,
        )

    @staticmethod
    def _paragraphs(joined: JoinedText) -> Iterator[_Unit]:
        text = joined.text
        position = 0
        bounds = [(m.start(), m.end()) for m in _PARAGRAPH_SPLIT.finditer(text)]
        for cut_start, cut_end in [*bounds, (len(text), len(text))]:
            raw = text[position:cut_start]
            stripped = raw.strip()
            if stripped:
                start = position + (len(raw) - len(raw.lstrip()))
                yield _Unit(stripped, joined.pages_between(start, start + len(stripped)))
            position = cut_end

    def _split_text(self, joined: JoinedText, budget: int) -> list[_Unit]:
        units = [
            piece for p in self._paragraphs(joined) for piece in self._fit_paragraph(p, budget)
        ]
        return self._merge(units, PARAGRAPH_SEPARATOR, budget)

    def _fit_paragraph(self, paragraph: _Unit, budget: int) -> list[_Unit]:
        """Return the paragraph as-is if it fits, else sentence-packed pieces within budget."""
        if self._counter.count(paragraph.text) <= budget:
            return [paragraph]
        sentences = [s.strip() for s in _SENTENCE_SPLIT.split(paragraph.text) if s.strip()]
        fitted = [
            _Unit(piece, paragraph.pages)
            for sentence in sentences
            for piece in (
                [sentence]
                if self._counter.count(sentence) <= budget
                else self._counter.split(sentence, budget)
            )
        ]
        return self._merge(fitted, " ", budget)

    def _merge(self, units: list[_Unit], separator: str, budget: int) -> list[_Unit]:
        """Greedy packing; every unit already fits the budget on its own."""
        merged: list[_Unit] = []
        current: _Unit | None = None
        for unit in units:
            if current is None:
                current = unit
                continue
            candidate = f"{current.text}{separator}{unit.text}"
            if self._counter.count(candidate) > budget:
                merged.append(current)
                current = unit
            else:
                current = _Unit(candidate, _hull(current.pages, unit.pages))
        if current is not None:
            merged.append(current)
        return merged
