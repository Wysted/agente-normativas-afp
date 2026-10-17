"""Budget-driven splitting of one piece of text, shared by the structural chunkers.

`BudgetSplitter.split_spans` cuts an over-budget text at the most natural boundary available,
trying in order: blank lines, a line that ends a sentence/clause (`.`, `;`, `:`), any line break,
a sentence/clause end followed by spaces, any space, and finally token boundaries (so the budget
always holds, even for a "word" longer than the budget). Pieces are packed greedily: neighbours
are re-joined while they still fit.

Every piece is reported as a `[start, end)` span of the ORIGINAL text, not just as a string: the
caller needs the offsets to tell which page(s) each piece covers. A packed piece is the original
slice (separators included), so it is exactly the text it points at.
"""

import re
from typing import Final, NamedTuple

from agintis.modules.knowledge.application import InvalidProfileError
from agintis.modules.knowledge.infrastructure.chunkers.tokens import TokenCounter

# Tokenizers add special tokens (bge-m3 adds <s> and </s>): chunkers leave this much room.
SPECIAL_TOKENS_RESERVE: Final = 8
# A heading/prefix/header may use at most budget // MAX_HEADING_BUDGET_SHARE tokens.
MAX_HEADING_BUDGET_SHARE: Final = 2
# Smallest token budget (after the special tokens) the structural chunkers work with: below it a
# heading prefix plus a few words of body cannot fit and every chunk would be a degenerate cut.
MIN_BODY_BUDGET: Final = 16
MIN_CHUNK_MAX_TOKENS: Final = SPECIAL_TOKENS_RESERVE + MIN_BODY_BUDGET

# Pieces are cached by text: only short ones (packing candidates are large and never repeat).
_CACHE_MAX_CHARS: Final = 1_000
_CACHE_MAX_ENTRIES: Final = 50_000

# Separators, finest last. A compiled pattern is matched with `finditer`; a string with `find`.
_SEPARATORS: Final[tuple[re.Pattern[str] | str, ...]] = (
    "\n\n",
    re.compile(r"(?<=[.;:])\n"),
    "\n",
    re.compile(r"(?<=[.;:])\s+"),
    " ",
)


class Span(NamedTuple):
    """`text[start:end]` of the text that was split (whitespace-trimmed at both ends)."""

    start: int
    end: int


def chunk_budget(chunk_max_tokens: int) -> int:
    """Token budget of a chunk: `chunk_max_tokens` minus the special tokens.

    `InvalidProfileError` when nothing is left. Profiles are also checked at boot against the
    chunker's `min_chunk_max_tokens`; this is the runtime guard for direct callers.
    """
    budget = chunk_max_tokens - SPECIAL_TOKENS_RESERVE
    if budget <= 0:
        raise InvalidProfileError(
            f"chunk_max_tokens must exceed {SPECIAL_TOKENS_RESERVE} (special tokens reserve)"
        )
    return budget


class BudgetSplitter:
    """Splits text so that every piece has at most `budget` tokens (per `counter`)."""

    def __init__(self, counter: TokenCounter) -> None:
        self._counter = counter
        self._cache: dict[str, int] = {}

    def count(self, text: str) -> int:
        if len(text) > _CACHE_MAX_CHARS:
            return self._counter.count(text)
        cached = self._cache.get(text)
        if cached is None:
            cached = self._counter.count(text)
            if len(self._cache) < _CACHE_MAX_ENTRIES:
                self._cache[text] = cached
        return cached

    def split(self, text: str, budget: int) -> list[str]:
        return [text[start:end] for start, end in self.split_spans(text, budget)]

    def split_spans(self, text: str, budget: int) -> list[Span]:
        """The pieces of `text` as offsets into it, in order, each within `budget` tokens."""
        if budget <= 0:
            raise ValueError("budget must be greater than 0")
        return self._split(text, 0, len(text), budget)

    def _split(self, text: str, start: int, end: int, budget: int) -> list[Span]:
        start, end = _trim(text, start, end)
        if start >= end:
            return []
        if self.count(text[start:end]) <= budget:
            return [Span(start, end)]
        for separator in _SEPARATORS:
            parts = _cut(text, start, end, separator)
            if len(parts) > 1:
                return self._pack(text, parts, budget)
        return self._cut_tokens(text, start, end, budget)

    def _pack(self, text: str, parts: list[Span], budget: int) -> list[Span]:
        packed: list[Span] = []
        current: Span | None = None
        for part in parts:
            if current is not None and self.count(text[current.start : part.end]) <= budget:
                current = Span(current.start, part.end)
                continue
            if current is not None:
                packed.append(current)
            if self.count(text[part.start : part.end]) > budget:
                packed.extend(self._split(text, part.start, part.end, budget))
                current = None
            else:
                current = part
        if current is not None:
            packed.append(current)
        return packed

    def _cut_tokens(self, text: str, start: int, end: int, budget: int) -> list[Span]:
        """Last resort: token boundaries. The counter returns verbatim, ordered substrings."""
        spans: list[Span] = []
        cursor = start
        for piece in self._counter.split(text[start:end], budget):
            found = text.find(piece, cursor, end)
            if found < 0:  # a counter that rewrites text: keep the order, lose the exact offset
                found = cursor
            spans.append(Span(found, min(found + len(piece), end)))
            cursor = found + len(piece)
        return spans


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _cut(text: str, start: int, end: int, separator: re.Pattern[str] | str) -> list[Span]:
    """The non-blank pieces of `text[start:end]` between occurrences of `separator`."""
    boundaries: list[tuple[int, int]] = []  # (separator start, separator end)
    if isinstance(separator, str):
        position = text.find(separator, start, end)
        while position >= 0:
            boundaries.append((position, position + len(separator)))
            position = text.find(separator, position + len(separator), end)
    else:
        boundaries = [(m.start(), m.end()) for m in separator.finditer(text, start, end)]
    pieces: list[Span] = []
    cursor = start
    for sep_start, sep_end in [*boundaries, (end, end)]:
        piece_start, piece_end = _trim(text, cursor, sep_start)
        if piece_start < piece_end:
            pieces.append(Span(piece_start, piece_end))
        cursor = sep_end
    return pieces
