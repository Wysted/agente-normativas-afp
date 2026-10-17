"""Safe use of profile-supplied regular expressions (cleaners now, chunkers later).

Profiles are configuration written by operators, but a pattern that backtracks catastrophically
would freeze an ingest worker (the GIL is held inside `re`). Two defenses, both applied to EVERY
regex a profile supplies:

* Validation (`check_profile_regex`, run when the profile loads): the pattern must compile and
  must not repeat, without bound, a group that itself repeats without bound, e.g. `(\\w+\\s?)*`,
  `(a+)+`, `(.*a)*`. Those are the classic exponential shapes. Rewrite them, or make the inner
  part atomic (`(?>...)`) or the repeat possessive (`*+`, `++`, `?+`, `{m,n}+`, Python 3.11+): an
  atomic group never gives characters back, so nesting it is safe. Other ambiguous shapes
  (overlapping alternatives such as `(a|aa)+`) are NOT detected: keep patterns simple.
* Runtime (`ProfileRegex`): the input is capped at `MAX_REGEX_INPUT_CHARS` characters (one line,
  or one collected value). A longer input does not match and is reported once per instance.
"""

import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Final, NamedTuple

logger = logging.getLogger(__name__)

MAX_REGEX_INPUT_CHARS: Final = 4096
# Extractors scan chunks or windows of a document (multi-line, `finditer`). The nested-quantifier
# check already rules out exponential patterns, but a polynomial one (`\w+\s+\w+x`) still costs
# the square of the input, so the cap stays small: callers feed texts bigger than this in
# windows of `SCAN_WINDOW_CHARS` (see `extractors.base.scan_windows`), never as one string.
MAX_REGEX_SCAN_CHARS: Final = 100_000
SCAN_WINDOW_CHARS: Final = 64_000
# A repeat with at least this many iterations (or no upper bound) counts as "unbounded": the
# damage of nesting two bounded repeats such as `(\d{1,3}\.){2}` is polynomial and tiny.
UNBOUNDED_REPEAT_FROM: Final = 16

_BRACE_QUANTIFIER = re.compile(r"\{(\d*)(?:(,)(\d*))?\}")


class _Repeat(NamedTuple):
    end: int  # index just after the quantifier (and its lazy/possessive suffix)
    unbounded: bool
    possessive: bool


@dataclass
class _Group:
    atomic: bool
    has_unbounded_repeat: bool = False


def _repeat_at(pattern: str, index: int) -> _Repeat | None:
    """The quantifier starting at `index`, if any."""
    char = pattern[index] if index < len(pattern) else ""
    if char in ("*", "+"):
        end, unbounded = index + 1, True
    elif char == "?":
        end, unbounded = index + 1, False
    elif char == "{" and (m := _BRACE_QUANTIFIER.match(pattern, index)) and (m[1] or m[3]):
        low = int(m[1] or 0)
        high = int(m[3]) if m[3] else (None if m[2] else low)
        end = m.end()
        unbounded = high is None or high >= UNBOUNDED_REPEAT_FROM
    else:
        return None
    suffix = pattern[end] if end < len(pattern) else ""
    if suffix in ("+", "?"):
        end += 1
    return _Repeat(end, unbounded, possessive=suffix == "+")


def _skip_class(pattern: str, index: int) -> int:
    """Index just after the character class that starts at `index` (a `[`)."""
    i = index + 1
    if i < len(pattern) and pattern[i] == "^":
        i += 1
    if i < len(pattern) and pattern[i] == "]":
        i += 1
    while i < len(pattern) and pattern[i] != "]":
        i += 2 if pattern[i] == "\\" else 1
    return i + 1


def find_nested_quantifier(pattern: str) -> int | None:
    """Index of the `)` closing a group that is repeated without bound while it contains an
    unbounded repeat (and is neither atomic nor possessive), else None."""
    groups = [_Group(atomic=False)]
    i, size = 0, len(pattern)
    while i < size:
        char = pattern[i]
        if char == "(":
            if pattern.startswith("(?#", i):  # inline comment
                close = pattern.find(")", i)
                i = size if close < 0 else close + 1
                continue
            groups.append(_Group(atomic=pattern.startswith("(?>", i)))
            i += 1
            continue
        if char == ")" and len(groups) > 1:
            group = groups.pop()
            closing = i
            repeat = _repeat_at(pattern, i + 1)
            i = repeat.end if repeat else i + 1
            if repeat is not None and repeat.possessive:
                group.atomic = True
            repeated = repeat is not None and repeat.unbounded
            if repeated and group.has_unbounded_repeat and not group.atomic:
                return closing
            if not group.atomic and (repeated or group.has_unbounded_repeat):
                groups[-1].has_unbounded_repeat = True
            continue
        if char == "\\":
            i += 2
        elif char == "[":
            i = _skip_class(pattern, i)
        else:
            i += 1
        repeat = _repeat_at(pattern, i)
        if repeat is not None:
            i = repeat.end
            if repeat.unbounded and not repeat.possessive:
                groups[-1].has_unbounded_repeat = True
    return None


def check_profile_regex(pattern: str) -> str:
    """Validate a profile regex (compiles, no nested unbounded repeats); returns it unchanged."""
    try:
        re.compile(pattern)
    except re.error as exc:
        raise ValueError(f"invalid regular expression {pattern!r}: {exc}") from exc
    if (at := find_nested_quantifier(pattern)) is not None:
        raise ValueError(
            f"regular expression {pattern!r} repeats a group that already repeats "
            f"(nested quantifier ending at position {at}): it can backtrack catastrophically; "
            "rewrite it, or make the inner part atomic '(?>...)' or the repeat possessive "
            "('*+', '++')"
        )
    return pattern


def compile_profile_regex(
    pattern: str, flags: int = 0, *, max_chars: int = MAX_REGEX_INPUT_CHARS
) -> "ProfileRegex":
    """Validate and compile a profile regex into its input-capped wrapper.

    `max_chars` raises the cap for scanners that read a whole chunk or document
    (`MAX_REGEX_SCAN_CHARS`); line-oriented use keeps the default.
    """
    return ProfileRegex(check_profile_regex(pattern), flags, max_chars=max_chars)


class ProfileRegex:
    """A compiled profile regex whose `search`/`match`/`finditer` refuse oversized inputs.

    Create one per document/operation: the oversize warning is logged once per instance.
    """

    def __init__(
        self, pattern: str, flags: int = 0, *, max_chars: int = MAX_REGEX_INPUT_CHARS
    ) -> None:
        self.regex = re.compile(pattern, flags)
        self._max_chars = max_chars
        self._oversized = 0

    @property
    def pattern(self) -> str:
        return self.regex.pattern

    def _fits(self, text: str) -> bool:
        if len(text) <= self._max_chars:
            return True
        if not self._oversized:
            logger.warning(
                "Regex %r skipped an input of %d characters (limit %d); further oversized "
                "inputs for this pattern are skipped silently",
                self.regex.pattern,
                len(text),
                self._max_chars,
            )
        self._oversized += 1
        return False

    def search(self, text: str) -> re.Match[str] | None:
        return self.regex.search(text) if self._fits(text) else None

    def match(self, text: str) -> re.Match[str] | None:
        return self.regex.match(text) if self._fits(text) else None

    def finditer(self, text: str) -> Iterator[re.Match[str]]:
        return self.regex.finditer(text) if self._fits(text) else iter(())
