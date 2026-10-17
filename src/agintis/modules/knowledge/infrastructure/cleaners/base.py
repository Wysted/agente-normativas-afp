"""Common machinery of the cleaners: params validation, worker thread, section helpers.

Conventions (shared by every cleaner)
* Input is never mutated; a new `ParsedDocument` is returned.
* "Page" cleaners (running headers, repeated page lines, TOC pages, drop-before) look only at
  sections that carry a page number (PDF pages); documents without pages are returned as-is.
  "Text" cleaners apply to every section.
* A section whose text becomes blank is dropped.
* Regex parameters are validated when the profile loads (`Pattern` type: compiles, no nested
  unbounded quantifiers), applied line by line (never `re.M` over a whole document) through
  `compile_profile_regex`, which refuses lines over `MAX_REGEX_INPUT_CHARS`.
* `version` is part of the profile fingerprint. Bump it whenever a change to the cleaner's code
  or defaults can change its output: documents ingested with the old version are re-indexed.
"""

import asyncio
from abc import abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import ClassVar

from agintis.modules.knowledge.application import Cleaner, ParsedDocument, ParsedSection
from agintis.modules.knowledge.domain import ParamValue
from agintis.modules.knowledge.profiles import ParamsModel, Pattern, validate_params

__all__ = ["ParamsCleaner", "Pattern"]


class ParamsCleaner[P: ParamsModel](Cleaner):
    """A cleaner with a typed, validated parameter model; subclasses implement `_apply`."""

    key: ClassVar[str]
    params_model: type[P]
    # Pipeline order rules (see `CleanerOrdering`): a cleaner that rebuilds the sections from
    # the raw source sets `rebuilds_text`; one that never edits the sections clears `edits_text`.
    rebuilds_text: ClassVar[bool] = False
    edits_text: ClassVar[bool] = True

    async def clean(
        self, parsed: ParsedDocument, params: Mapping[str, ParamValue]
    ) -> ParsedDocument:
        typed = validate_params(self.params_model, params, where=f"cleaner {self.key!r} params")
        return await asyncio.to_thread(self._apply, parsed, typed)

    @abstractmethod
    def _apply(self, parsed: ParsedDocument, params: P) -> ParsedDocument: ...


def with_sections(parsed: ParsedDocument, sections: Sequence[ParsedSection]) -> ParsedDocument:
    """Same document with new sections; blank ones are dropped."""
    kept = tuple(s for s in sections if s.text.strip())
    return replace(parsed, sections=kept)


def map_texts(parsed: ParsedDocument, transform: Callable[[str], str]) -> ParsedDocument:
    return with_sections(parsed, [replace(s, text=transform(s.text)) for s in parsed.sections])


def map_lines(
    parsed: ParsedDocument, transform: Callable[[list[str]], list[str]]
) -> ParsedDocument:
    return map_texts(parsed, lambda text: "\n".join(transform(text.splitlines())))


def paged_indexes(parsed: ParsedDocument) -> list[int]:
    return [i for i, s in enumerate(parsed.sections) if s.page is not None]


def nonblank_indexes(lines: Sequence[str]) -> list[int]:
    return [i for i, line in enumerate(lines) if line.strip()]
