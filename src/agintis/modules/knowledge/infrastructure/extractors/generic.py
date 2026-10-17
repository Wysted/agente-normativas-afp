"""General-purpose extractors, configured entirely by params (nothing here knows a language or
a legal system): regex lists, labeled references, dates after a heading, classification by rules,
templates, layered lookup."""

import re
from collections.abc import Mapping
from datetime import date
from functools import lru_cache
from typing import Final, Literal, Self

from pydantic import Field, field_validator, model_validator

from agintis.modules.knowledge.application import ExtractionContext, RawValue
from agintis.modules.knowledge.domain import ParamValue
from agintis.modules.knowledge.infrastructure.extractors.base import (
    ParamsExtractor,
    memoized,
    regex_flags,
    scan_regex,
    scan_windows,
)
from agintis.modules.knowledge.infrastructure.templates import (
    check_template_syntax,
    render_template,
    template_names,
)
from agintis.modules.knowledge.profiles import MAX_REGEX_SCAN_CHARS, Flag, ParamsModel, Pattern

FLAGS_PATTERN: Final = r"^i?$"
_FIELD_JOINER: Final = " | "


def _text_of(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, tuple):
        return _FIELD_JOINER.join(str(item) for item in value)
    return str(value)


class RegexAllParams(ParamsModel):
    pattern: Pattern
    group: int = Field(default=0, ge=0)
    unique: Flag = False
    flags: str = Field(default="", pattern=FLAGS_PATTERN)
    scope: Literal["chunk", "document"] = "chunk"
    normalize: Literal["none", "compact_spaces"] = "none"

    @model_validator(mode="after")
    def _group_exists(self) -> Self:
        if self.group > re.compile(self.pattern).groups:
            raise ValueError(f"group {self.group} does not exist in the pattern")
        return self


class RegexAll(ParamsExtractor[RegexAllParams]):
    """Every match of `pattern` (capture `group`) in the chunk text or in the whole document.

    The pattern runs over multi-line text, so a reference wrapped over two lines still
    matches (use `\\s+` between words). Text is scanned in windows of `SCAN_WINDOW_CHARS`
    cut at blank lines: a match can only be lost if it straddles such an edge. Values stay
    strings: a list[int] field coerces them. Document-scope results are computed once per
    document.
    """

    key = "regex_all"
    version = 2  # scanned in windows
    params_model = RegexAllParams

    def _extract(self, context: ExtractionContext, params: RegexAllParams) -> RawValue:
        if params.scope == "document":
            return memoized(
                context,
                self.key,
                params,
                lambda: self._find(context.document_text, params),
            )
        return self._find(context.draft.body_text, params)

    @staticmethod
    def _find(text: str, params: RegexAllParams) -> tuple[str, ...]:
        regex = scan_regex(params.pattern, regex_flags(params.flags))
        values: list[str] = []
        for window in scan_windows(text):
            for match in regex.finditer(window):
                value = match.group(params.group)
                if value is None:
                    continue
                if params.normalize == "compact_spaces":
                    value = re.sub(r"\s+", " ", value).strip()
                if value and not (params.unique and value in values):
                    values.append(value)
        return tuple(values)


class LabeledRule(ParamsModel):
    pattern: Pattern
    label: str = Field(min_length=1)
    group: int = Field(default=1, ge=0)
    strip_chars: str = ""
    flags: str = Field(default="", pattern=FLAGS_PATTERN)

    @model_validator(mode="after")
    def _group_exists(self) -> Self:
        if self.group > re.compile(self.pattern).groups:
            raise ValueError(f"group {self.group} does not exist in the pattern")
        return self


class RegexLabeledParams(ParamsModel):
    rules: list[LabeledRule] = Field(min_length=1)
    separator: str = " "
    unique: Flag = True


class RegexLabeled(ParamsExtractor[RegexLabeledParams]):
    """References found by several patterns, each tagged with its own label.

    For every rule, every match of `pattern` yields `<label><separator><value>`, where the value
    is the capture `group` without the characters of `strip_chars` (a thousands separator, say).
    With rules `{pattern: 'Law No\\.?\\s*([\\d.]+)', label: Law, strip_chars: '.'}` the text
    "Law No. 21.735" gives "Law 21735". Rules run in order and their values are concatenated;
    `unique` drops repeats.
    """

    key = "regex_labeled"
    params_model = RegexLabeledParams

    def _extract(self, context: ExtractionContext, params: RegexLabeledParams) -> RawValue:
        found: list[str] = []
        text = context.draft.body_text
        for rule in params.rules:
            regex = scan_regex(rule.pattern, regex_flags(rule.flags))
            for window in scan_windows(text):
                for match in regex.finditer(window):
                    value = match.group(rule.group) or ""
                    for char in rule.strip_chars:
                        value = value.replace(char, "")
                    value = value.strip()
                    if not value:
                        continue
                    reference = f"{rule.label}{params.separator}{value}"
                    if not (params.unique and reference in found):
                        found.append(reference)
        return tuple(found)


class DateParams(ParamsModel):
    """How a written date is read: one regex with a group for day, month name and year."""

    pattern: Pattern
    day_group: int = Field(ge=1)
    month_group: int = Field(ge=1)
    year_group: int = Field(ge=1)
    months: dict[str, int] = Field(min_length=1)  # month name (any case) -> 1..12
    flags: str = Field(default="", pattern=FLAGS_PATTERN)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        groups = re.compile(self.pattern).groups
        for name, index in (
            ("day_group", self.day_group),
            ("month_group", self.month_group),
            ("year_group", self.year_group),
        ):
            if index > groups:
                raise ValueError(f"{name} {index} does not exist in the date pattern")
        if bad := sorted(name for name, number in self.months.items() if not 1 <= number <= 12):
            raise ValueError(f"months {bad} must map to a number from 1 to 12")
        return self


class DateAfterHeadingParams(ParamsModel):
    heading_pattern: Pattern
    window_chars: int = Field(ge=1, le=MAX_REGEX_SCAN_CHARS)
    phrase_pattern: Pattern | None = None
    date: DateParams
    heading_flags: str = Field(default="", pattern=FLAGS_PATTERN)
    phrase_flags: str = Field(default="", pattern=FLAGS_PATTERN)


@lru_cache(maxsize=256)
def _lowered(months: tuple[tuple[str, int], ...]) -> dict[str, int]:
    return {name.lower(): number for name, number in months}


def read_date(text: str, params: DateParams) -> str | None:
    """First date of `text` as ISO `YYYY-MM-DD`; None when absent, unknown month or not a real
    calendar date ("31 February" is no date)."""
    match = scan_regex(params.pattern, regex_flags(params.flags)).search(text)
    if match is None:
        return None
    month = _lowered(tuple(params.months.items())).get(
        (match.group(params.month_group) or "").lower()
    )
    try:
        day, year = int(match.group(params.day_group)), int(match.group(params.year_group))
        return date(year, month, day).isoformat() if month is not None else None
    except (TypeError, ValueError):
        return None


class DateAfterHeading(ParamsExtractor[DateAfterHeadingParams]):
    """Effective date of the whole document (ISO), shared by all its chunks.

    Looks for a line matching `heading_pattern` (tested one stripped line at a time, never across
    a newline) and reads the first date of the next `window_chars` characters; with no such
    heading, the first match of `phrase_pattern` (when given) is searched for a date instead. A
    date that is not a valid calendar date gives no value.
    """

    key = "date_after_heading"
    params_model = DateAfterHeadingParams

    def _extract(self, context: ExtractionContext, params: DateAfterHeadingParams) -> RawValue:
        return memoized(
            context, self.key, params, lambda: self._find(context.document_text, params)
        )

    @staticmethod
    def _find(text: str, params: DateAfterHeadingParams) -> str | None:
        heading = scan_regex(params.heading_pattern, regex_flags(params.heading_flags))
        position = 0
        for line in text.split("\n"):
            position += len(line) + 1
            if heading.match(line.strip()):
                return read_date(text[position : position + params.window_chars], params.date)
        if params.phrase_pattern is None:
            return None
        phrase = scan_regex(params.phrase_pattern, regex_flags(params.phrase_flags))
        for window in scan_windows(text):
            if match := phrase.search(window):
                return read_date(match.group(0), params.date)
        return None


class ClassifyRule(ParamsModel):
    pattern: Pattern
    label: str
    field: str | None = None
    head_chars: int | None = Field(default=None, ge=1, le=MAX_REGEX_SCAN_CHARS)
    flags: str = Field(default="", pattern=FLAGS_PATTERN)

    @model_validator(mode="after")
    def _one_source(self) -> Self:
        if (self.field is None) == (self.head_chars is None):
            raise ValueError("a rule needs exactly one of field or head_chars")
        return self


class ClassifyParams(ParamsModel):
    rules: list[ClassifyRule] = Field(min_length=1)
    default: str | None = None


class Classify(ParamsExtractor[ClassifyParams]):
    """A label from the first rule whose pattern is found.

    A rule reads a document-metadata `field` (a list is joined with " | ") or the first
    `head_chars` characters of the document text. No rule matches -> `default` (or no value).
    """

    key = "classify"
    params_model = ClassifyParams

    def _extract(self, context: ExtractionContext, params: ClassifyParams) -> RawValue:
        # Rules read the document only, so the label is the same for every chunk.
        return memoized(context, self.key, params, lambda: self._label(context, params))

    @staticmethod
    def _label(context: ExtractionContext, params: ClassifyParams) -> str | None:
        for rule in params.rules:
            if rule.field is not None:
                source = context.parsed.document_metadata.get(rule.field)
                if rule.field == "title" and source is None:
                    source = context.parsed.title
                text = _text_of(source)
            else:
                text = context.document_text[: rule.head_chars or 0]
            if scan_regex(rule.pattern, regex_flags(rule.flags)).search(text):
                return rule.label
        return params.default


class TemplateParams(ParamsModel):
    template: str = Field(min_length=1)
    required: list[str] = Field(default_factory=list)

    @field_validator("template")
    @classmethod
    def _syntax(cls, value: str) -> str:
        return check_template_syntax(value)


class Template(ParamsExtractor[TemplateParams]):
    """Text rendered from values already known: earlier metadata fields and chunk groups.

    Uses the heading template syntax (`{x}`, `{x?text}`). `required` names that must have a
    value, otherwise there is no result. Declare the fields it reads BEFORE this one.
    """

    key = "template"
    params_model = TemplateParams

    def references(self, params: Mapping[str, ParamValue]) -> frozenset[str]:
        template, required = params.get("template"), params.get("required")
        names = set(template_names(template)) if isinstance(template, str) else set()
        if isinstance(required, tuple):
            names.update(item for item in required if isinstance(item, str))
        return frozenset(names)

    def _extract(self, context: ExtractionContext, params: TemplateParams) -> RawValue:
        values: dict[str, object] = {
            **{key: group[0] for key, group in context.draft.groups.items() if group},
            **{key: _text_of(value) for key, value in context.resolved.items()},
        }
        if any(not values.get(name) for name in params.required):
            return None
        return render_template(params.template, values) or None


class LayerParams(ParamsModel):
    field: str | None = None
    group: str | None = None

    @model_validator(mode="after")
    def _one_source(self) -> Self:
        if (self.field is None) == (self.group is None):
            raise ValueError("a layer needs exactly one of field or group")
        return self


class KeyParams(ParamsModel):
    pattern: Pattern
    flags: str = Field(default="", pattern=FLAGS_PATTERN)
    upper: Flag = False

    @model_validator(mode="after")
    def _has_capture_group(self) -> Self:
        if re.compile(self.pattern).groups < 1:
            raise ValueError("the pattern needs a capture group")
        return self


class ResetParams(ParamsModel):
    when: str
    unless: str
    clear: list[str] = Field(min_length=1)


class QuotePairParams(ParamsModel):
    open: str = Field(min_length=1, max_length=1)
    close: str = Field(min_length=1, max_length=1)

    @model_validator(mode="after")
    def _distinct(self) -> Self:
        if self.open == self.close:
            raise ValueError("open and close quote characters must differ")
        return self


class LayeredRegexParams(ParamsModel):
    layers: list[LayerParams] = Field(min_length=1)
    keys: dict[str, KeyParams] = Field(min_length=1)
    strip_quoted: QuotePairParams | None = None
    max_chars: int = Field(default=260, ge=1, le=MAX_REGEX_SCAN_CHARS)
    require: list[str] = Field(default_factory=list)
    reset: list[ResetParams] = Field(default_factory=list)

    @model_validator(mode="after")
    def _names_exist(self) -> Self:
        names = set(self.keys)
        used = {*self.require, *(r.when for r in self.reset), *(r.unless for r in self.reset)}
        used.update(c for r in self.reset for c in r.clear)
        if unknown := sorted(used - names):
            raise ValueError(f"require/reset name undeclared keys {unknown}")
        return self


class LayeredRegex(ParamsExtractor[LayeredRegexParams]):
    """Values found in several texts, the later text overriding the earlier.

    For each layer (a document-metadata `field`, or a chunk regex `group`) every key's pattern
    is searched in the layer's text (optionally without the quoted passages, cut to
    `max_chars`); a hit overwrites the key. `reset` rules model "a deeper reference replaces
    the shallower one": when `when` is found in a layer and `unless` is not, `clear` keys drop.
    `require`d keys missing -> nothing at all. Select a key with `extractor:layered_regex.<key>`.
    """

    key = "layered_regex"
    version = 2  # `strip_quoted` takes the quote characters as params
    params_model = LayeredRegexParams

    def selectors_for(self, params: Mapping[str, ParamValue]) -> frozenset[str]:
        keys = params.get("keys")
        return frozenset(keys) if isinstance(keys, Mapping) else frozenset()

    def _extract(self, context: ExtractionContext, params: LayeredRegexParams) -> RawValue:
        found = self._found(context, params)
        if any(name not in found for name in params.require):
            return None
        return found.get(context.selector or "")

    def _found(self, context: ExtractionContext, params: LayeredRegexParams) -> dict[str, str]:
        found: dict[str, str] = {}
        for layer in params.layers:
            text = self._layer_text(context, layer)
            if params.strip_quoted is not None:
                text = _strip_quoted(text, params.strip_quoted.open, params.strip_quoted.close)
            text = text[: params.max_chars]
            hits: dict[str, str] = {}
            for name, spec in params.keys.items():
                match = scan_regex(spec.pattern, regex_flags(spec.flags)).search(text)
                if match and match.group(1):
                    hits[name] = match.group(1).upper() if spec.upper else match.group(1)
            found.update(hits)
            for rule in params.reset:
                if rule.when in hits and rule.unless not in hits:
                    for name in rule.clear:
                        found.pop(name, None)
        return found

    @staticmethod
    def _layer_text(context: ExtractionContext, layer: LayerParams) -> str:
        if layer.group is not None:
            values = context.draft.groups.get(layer.group)
            return values[0] if values else ""
        source = context.parsed.document_metadata.get(layer.field or "")
        if source is None and layer.field == "title":
            source = context.parsed.title
        return _text_of(source)


@lru_cache(maxsize=16)
def _quoted_span(opening: str, closing: str) -> re.Pattern[str]:
    return re.compile(f"{re.escape(opening)}[^{re.escape(closing)}]*{re.escape(closing)}")


def _strip_quoted(text: str, opening: str, closing: str) -> str:
    """Text outside `opening`...`closing` quotes: an unclosed quote hides everything after it."""
    if text.count(opening) > text.count(closing):
        text = text.split(opening)[0]
    return _quoted_span(opening, closing).sub(" ", text)
