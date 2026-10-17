"""Resolve a profile's declared metadata fields for every chunk of a document.

A field's `source` says where its value comes from (see `MetadataField`):

* `group:<name>`     named regex group the chunker recorded in `ChunkDraft.groups`
* `document:<field>` caller-supplied `SourceFile.document_metadata`, as seen by the parsed document
* `parsed:<key>`     value a cleaner collected (`ParsedDocument.collected`)
* `const:<value>`    the literal text after the prefix
* `extractor:<key>[.<selector>]` a registered `Extractor` fed with the field's `params`

The raw value is shaped to the field (a list field takes all values, a scalar field the first),
passed through the field's `transforms` in order and dropped when empty; type checking is left to
`conform_metadata`. Fields resolve in declaration order and an extractor can read the ones before
it (`ExtractionContext.resolved`), which is how a path is built from already extracted parts.

Extractors and transforms are registered strategies (keys in a whitelist registry, never import
paths); their `version` joins the profile fingerprint like any other strategy version.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date
from types import MappingProxyType
from typing import ClassVar, cast

from agintis.modules.knowledge.application.dto import ChunkDraft, DocumentValue, ParsedDocument
from agintis.modules.knowledge.application.errors import UnknownStrategyError
from agintis.modules.knowledge.domain import (
    InvalidMetadataError,
    MetadataField,
    MetadataInput,
    MetadataValue,
    ParamValue,
    RagProfile,
    TransformSpec,
)
from agintis.shared_kernel.ports import Clock

type RawValue = DocumentValue


@dataclass(frozen=True)
class ExtractionContext:
    """Everything an extractor may read for ONE chunk.

    `memo` is shared by all the chunks of a document: use it to compute document-level values
    once (e.g. the effective date of a whole norm) instead of once per chunk.
    """

    draft: ChunkDraft
    parsed: ParsedDocument
    document_text: str
    resolved: Mapping[str, MetadataInput]
    today: date
    selector: str | None = None
    memo: dict[str, object] = field(default_factory=dict)


class Extractor(ABC):
    """Computes a raw metadata value from a chunk and its document (a small piece of code)."""

    key: ClassVar[str]
    version: ClassVar[int] = 1  # see `Parser.version`

    def selectors_for(self, params: Mapping[str, ParamValue]) -> frozenset[str]:
        """Sub-values this extractor yields for `params` (`extractor:key.<selector>`).

        Empty = it takes no selector. Profiles naming any other selector are rejected at boot.
        """
        return frozenset()

    def references(self, params: Mapping[str, ParamValue]) -> frozenset[str]:
        """Names this extractor reads from the chunk's regex groups and from metadata fields
        declared BEFORE it (empty = reads neither). Profiles naming anything else are rejected
        at boot."""
        return frozenset()

    @abstractmethod
    def extract(self, context: ExtractionContext, params: Mapping[str, ParamValue]) -> RawValue: ...


class ValueTransform(ABC):
    """Rewrites a raw metadata value (value maps, date flooring, ...); None drops the field."""

    key: ClassVar[str]
    version: ClassVar[int] = 1  # see `Parser.version`

    @abstractmethod
    def apply(self, value: RawValue, params: Mapping[str, ParamValue], today: date) -> RawValue: ...


def _is_empty(value: RawValue) -> bool:
    return value is None or value == "" or value == ()


def _shape(value: RawValue, *, is_list: bool) -> RawValue:
    """A list field holds a tuple, a scalar field its first value."""
    if value is None:
        return None
    if is_list:
        return value if isinstance(value, tuple) else (value,)
    if isinstance(value, tuple):
        return value[0] if value else None
    return value


def _to_input(value: RawValue) -> MetadataInput:
    """The resolved value as chunk metadata input (tuples stay tuples; type check is later)."""
    if isinstance(value, tuple):
        return tuple(item for item in value if item is not None)  # type: ignore[return-value]
    return value  # type: ignore[return-value]


def _resolve_extractor_ref(
    profile: RagProfile, declared: MetadataField
) -> tuple[str, Mapping[str, ParamValue], str | None]:
    """(registry key, params, selector) of an `extractor:<name>[.<selector>]` source.

    `<name>` is first looked up among the profile's named `extractors`; otherwise it is a
    registry key configured by the field's own `params`.
    """
    name, _, selector = (declared.source or "").removeprefix("extractor:").partition(".")
    named = profile.extractors.get(name)
    if named is not None:
        return named.key, named.params, selector or None
    return name, declared.params, selector or None


class MetadataResolver:
    def __init__(
        self,
        extractors: Mapping[str, Extractor],
        transforms: Mapping[str, ValueTransform],
        clock: Clock,
    ) -> None:
        self._extractors = extractors
        self._transforms = transforms
        self._clock = clock

    def versions(self, profile: RagProfile) -> dict[str, int]:
        """Implementation versions of the extractors and transforms the profile uses."""
        versions: dict[str, int] = {}
        for declared in profile.metadata_fields:
            if declared.source and declared.source.startswith("extractor:"):
                key, _, _ = _resolve_extractor_ref(profile, declared)
                versions[f"extractor:{key}"] = self._extractor(key).version
            for spec in declared.transforms:
                versions[f"transform:{spec.key}"] = self._transform(spec.key).version
        return versions

    def resolve(
        self, profile: RagProfile, parsed: ParsedDocument, drafts: Sequence[ChunkDraft]
    ) -> list[ChunkDraft]:
        fields = [f for f in profile.metadata_fields if f.source is not None]
        if not fields:
            return list(drafts)
        today = self._clock.now().date()
        document_text = parsed.joined().text
        memo: dict[str, object] = {}
        resolved_drafts: list[ChunkDraft] = []
        for draft in drafts:
            resolved: dict[str, MetadataInput] = {}
            for declared in fields:
                value = self._value(
                    profile, declared, draft, parsed, document_text, resolved, today, memo
                )
                if not _is_empty(value):
                    resolved[declared.name] = _to_input(value)
            merged: dict[str, MetadataInput] = {**draft.metadata, **resolved}
            # ChunkDraft freezes (and so validates) whatever it is given.
            resolved_drafts.append(
                replace(draft, metadata=cast(Mapping[str, MetadataValue], merged))
            )
        return resolved_drafts

    def _value(
        self,
        profile: RagProfile,
        declared: MetadataField,
        draft: ChunkDraft,
        parsed: ParsedDocument,
        document_text: str,
        resolved: Mapping[str, MetadataInput],
        today: date,
        memo: dict[str, object],
    ) -> RawValue:
        source = declared.source or ""
        kind, _, name = source.partition(":")
        raw: RawValue
        if kind == "group":
            raw = draft.groups.get(name)
            raw = tuple(raw) if raw else None
        elif kind == "document":
            raw = parsed.document_metadata.get(name)
        elif kind == "parsed":
            collected = parsed.collected.get(name)
            raw = tuple(collected) if collected else None
        elif kind == "const":
            raw = name
        else:  # extractor
            key, params, selector = _resolve_extractor_ref(profile, declared)
            context = ExtractionContext(
                draft,
                parsed,
                document_text,
                MappingProxyType(dict(resolved)),  # a snapshot: later fields do not leak in
                today,
                selector,
                memo,
            )
            raw = self._extractor(key).extract(context, params)
        value = _shape(raw, is_list=declared.is_list)
        for spec in declared.transforms:
            if _is_empty(value):
                break
            value = _shape(self._apply(declared, spec, value, today), is_list=declared.is_list)
        return value

    def _apply(
        self, declared: MetadataField, spec: TransformSpec, value: RawValue, today: date
    ) -> RawValue:
        """Run one transform; a value it cannot handle is a bad document, never a crash."""
        try:
            return self._transform(spec.key).apply(value, spec.params, today)
        except (TypeError, ValueError, AttributeError, OverflowError) as exc:
            raise InvalidMetadataError(
                f"Metadata field {declared.name!r}: transform {spec.key!r} cannot handle "
                f"{value!r} ({type(exc).__name__}: {exc})"
            ) from exc

    def _extractor(self, key: str) -> Extractor:
        try:
            return self._extractors[key]
        except KeyError:
            raise UnknownStrategyError(f"No extractor registered for key {key!r}") from None

    def _transform(self, key: str) -> ValueTransform:
        try:
            return self._transforms[key]
        except KeyError:
            raise UnknownStrategyError(f"No transform registered for key {key!r}") from None
