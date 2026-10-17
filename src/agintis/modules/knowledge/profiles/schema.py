"""YAML shape of a profile file (Pydantic) and its conversion to the domain `RagProfile`.

Strategy-specific parameters stay raw here: their schemas belong to the strategies and are
checked by `validate_profiles`, which is also the place that rejects unknown keys. Unknown
keys of the file itself are rejected right away.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from agintis.modules.knowledge.application import NO_RERANKER_KEY
from agintis.modules.knowledge.domain import (
    DENSE_RETRIEVAL_MODE,
    CleanerSpec,
    ExtractorSpec,
    MetadataField,
    MetadataFieldType,
    ProfileStatus,
    RagProfile,
    RoutingCondition,
    RoutingRule,
    TransformSpec,
)
from agintis.modules.knowledge.profiles.params import Pattern


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class TransformDocument(_Strict):
    key: str
    params: dict[str, Any] = Field(default_factory=dict)

    def to_domain(self) -> TransformSpec:
        return TransformSpec(self.key, self.params)


class ExtractorDocument(_Strict):
    key: str
    params: dict[str, Any] = Field(default_factory=dict)

    def to_domain(self) -> ExtractorSpec:
        return ExtractorSpec(self.key, self.params)


class MetadataFieldDocument(_Strict):
    name: str
    type: MetadataFieldType = "str"
    filterable: bool = False
    indexed: bool = False
    source: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    transforms: list[TransformDocument] = Field(default_factory=list)

    def to_domain(self) -> MetadataField:
        return MetadataField(
            name=self.name,
            type=self.type,
            filterable=self.filterable,
            indexed=self.indexed,
            source=self.source,
            params=self.params,
            transforms=tuple(t.to_domain() for t in self.transforms),
        )


class RegexSpec(_Strict):
    """`{regex: <pattern>}`: the field's text must contain a match."""

    regex: Pattern


type RoutingValue = str | int | list[str | int] | RegexSpec


def _condition(field: str, value: RoutingValue) -> RoutingCondition:
    if isinstance(value, RegexSpec):
        return RoutingCondition(field, pattern=value.regex)
    values = value if isinstance(value, list) else [value]
    return RoutingCondition(field, any_of=tuple(str(item) for item in values))


class RoutingRuleDocument(_Strict):
    """Which documents the profile is meant for, by their document metadata.

    `match`: every listed field must hold (a scalar or a list = one of these values, or
    `{regex: ...}`); `exclude`: no listed field may hold. `priority` (default 0) ranks rules that
    match the same document: the highest wins, a tie between profiles leaves it unrouted.
    """

    match: dict[str, RoutingValue] = Field(min_length=1)
    exclude: dict[str, RoutingValue] = Field(default_factory=dict)
    priority: int = 0

    def to_domain(self) -> RoutingRule:
        return RoutingRule(
            match=tuple(_condition(f, v) for f, v in self.match.items()),
            exclude=tuple(_condition(f, v) for f, v in self.exclude.items()),
            priority=self.priority,
        )


class CleanerDocument(_Strict):
    key: str
    params: dict[str, Any] = Field(default_factory=dict)

    def to_domain(self) -> CleanerSpec:
        return CleanerSpec(self.key, self.params)


class ProfileDocument(_Strict):
    """One profile file. `key` is the profile name; all strategy keys are registry keys."""

    schema_version: Literal[1] = 1
    key: str
    top_k: int
    chunk_max_tokens: int
    parser_key: str
    chunker_key: str
    chunker_params: dict[str, Any] = Field(default_factory=dict)
    cleaners: list[CleanerDocument] = Field(default_factory=list)
    metadata_fields: list[MetadataFieldDocument] = Field(default_factory=list)
    default_filter: None = None
    reranker_key: str = NO_RERANKER_KEY
    retrieval_mode: str = DENSE_RETRIEVAL_MODE
    routing: list[RoutingRuleDocument] = Field(default_factory=list)
    status: ProfileStatus = "active"
    extractors: dict[str, ExtractorDocument] = Field(default_factory=dict)

    def to_domain(self) -> RagProfile:
        return RagProfile(
            name=self.key,
            top_k=self.top_k,
            chunk_max_tokens=self.chunk_max_tokens,
            parser_key=self.parser_key,
            chunker_key=self.chunker_key,
            chunker_params=self.chunker_params,
            default_filter=self.default_filter,
            reranker_key=self.reranker_key,
            retrieval_mode=self.retrieval_mode,
            metadata_fields=tuple(f.to_domain() for f in self.metadata_fields),
            cleaners=tuple(c.to_domain() for c in self.cleaners),
            routing=tuple(r.to_domain() for r in self.routing),
            status=self.status,
            extractors={name: spec.to_domain() for name, spec in self.extractors.items()},
        )
