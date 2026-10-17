"""Metadata extractors and value transforms (whitelisted by key, configured by profile params)."""

from agintis.modules.knowledge.infrastructure.extractors.base import (
    NoParams,
    ParamsExtractor,
    ParamsTransform,
)
from agintis.modules.knowledge.infrastructure.extractors.generic import (
    Classify,
    DateAfterHeading,
    LayeredRegex,
    RegexAll,
    RegexLabeled,
    Template,
)
from agintis.modules.knowledge.infrastructure.extractors.transforms import (
    DateFloor,
    PathSegment,
    PrefixMap,
    RejectAfter,
    ValueMap,
)

EXTRACTOR_CLASSES: tuple[type[ParamsExtractor], ...] = (  # type: ignore[type-arg]
    RegexAll,
    Classify,
    Template,
    LayeredRegex,
    RegexLabeled,
    DateAfterHeading,
)
TRANSFORM_CLASSES: tuple[type[ParamsTransform], ...] = (  # type: ignore[type-arg]
    ValueMap,
    DateFloor,
    RejectAfter,
    PathSegment,
    PrefixMap,
)

__all__ = [
    "EXTRACTOR_CLASSES",
    "TRANSFORM_CLASSES",
    "Classify",
    "DateAfterHeading",
    "DateFloor",
    "LayeredRegex",
    "NoParams",
    "ParamsExtractor",
    "ParamsTransform",
    "PathSegment",
    "PrefixMap",
    "RegexAll",
    "RegexLabeled",
    "RejectAfter",
    "Template",
    "ValueMap",
]
