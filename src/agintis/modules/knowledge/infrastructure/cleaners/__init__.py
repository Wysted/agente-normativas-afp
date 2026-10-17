"""Cleaner adapters (profile `cleaners` pipeline)."""

from agintis.modules.knowledge.infrastructure.cleaners.base import ParamsCleaner, Pattern
from agintis.modules.knowledge.infrastructure.cleaners.html import HtmlCollect, HtmlMainContent
from agintis.modules.knowledge.infrastructure.cleaners.pages import (
    DropPagesBefore,
    DropTocPages,
    StripRepeatedPageLines,
    StripRunningHeaders,
)
from agintis.modules.knowledge.infrastructure.cleaners.text import (
    DedupeHeadingEcho,
    JoinContinuationLines,
    NormalizeQuotes,
    NormalizeWhitespace,
    StripLines,
)

CLEANER_CLASSES: tuple[type[ParamsCleaner], ...] = (  # type: ignore[type-arg]
    StripRunningHeaders,
    StripRepeatedPageLines,
    StripLines,
    DropTocPages,
    DropPagesBefore,
    DedupeHeadingEcho,
    NormalizeQuotes,
    NormalizeWhitespace,
    JoinContinuationLines,
    HtmlMainContent,
    HtmlCollect,
)

__all__ = [
    "CLEANER_CLASSES",
    "DedupeHeadingEcho",
    "DropPagesBefore",
    "DropTocPages",
    "HtmlCollect",
    "HtmlMainContent",
    "JoinContinuationLines",
    "NormalizeQuotes",
    "NormalizeWhitespace",
    "ParamsCleaner",
    "Pattern",
    "StripLines",
    "StripRepeatedPageLines",
    "StripRunningHeaders",
]
