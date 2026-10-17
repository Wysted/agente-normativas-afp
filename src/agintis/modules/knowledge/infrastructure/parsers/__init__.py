"""Parser adapters."""

from agintis.modules.knowledge.infrastructure.parsers.index_card import (
    INDEX_CARD_PARSER_KEY,
    IndexCardParser,
)
from agintis.modules.knowledge.infrastructure.parsers.limits import ParseLimits
from agintis.modules.knowledge.infrastructure.parsers.simple import SIMPLE_PARSER_KEY, SimpleParser

__all__ = [
    "INDEX_CARD_PARSER_KEY",
    "SIMPLE_PARSER_KEY",
    "IndexCardParser",
    "ParseLimits",
    "SimpleParser",
]
