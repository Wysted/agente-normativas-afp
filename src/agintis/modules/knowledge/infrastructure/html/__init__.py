"""Shared HTML toolkit (lxml): parsing, CSS-subset selectors and block extraction.

Public within the infrastructure layer: parsers and cleaners both build on it.
"""

from agintis.modules.knowledge.infrastructure.html.blocks import (
    BOILERPLATE,
    HARD_SKIP,
    Block,
    BlockOptions,
    blocks_to_sections,
    collect_blocks,
    normalize_space,
)
from agintis.modules.knowledge.infrastructure.html.dom import (
    HtmlElement,
    drop_subtrees,
    has_ancestor_in,
    parse_html_tree,
    text_content,
)
from agintis.modules.knowledge.infrastructure.html.selectors import (
    Selector,
    parse_selector,
    select_all,
)

__all__ = [
    "BOILERPLATE",
    "HARD_SKIP",
    "Block",
    "BlockOptions",
    "HtmlElement",
    "Selector",
    "blocks_to_sections",
    "collect_blocks",
    "drop_subtrees",
    "has_ancestor_in",
    "normalize_space",
    "parse_html_tree",
    "parse_selector",
    "select_all",
    "text_content",
]
