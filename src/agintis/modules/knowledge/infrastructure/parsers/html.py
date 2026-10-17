"""HTML source: generic main-text extraction (lxml).

Without profile cleaners the whole `<body>` is converted, dropping scripts, styles and the
semantic page furniture (`nav`, `footer`, `aside`). Sites with their own boilerplate declare
the `html_main_content` cleaner (CSS selectors) which rebuilds the sections from the raw
markup kept in `ParsedDocument.source_html`.
"""

from agintis.modules.knowledge.application import ParsedDocument
from agintis.modules.knowledge.infrastructure.html import (
    BOILERPLATE,
    HARD_SKIP,
    BlockOptions,
    blocks_to_sections,
    collect_blocks,
    normalize_space,
    parse_html_tree,
    text_content,
)
from agintis.modules.knowledge.infrastructure.parsers._title import title_from_uri
from agintis.modules.knowledge.infrastructure.parsers.limits import ParseContext
from agintis.modules.knowledge.infrastructure.text import decode_text


def parse_html(content: bytes, uri: str, context: ParseContext) -> ParsedDocument:
    html = decode_text(content, context.charset, sniff_meta=True)
    root = parse_html_tree(html)
    body = root.find("body")
    title_node = next(root.iter("title"), None)
    title_text = normalize_space(text_content(title_node)) if title_node is not None else ""
    options = BlockOptions(skip_tags=HARD_SKIP | BOILERPLATE)
    first_heading, sections = blocks_to_sections(
        collect_blocks([body if body is not None else root], options)
    )
    title = title_text or first_heading or title_from_uri(uri)
    return ParsedDocument(title=title, sections=sections, source_html=html)
