"""HTML parsing on top of lxml (libxml2's HTML4 parser) plus tree helpers.

libxml2 applies the HTML implied-end-tag rules (`<p>` closed by a block start, `<li>`, `<td>`,
`<tr>`, omitted `</head>` / `<body>`) and keeps `<title>` and other head content out of `<body>`,
so none of that is re-implemented here. The parser never touches the network and drops
comments and processing instructions; `script`, `style`, `noscript` and `template` elements are
removed from the tree right after parsing (their tail text is kept).

Known limits of the engine (hostile input only): nesting deeper than 2048 levels aborts the
parse and drops the rest of the document, and a single start tag with thousands of attributes
is quadratic, so such tags are rejected up front (`MAX_TAG_ATTRIBUTES`).
"""

import re
from collections.abc import Iterable
from typing import Final

from lxml import etree
from lxml import html as lxml_html

from agintis.modules.knowledge.application import UnparseableDocumentError
from agintis.modules.knowledge.infrastructure.text import strip_unstorable

type HtmlElement = lxml_html.HtmlElement

# Never content, anywhere: removed from the tree at parse time.
DROPPED_TAGS: Final = ("script", "style", "noscript", "template")
# libxml2 checks every new attribute against the previous ones (quadratic per tag: 50k
# attributes take ~4 s, 200k over a minute). 4096 attributes are far beyond real markup.
MAX_TAG_ATTRIBUTES: Final = 4096
_MIN_ATTRIBUTE_BYTES: Final = 4  # `a=1 `
_LONG_TAG = re.compile(rf"<[A-Za-z][^<>]{{{MAX_TAG_ATTRIBUTES * _MIN_ATTRIBUTE_BYTES},}}")
_WHITESPACE_RUN = re.compile(r"\s+")


def _reject_attribute_floods(markup: str) -> None:
    for match in _LONG_TAG.finditer(markup):
        if len(_WHITESPACE_RUN.findall(match.group())) > MAX_TAG_ATTRIBUTES:
            raise UnparseableDocumentError(
                f"HTML start tag with more than {MAX_TAG_ATTRIBUTES} attributes"
            )


def parse_html_tree(markup: str) -> HtmlElement:
    """Parse `markup` into an `<html>` element (always present, possibly empty).

    NUL characters and lone surrogates are removed first (they cannot reach the database and
    NUL would truncate the document).
    """
    text = strip_unstorable(markup)
    _reject_attribute_floods(text)
    parser = lxml_html.HTMLParser(
        recover=True,
        no_network=True,
        remove_comments=True,
        remove_pis=True,
        # Without it, text nodes over 10 MB are silently dropped (the byte limit is enforced
        # by the parser limits before we get here).
        huge_tree=True,
        encoding="utf-8",
    )
    try:
        root = lxml_html.document_fromstring(text.encode("utf-8"), parser=parser)
    except etree.ParserError:  # empty document (blank, or only comments)
        return lxml_html.document_fromstring("<html></html>")
    except etree.XMLSyntaxError as exc:
        raise UnparseableDocumentError(f"Cannot parse HTML: {exc}") from exc
    etree.strip_elements(root, *DROPPED_TAGS, with_tail=False)
    return root


def drop_subtrees(elements: Iterable[HtmlElement]) -> None:
    """Remove each element with everything inside it, keeping the text that follows it.

    Because the whole subtree disappears, nodes inside an excluded element can neither be
    selected nor contribute text afterwards.
    """
    for element in list(elements):
        if element.getparent() is not None:
            element.drop_tree()


def has_ancestor_in(element: HtmlElement, others: set[HtmlElement]) -> bool:
    return any(ancestor in others for ancestor in element.iterancestors())


# Elements that start/end a line of text: a separator is inserted at their boundaries so
# `<td><p>a</p><p>b</p></td>` reads "a b", never "ab".
SEPARATING_TAGS: Final = frozenset(
    {"address", "article", "aside", "blockquote", "br", "caption", "dd", "details", "div", "dl",
     "dt", "fieldset", "figcaption", "figure", "footer", "form", "h1", "h2", "h3", "h4", "h5",
     "h6", "header", "hr", "li", "main", "nav", "ol", "p", "pre", "section", "summary", "table",
     "tbody", "td", "tfoot", "th", "thead", "tr", "ul"}
)  # fmt: skip


def text_content(element: HtmlElement, skip_tags: frozenset[str] = frozenset()) -> str:
    """Text of `element` (not its tail), with a space at block boundaries and `<br>`.

    Subtrees of `skip_tags` contribute nothing. Iterative: any nesting depth is safe.
    """
    parts: list[str] = []
    walker = etree.iterwalk(element, events=("start", "end"))
    for event, node in walker:
        tag = node.tag
        if not isinstance(tag, str):
            continue
        if event == "start":
            if tag in skip_tags:
                walker.skip_subtree()
                continue
            if tag in SEPARATING_TAGS:
                parts.append(" ")
            if node.text:
                parts.append(node.text)
        else:
            if tag in SEPARATING_TAGS:
                parts.append(" ")
            if node is not element and node.tail:
                parts.append(node.tail)
    return "".join(parts)
