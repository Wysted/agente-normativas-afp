"""A small CSS selector subset compiled to XPath 1.0 (evaluated by libxml2).

Supported: `tag`, `*`, `#id`, `.class`, `[attr]`, `[attr=v]`, `[attr*=v]`, `[attr^=v]`,
`[attr$=v]` (quoted or bare values), compounds of those, the descendant (` `) and child (`>`)
combinators and comma lists. Anything else is a `ValueError` at profile load. Like CSS, an
empty value never matches with `*=`, `^=` or `$=`; the `style` attribute is compared without
whitespace (`display: none` equals `display:none`).

Values travel as XPath variables, never spliced into the expression, so no value can break out
of it. Chains become step paths (`descendant::div/child::p`), whose cost is linear in the tree
per step: a selector that matches nothing cannot blow up on deeply nested markup.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final, cast

from lxml import etree

from agintis.modules.knowledge.infrastructure.html.dom import HtmlElement

_TOKEN = re.compile(
    r"""
    (?P<id>\#[\w-]+)
  | (?P<class>\.[\w-]+)
  | \[\s*(?P<attr>[\w:-]+)\s*
      (?:(?P<op>[*^$]?=)\s*(?:"(?P<dq>[^"]*)"|'(?P<sq>[^']*)'|(?P<bare>[^\]\s]*)))?
      \s*\]
  | (?P<tag>\*|[A-Za-z][\w-]*)
    """,
    re.VERBOSE,
)
# XPath 1.0 whitespace set used to compare `style` values.
_XPATH_SPACES: Final = " \t\n\r"
_SPACES = re.compile(r"\s+")


@dataclass(frozen=True)
class _AttrTest:
    name: str
    op: str | None
    value: str


@dataclass(frozen=True)
class _Compound:
    tag: str | None = None
    ids: tuple[str, ...] = ()
    classes: tuple[str, ...] = ()
    attrs: tuple[_AttrTest, ...] = ()


# A chain is [(combinator-to-previous, compound), ...]; the first combinator is unused.
type _Chain = tuple[tuple[str, _Compound], ...]


@dataclass(frozen=True, eq=False)
class Selector:
    """A compiled selector list; build with `parse_selector` (validates at boot)."""

    source: str
    xpath: etree.XPath = field(repr=False)
    variables: dict[str, str] = field(repr=False)

    def select(self, root: HtmlElement) -> list[HtmlElement]:
        """Descendants of `root` matching the selector, in document order, without duplicates."""
        return cast(list[HtmlElement], self.xpath(root, **self.variables))


def _parse_compound(text: str, whole: str) -> _Compound:
    tag: str | None = None
    ids: list[str] = []
    classes: list[str] = []
    attrs: list[_AttrTest] = []
    position = 0
    while position < len(text):
        start = position
        match = _TOKEN.match(text, position)
        if match is None or match.end() == position:
            raise ValueError(f"unsupported CSS selector syntax in {whole!r} near {text[start:]!r}")
        position = match.end()
        if match.group("id"):
            ids.append(match.group("id")[1:])
        elif match.group("class"):
            classes.append(match.group("class")[1:])
        elif match.group("attr"):
            value = next(
                (match.group(g) for g in ("dq", "sq", "bare") if match.group(g) is not None), ""
            )
            attrs.append(_AttrTest(match.group("attr").lower(), match.group("op"), value))
        elif match.group("tag") and tag is None and start == 0:
            tag = match.group("tag").lower()
        else:
            raise ValueError(
                f"unsupported CSS selector syntax in {whole!r} near {match.group(0)!r}"
            )
    return _Compound(tag if tag != "*" else None, tuple(ids), tuple(classes), tuple(attrs))


def _split_top_level(text: str, separator: str) -> list[str]:
    """Split on `separator` outside [...] and quotes."""
    parts: list[str] = []
    depth = 0
    quote = ""
    current: list[str] = []
    for ch in text:
        if quote:
            quote = "" if ch == quote else quote
        elif ch in "\"'":
            quote = ch
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        if ch == separator and depth == 0 and not quote:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    parts.append("".join(current))
    return parts


def _parse_chain(text: str, whole: str) -> _Chain:
    # Normalise child combinators so splitting on whitespace yields compounds and ">" markers.
    spaced = re.sub(r"\s*>\s*(?![^\[]*\])", " > ", text.strip())
    words = [w for w in _split_top_level(spaced, " ") if w]
    chain: list[tuple[str, _Compound]] = []
    combinator = " "
    for word in words:
        if word == ">":
            combinator = ">"
            continue
        chain.append((combinator, _parse_compound(word, whole)))
        combinator = " "
    if not chain or combinator == ">":
        raise ValueError(f"empty or dangling CSS selector {whole!r}")
    return tuple(chain)


class _XPathBuilder:
    def __init__(self) -> None:
        self.variables: dict[str, str] = {}

    def _var(self, value: str) -> str:
        name = f"v{len(self.variables)}"
        self.variables[name] = value
        return f"${name}"

    def _attr_predicate(self, test: _AttrTest) -> str:
        attribute = f"@*[name()={self._var(test.name)}]"
        if test.op is None:
            return attribute
        value, subject = test.value, "."
        if test.name == "style":
            value = _SPACES.sub("", value)
            subject = f"translate(., '{_XPATH_SPACES}', '')"
        if test.op == "=":
            return f"{attribute}[{subject}={self._var(value)}]"
        if not value:
            return "false()"  # as in CSS, `[a*=""]`, `[a^=""]`, `[a$=""]` match nothing
        wanted = self._var(value)
        match test.op:
            case "*=":
                condition = f"contains({subject},{wanted})"
            case "^=":
                condition = f"starts-with({subject},{wanted})"
            case _:  # "$="
                condition = (
                    f"substring({subject},string-length({subject})-string-length({wanted})+1)"
                    f"={wanted}"
                )
        return f"{attribute}[{condition}]"

    def _step(self, axis: str, compound: _Compound) -> str:
        predicates = [f"@id={self._var(value)}" for value in compound.ids]
        predicates += [
            f"contains(concat(' ',normalize-space(@class),' '),{self._var(f' {name} ')})"
            for name in compound.classes
        ]
        predicates += [self._attr_predicate(test) for test in compound.attrs]
        return f"{axis}::{compound.tag or '*'}" + "".join(f"[{p}]" for p in predicates)

    def path(self, chain: _Chain) -> str:
        steps = [
            self._step("child" if index and combinator == ">" else "descendant", compound)
            for index, (combinator, compound) in enumerate(chain)
        ]
        return "/".join(steps)


def parse_selector(selector: str) -> Selector:
    """Parse and compile a selector list (`a, b > c.d[x*="y"]`); `ValueError` if unsupported."""
    if not selector.strip():
        raise ValueError("CSS selector must not be empty")
    chains: Sequence[_Chain] = [
        _parse_chain(part, selector) for part in _split_top_level(selector, ",")
    ]
    builder = _XPathBuilder()
    expression = " | ".join(builder.path(chain) for chain in chains)
    try:
        compiled = etree.XPath(expression)
    except etree.XPathSyntaxError as exc:  # pragma: no cover - the grammar above is closed
        raise ValueError(f"unsupported CSS selector syntax in {selector!r}") from exc
    return Selector(selector, compiled, builder.variables)


def select_all(root: HtmlElement, selectors: Sequence[Selector]) -> list[HtmlElement]:
    """Descendants of `root` matching any selector, in document order, without duplicates."""
    found: dict[HtmlElement, None] = {}
    for selector in selectors:
        for element in selector.select(root):
            found.setdefault(element, None)
    if len(selectors) > 1:
        order = {element: index for index, element in enumerate(root.iter())}
        return sorted(found, key=lambda element: order[element])
    return list(found)
