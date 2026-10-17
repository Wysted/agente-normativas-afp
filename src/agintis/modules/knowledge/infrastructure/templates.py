"""Tiny string templates for headings (`heading_template`, `heading_prefix_template`, ...).

Syntax
* `{name}`         the value of `name` (empty when missing).
* `{name:50}`      the value cut to 50 characters, ending in an ellipsis when cut (the limit is
                   at least 1; `{name:0}` is not a placeholder and is rejected at boot).
* `{name?text}`    `text` (which may hold plain `{x}` / `{x:n}` placeholders) only when `name`
                   has a value (any non-empty text, `0` included); conditionals do not nest.
Runs of whitespace collapse to one space and the result is stripped, so an absent optional part
leaves no hole.

Rendering is a SINGLE pass over the template: what a placeholder produces is never scanned for
placeholders again, so a value such as `{secret}` coming from a document stays literal text.
"""

import re
from collections.abc import Mapping
from typing import Final

_PLACEHOLDER: Final = re.compile(
    r"\{(?P<cond>\w+)\?(?P<body>(?:[^{}]|\{\w+(?::[1-9]\d*)?\})*)\}"
    r"|\{(?P<name>\w+)(?::(?P<limit>[1-9]\d*))?\}"
)
_VARIABLE: Final = re.compile(r"\{(\w+)(?::([1-9]\d*))?\}")
_SPACES: Final = re.compile(r"\s{2,}")
# Anything that looks like a placeholder, used to reject malformed ones at boot.
_BRACES: Final = re.compile(r"[{}]")


def render_template(template: str, values: Mapping[str, object]) -> str:
    return _SPACES.sub(" ", _PLACEHOLDER.sub(lambda m: _expand(m, values), template)).strip()


def _present(values: Mapping[str, object], name: str) -> bool:
    raw = values.get(name)
    return raw is not None and str(raw) != ""


def _expand(match: re.Match[str], values: Mapping[str, object]) -> str:
    if (cond := match.group("cond")) is not None:
        if not _present(values, cond):
            return ""
        return _VARIABLE.sub(lambda m: _value(m.group(1), m.group(2), values), match["body"])
    return _value(match["name"], match["limit"], values)


def _value(name: str, limit: str | None, values: Mapping[str, object]) -> str:
    raw = values.get(name)
    text = "" if raw is None else str(raw)
    if limit is not None and len(text) > (size := int(limit)):
        text = text[: size - 1].rstrip() + "…"
    return text


def template_names(template: str) -> frozenset[str]:
    """Every variable the template reads (conditions and placeholders)."""
    names: set[str] = set()
    for match in _PLACEHOLDER.finditer(template):
        if (cond := match.group("cond")) is not None:
            names.add(cond)
            names.update(m.group(1) for m in _VARIABLE.finditer(match["body"]))
        else:
            names.add(match["name"])
    return frozenset(names)


def check_template_syntax(template: str) -> str:
    """Return `template` unchanged, or raise `ValueError` when it has a stray or malformed brace.

    Everything left once the valid placeholders are removed must be brace-free (so `{x:0}`,
    `{a?{b?c}}` or an unclosed `{x` fail at boot instead of rendering literally).
    """
    leftover = _PLACEHOLDER.sub("", template)
    if _BRACES.search(leftover):
        raise ValueError(
            f"template {template!r} has a malformed placeholder "
            "(use {name}, {name:N} with N >= 1, or {name?text}; conditionals do not nest)"
        )
    return template
