"""Filter AST: leaf comparisons plus AND / OR composites.

Reference semantics (the in-memory evaluator in `in_memory.py` is the oracle; every
backend converter MUST emit these explicitly, never rely on the backend's default)

- Missing field or NULL value: only NEQ and NOT_IN match. SQL/PGVector converters emit
  `col IS NULL OR col <> :v` (and `IS NULL OR NOT IN`); Qdrant `must_not` already behaves so.
  EQ, IN and the ordering operators never match a missing value.
- Types are strict: bool is distinct from int; int and float compare numerically; str and
  datetime only compare with their own kind. Incompatible operands never match for EQ, IN and
  ordering operators (no TypeError), and always differ for NEQ / NOT_IN.
- List-valued fields (JSON arrays) behave as the SET of their items: EQ matches when some item
  equals the value, IN when some item equals some value (overlap); NEQ / NOT_IN are the exact
  negations (an empty list and a missing field both match them); ordering operators never
  match. Items follow the same strict typing as scalars.
- IN / NOT_IN tuples are homogeneous (all str, all int, all float, all bool or all datetime).
- Field names match `FIELD_NAME_PATTERN`; each repository declares its FILTERABLE_FIELDS and
  rejects others (`ensure_fields`). Converters raise `UnsupportedFilterError` for anything
  they cannot translate faithfully instead of approximating it.
- Ordering (S5): converters sort NULLS LAST for both directions and break ties by `id` so
  pagination is stable.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

type Scalar = str | int | float | bool | datetime
type FilterValue = Scalar | tuple[Scalar, ...]

_SCALAR_TYPES = (str, int, float, bool, datetime)

FIELD_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def is_valid_field_name(name: str) -> bool:
    return FIELD_NAME_PATTERN.fullmatch(name) is not None


def value_kind(value: object) -> str:
    """Return the strict kind of a value: bool, int, float, str, datetime or the type name."""
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, datetime):
        return "datetime"
    return type(value).__name__


class FilterOperator(StrEnum):
    EQ = "eq"
    NEQ = "neq"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    IN = "in"
    NOT_IN = "not_in"


_MULTI_VALUED = frozenset({FilterOperator.IN, FilterOperator.NOT_IN})


def _is_scalar(value: object) -> bool:
    return isinstance(value, _SCALAR_TYPES)


@dataclass(frozen=True)
class Filter:
    """A single `field <operator> value` comparison."""

    field: str
    operator: FilterOperator
    value: FilterValue

    def __post_init__(self) -> None:
        if not is_valid_field_name(self.field):
            raise ValueError(
                f"Filter field {self.field!r} is invalid (must match {FIELD_NAME_PATTERN.pattern})"
            )
        if self.operator in _MULTI_VALUED:
            if not isinstance(self.value, tuple) or not self.value:
                raise ValueError(f"{self.operator.name} requires a non-empty tuple of scalars")
            if not all(_is_scalar(item) for item in self.value):
                raise ValueError(f"{self.operator.name} values must all be scalars")
            if len({value_kind(item) for item in self.value}) != 1:
                raise ValueError(f"{self.operator.name} values must all be of the same type")
        elif not _is_scalar(self.value):
            raise ValueError(f"{self.operator.name} requires a scalar value")

    @classmethod
    def eq(cls, field: str, value: Scalar) -> Filter:
        return cls(field, FilterOperator.EQ, value)

    @classmethod
    def neq(cls, field: str, value: Scalar) -> Filter:
        return cls(field, FilterOperator.NEQ, value)

    @classmethod
    def gt(cls, field: str, value: Scalar) -> Filter:
        return cls(field, FilterOperator.GT, value)

    @classmethod
    def gte(cls, field: str, value: Scalar) -> Filter:
        return cls(field, FilterOperator.GTE, value)

    @classmethod
    def lt(cls, field: str, value: Scalar) -> Filter:
        return cls(field, FilterOperator.LT, value)

    @classmethod
    def lte(cls, field: str, value: Scalar) -> Filter:
        return cls(field, FilterOperator.LTE, value)

    @classmethod
    def in_(cls, field: str, values: Iterable[Scalar]) -> Filter:
        return cls(field, FilterOperator.IN, tuple(values))

    @classmethod
    def not_in(cls, field: str, values: Iterable[Scalar]) -> Filter:
        return cls(field, FilterOperator.NOT_IN, tuple(values))


@dataclass(frozen=True, init=False)
class AllOf:
    """Conjunction: every expression must match."""

    exprs: tuple[FilterExpr, ...]

    def __init__(self, *exprs: FilterExpr) -> None:
        if not exprs:
            raise ValueError("AllOf requires at least one expression")
        object.__setattr__(self, "exprs", exprs)


@dataclass(frozen=True, init=False)
class AnyOf:
    """Disjunction: at least one expression must match."""

    exprs: tuple[FilterExpr, ...]

    def __init__(self, *exprs: FilterExpr) -> None:
        if not exprs:
            raise ValueError("AnyOf requires at least one expression")
        object.__setattr__(self, "exprs", exprs)


type FilterExpr = Filter | AllOf | AnyOf
