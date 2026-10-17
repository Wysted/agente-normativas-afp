"""SQLAlchemy converter for the neutral Criteria AST (`agintis.shared_kernel.criteria`).

The in-memory evaluator (`shared_kernel/criteria/in_memory.py`) is the semantics oracle; this
module emits the same results in SQL, explicitly:

- Missing / NULL values match only NEQ and NOT_IN. Plain columns emit `col IS NULL OR NOT (...)`;
  JSONB-backed fields negate a containment test (`NOT (metadata @> ...)`), which is already true
  for a missing key.
- Strict types. A value whose kind does not fit the field's kind (a str against an int field, a
  bool against a number, a naive datetime against a timestamptz, ...) can never be equal:
  EQ / IN / ordering fold to FALSE, NEQ / NOT_IN fold to TRUE. int and float are one `number`
  kind and compare numerically (exact `numeric` arithmetic, floats go through their shortest
  repr so `0.1` is `0.1`).
- Text ordering uses `COLLATE "C"` (code-point order, like Python), not the database locale.
- Ordering puts NULLs last in both directions and breaks ties by `id`.

Safety: field names come from the caller's allowlist (`FieldSpec` mapping built by the
repository) and are re-validated against `FIELD_NAME_PATTERN`; values are ALWAYS bound
parameters. JSONB keys are rendered as inline literals (instead of bind parameters) so the
predicates textually match expression indexes; that is safe because of the pattern check.

JSONB equality uses containment (`metadata @> '{"k": v}'`): it is strict about types, and it is
the one operator a `jsonb_path_ops` GIN index can serve. List-valued fields (JSON arrays) are
sets: EQ is `metadata @> '{"k": [v]}'` and IN an OR-chain of those (the `?|` overlap operator
only matches strings and bypasses `jsonb_path_ops`), NEQ / NOT_IN their negation, ordering folds
to FALSE (the oracle never matches a list against an ordering operator).

Known divergence (data shape vs declared shape): the oracle decides by the SHAPE OF EACH STORED
VALUE (a list is a set, anything else a scalar), while this converter decides by the DECLARED
shape of the field (`JsonbField.array`). They only disagree for rows stored before a field was
retyped between scalar and list (`str` <-> `list[str]`, `int` <-> `list[int]`): the type is part
of the profile fingerprint, so the next ingest of each document rewrites it in the new shape, but
until then a stale row simply does not match through the new spec (`{"k": "v"}` never contains
`{"k": ["v"]}`, nor the reverse); NEQ / NOT_IN match it. Re-ingest the affected profile after
retyping a field.

Cardinality: IN / NOT_IN over `MAX_IN_VALUES` values raise `UnsupportedFilterError`. On JSONB
fields an IN is an OR of one containment per value (bound parameters and plan size grow with it),
and every filter value is bound, so an unbounded list could exhaust the driver's parameter limit.
"""

import math
import operator
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Literal

import sqlalchemy as sa
from sqlalchemy import ColumnElement
from sqlalchemy.dialects.postgresql import JSONB

from agintis.shared_kernel.criteria import (
    AllOf,
    AnyOf,
    Criteria,
    Filter,
    FilterExpr,
    FilterOperator,
    Order,
    OrderDirection,
    UnknownFieldError,
    UnsupportedFilterError,
)
from agintis.shared_kernel.criteria.filters import FIELD_NAME_PATTERN, Scalar, value_kind

type FieldKind = Literal["str", "number", "bool", "datetime"]

TEXT_COLLATION = "C"
TIE_BREAKER_FIELD = "id"
MAX_IN_VALUES = 256

_ORDERING: dict[FilterOperator, Callable[[Any, Any], ColumnElement[bool]]] = {
    FilterOperator.GT: operator.gt,
    FilterOperator.GTE: operator.ge,
    FilterOperator.LT: operator.lt,
    FilterOperator.LTE: operator.le,
}

# JSON scalar type name (`jsonb_typeof`) holding each kind.
_JSON_TYPE = {"str": "string", "number": "number", "bool": "boolean"}


@dataclass(frozen=True)
class ColumnField:
    """A field backed by a real column (or any scalar SQL expression)."""

    column: ColumnElement[Any]
    kind: FieldKind


@dataclass(frozen=True)
class JsonbField:
    """A field stored under `key` of a JSONB column."""

    column: ColumnElement[Any]
    key: str
    kind: Literal["str", "number", "bool"]
    # The key holds a JSON ARRAY of `kind` items, filtered as a set: EQ = contains the value,
    # IN = overlaps. Both are emitted as containment (`@>`), which a jsonb_path_ops GIN serves.
    array: bool = False

    def __post_init__(self) -> None:
        if FIELD_NAME_PATTERN.fullmatch(self.key) is None:
            raise ValueError(f"Invalid JSONB key {self.key!r}")


type FieldSpec = ColumnField | JsonbField


def _json_key(spec: JsonbField) -> ColumnElement[Any]:
    return sa.bindparam(None, spec.key, type_=sa.String, unique=True, literal_execute=True)


def jsonb_typed_expression(spec: JsonbField) -> ColumnElement[Any]:
    """Typed scalar for a JSONB key: NULL when the key is missing OR holds another JSON type.

    Casts only happen under the `jsonb_typeof` guard, so a stray value can never raise a cast
    error. Built from immutable operators only, so it can back an expression index, and used
    textually as-is by ordering predicates (query and index must match).
    """
    node = spec.column.op("->", return_type=JSONB)(_json_key(spec))
    text = spec.column.op("->>", return_type=sa.Text)(_json_key(spec))
    typed: ColumnElement[Any] = text
    if spec.kind == "number":
        typed = sa.cast(text, sa.Numeric)
    elif spec.kind == "bool":
        typed = sa.cast(text, sa.Boolean)
    # Inline constant (fixed internal vocabulary), so the expression matches an expression index.
    json_type: ColumnElement[Any] = sa.literal_column(f"'{_JSON_TYPE[spec.kind]}'")
    return sa.case((sa.func.jsonb_typeof(node) == json_type, typed), else_=sa.null())


def _scalar_expression(spec: FieldSpec) -> ColumnElement[Any]:
    return spec.column if isinstance(spec, ColumnField) else jsonb_typed_expression(spec)


def ordering_expression(spec: FieldSpec) -> ColumnElement[Any]:
    """The exact expression ordering predicates and ORDER BY compare: scalar + text collation.

    A btree expression index on this expression serves range filters and ordering on the field;
    it is the single source of truth for both (the index sync derives its DDL from it).
    """
    expression = _scalar_expression(spec)
    if spec.kind == "str":
        expression = expression.collate(TEXT_COLLATION)
    return expression


def _kind_of(value: Scalar) -> FieldKind:
    match value_kind(value):
        case "bool":
            return "bool"
        case "int" | "float":
            return "number"
        case "str":
            return "str"
        case _:
            return "datetime"


def _check_representable(value: Scalar) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise UnsupportedFilterError(f"Non-finite number {value!r} cannot be filtered in SQL")
    if isinstance(value, str):
        if "\x00" in value:
            raise UnsupportedFilterError("Strings with NUL characters cannot be filtered in SQL")
        try:
            value.encode("utf-8")
        except UnicodeEncodeError:
            raise UnsupportedFilterError(
                "Strings with lone surrogates cannot be filtered"
            ) from None


def _compatible(spec: FieldSpec, value: Scalar) -> bool:
    if _kind_of(value) != spec.kind:
        return False
    # A naive datetime never equals nor orders against the aware values stored in timestamptz.
    return not (isinstance(value, datetime) and value.utcoffset() is None)


def _param(value: Scalar, spec: FieldSpec | None = None) -> ColumnElement[Any]:
    """Bind `value` as a parameter typed like its comparison domain.

    Numbers bind as `numeric` (exact, shortest-repr floats) EXCEPT an int against a real integer
    column: that binds as `bigint`, so the comparison stays `int <op> bigint` and a btree on the
    column is usable (`int_col <op> numeric` would cast the column and defeat the index).
    `int(...)` / `float(...)` normalize subclasses (IntEnum members, numpy-like scalars) whose
    `repr` is not a number.
    """
    if isinstance(value, bool):
        return sa.literal(value, sa.Boolean)
    if isinstance(value, int):
        if isinstance(spec, ColumnField) and isinstance(spec.column.type, sa.Integer):
            return sa.literal(int(value), sa.BigInteger)
        return sa.literal(Decimal(int(value)), sa.Numeric)
    if isinstance(value, float):
        return sa.literal(Decimal(repr(float(value))), sa.Numeric)
    if isinstance(value, datetime):
        return sa.literal(value, sa.DateTime(timezone=True))
    return sa.literal(str(value), sa.Text)  # str(): StrEnum members bind as their value


def _equals(spec: FieldSpec, values: Sequence[Scalar]) -> ColumnElement[bool]:
    """Positive equality against any of the (kind-compatible) `values`."""
    if isinstance(spec, JsonbField):
        # json.dumps of a finite scalar; Enum subclasses are normalized to plain int/float/str.
        # A list field is a JSON array: `{"k": ["v"]}` is contained when "v" is one of its items.
        return sa.or_(
            *(
                spec.column.contains({spec.key: [_json_value(v)] if spec.array else _json_value(v)})
                for v in values
            )
        )
    params = [_param(v, spec) for v in values]
    if len(params) == 1:
        return spec.column == params[0]
    return spec.column.in_(params)


def _json_value(value: Scalar) -> str | int | float | bool:
    if isinstance(value, datetime):  # unreachable: datetime is never JSONB-compatible
        raise UnsupportedFilterError("datetime values cannot be matched inside JSONB")
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    return str(value)


def _negated(spec: FieldSpec, positive: ColumnElement[bool]) -> ColumnElement[bool]:
    if isinstance(spec, JsonbField):
        return sa.not_(positive)  # missing key -> containment false -> NOT true
    return sa.or_(spec.column.is_(None), sa.not_(positive))


def _ordering(spec: FieldSpec, op: FilterOperator, value: Scalar) -> ColumnElement[bool]:
    if isinstance(value, bool):
        raise UnsupportedFilterError("Ordering comparisons on bool values are not supported")
    if isinstance(spec, JsonbField) and spec.kind == "bool":
        raise UnsupportedFilterError("Ordering comparisons on bool fields are not supported")
    return _ORDERING[op](ordering_expression(spec), _param(value, spec))


def _leaf(expr: Filter, fields: Mapping[str, FieldSpec]) -> ColumnElement[bool]:
    spec = fields.get(expr.field)
    if spec is None:
        raise UnknownFieldError(f"Field {expr.field!r} is not mapped to a column")
    if FIELD_NAME_PATTERN.fullmatch(expr.field) is None:  # defense in depth; Filter checks too
        raise UnsupportedFilterError(f"Invalid field name {expr.field!r}")
    values: tuple[Scalar, ...] = expr.value if isinstance(expr.value, tuple) else (expr.value,)
    if len(values) > MAX_IN_VALUES:
        raise UnsupportedFilterError(
            f"{expr.operator.name} on {expr.field!r} has {len(values)} values "
            f"(limit {MAX_IN_VALUES})"
        )
    for value in values:
        _check_representable(value)
    compatible = [v for v in values if _compatible(spec, v)]

    if expr.operator in _ORDERING:
        if not compatible or (isinstance(spec, JsonbField) and spec.array):
            return sa.false()  # list fields never order (oracle: ordering never matches a list)
        return _ordering(spec, expr.operator, compatible[0])

    negative = expr.operator in (FilterOperator.NEQ, FilterOperator.NOT_IN)
    if not compatible:
        return sa.true() if negative else sa.false()
    positive = _equals(spec, compatible)
    return _negated(spec, positive) if negative else positive


def compile_filter(expr: FilterExpr, fields: Mapping[str, FieldSpec]) -> ColumnElement[bool]:
    """Translate a filter expression to a SQL boolean using only the mapped `fields`."""
    match expr:
        case Filter():
            return _leaf(expr, fields)
        case AllOf():
            return sa.and_(*(compile_filter(sub, fields) for sub in expr.exprs))
        case AnyOf():
            return sa.or_(*(compile_filter(sub, fields) for sub in expr.exprs))


def _order_clause(order: Order, fields: Mapping[str, FieldSpec]) -> ColumnElement[Any]:
    spec = fields.get(order.field)
    if spec is None:
        raise UnknownFieldError(f"Field {order.field!r} is not mapped to a column")
    if isinstance(spec, JsonbField) and (spec.array or spec.kind in ("bool", "datetime")):
        raise UnsupportedFilterError(f"Cannot order by JSONB field {order.field!r}")
    expression = ordering_expression(spec)
    ordered = expression.desc() if order.direction is OrderDirection.DESC else expression.asc()
    return ordered.nulls_last()


def compile_order(
    orders: Sequence[Order], fields: Mapping[str, FieldSpec]
) -> list[ColumnElement[Any]]:
    """ORDER BY clauses: requested orders (NULLS LAST) then `id` ascending as tiebreak."""
    clauses = [_order_clause(order, fields) for order in orders]
    clauses.append(_order_clause(Order(TIE_BREAKER_FIELD), fields))
    return clauses


def apply_criteria[S: sa.Select[Any]](
    stmt: S, criteria: Criteria, fields: Mapping[str, FieldSpec]
) -> S:
    """Add WHERE / ORDER BY / LIMIT / OFFSET of `criteria` to a SELECT."""
    if criteria.filter is not None:
        stmt = stmt.where(compile_filter(criteria.filter, fields))
    stmt = stmt.order_by(*compile_order(criteria.order, fields))
    if criteria.limit is not None:
        stmt = stmt.limit(criteria.limit)
    if criteria.offset:
        stmt = stmt.offset(criteria.offset)
    return stmt
