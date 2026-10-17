"""In-memory evaluation of Criteria.

This module is the REFERENCE SEMANTICS ORACLE: fakes use it and every real backend
converter (SQL, PGVector, Qdrant) must produce the same results for the same filter. The
semantics are documented in `filters.py`; adapter contract tests (tests/contract) enforce them.
"""

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, assert_never, cast

from agintis.shared_kernel.criteria.criteria import Criteria, OrderDirection
from agintis.shared_kernel.criteria.filters import (
    AllOf,
    AnyOf,
    Filter,
    FilterExpr,
    FilterOperator,
    value_kind,
)

_NUMERIC_KINDS = frozenset({"int", "float"})


def _comparable(actual: Any, expected: Any) -> bool:
    kind_a, kind_e = value_kind(actual), value_kind(expected)
    return kind_a == kind_e or (kind_a in _NUMERIC_KINDS and kind_e in _NUMERIC_KINDS)


def _equal(actual: Any, expected: Any) -> bool:
    return _comparable(actual, expected) and bool(actual == expected)


def _ordered(operator: FilterOperator, actual: Any, expected: Any) -> bool:
    if not _comparable(actual, expected):
        return False
    try:
        match operator:
            case FilterOperator.GT:
                return bool(actual > expected)
            case FilterOperator.GTE:
                return bool(actual >= expected)
            case FilterOperator.LT:
                return bool(actual < expected)
            case FilterOperator.LTE:
                return bool(actual <= expected)
            case _:
                raise AssertionError(f"not an ordering operator: {operator}")
    except TypeError:  # e.g. naive vs aware datetimes
        return False


def _compare_collection(operator: FilterOperator, items: Sequence[Any], expected: Any) -> bool:
    """A list-valued field behaves as the set of its items.

    EQ: some item equals the value. IN: some item equals some value (overlap). NEQ / NOT_IN are
    the exact negations, so an empty list matches them. Ordering never matches.
    """
    match operator:
        case FilterOperator.EQ:
            return any(_equal(item, expected) for item in items)
        case FilterOperator.NEQ:
            return not any(_equal(item, expected) for item in items)
        case FilterOperator.IN:
            return any(_equal(item, wanted) for item in items for wanted in expected)
        case FilterOperator.NOT_IN:
            return not any(_equal(item, wanted) for item in items for wanted in expected)
        case FilterOperator.GT | FilterOperator.GTE | FilterOperator.LT | FilterOperator.LTE:
            return False
        case _:
            assert_never(operator)


def _compare(operator: FilterOperator, actual: Any, expected: Any) -> bool:
    # Missing/None field: only the negative operators match (see filters.py semantics).
    if actual is None:
        return operator in {FilterOperator.NEQ, FilterOperator.NOT_IN}
    if isinstance(actual, list | tuple):
        return _compare_collection(operator, actual, expected)
    match operator:
        case FilterOperator.EQ:
            return _equal(actual, expected)
        case FilterOperator.NEQ:
            return not _equal(actual, expected)
        case FilterOperator.IN:
            return any(_equal(actual, item) for item in expected)
        case FilterOperator.NOT_IN:
            return not any(_equal(actual, item) for item in expected)
        case FilterOperator.GT | FilterOperator.GTE | FilterOperator.LT | FilterOperator.LTE:
            return _ordered(operator, actual, expected)
        case _:
            assert_never(operator)


def matches(expr: FilterExpr, record: Mapping[str, Any]) -> bool:
    """Return True when `record` satisfies the filter expression."""
    match expr:
        case Filter():
            return _compare(expr.operator, record.get(expr.field), expr.value)
        case AllOf():
            return all(matches(sub, record) for sub in expr.exprs)
        case AnyOf():
            return any(matches(sub, record) for sub in expr.exprs)
        case _:
            assert_never(expr)


def _as_mapping[T](item: T) -> Mapping[str, Any]:
    return cast(Mapping[str, Any], item)


def _sorted_by[T](
    items: list[T],
    field: str,
    as_record: Callable[[T], Mapping[str, Any]],
    *,
    descending: bool,
) -> list[T]:
    """Stable sort on one field; records missing it go last regardless of direction."""
    present = [i for i in items if as_record(i).get(field) is not None]
    missing = [i for i in items if as_record(i).get(field) is None]
    present.sort(key=lambda item: as_record(item)[field], reverse=descending)
    return present + missing


def apply[T](
    criteria: Criteria,
    items: Iterable[T],
    *,
    as_record: Callable[[T], Mapping[str, Any]] = _as_mapping,
) -> list[T]:
    """Filter, order, then paginate `items` according to `criteria`.

    Items are mappings by default; pass `as_record` to project entities to records.
    Records missing an order field sort last, regardless of direction; ties break by `id`.
    """
    result = list(items)
    if criteria.filter is not None:
        expr = criteria.filter
        result = [item for item in result if matches(expr, as_record(item))]
    # Stable sorts applied from the least to the most significant key; `id` (ascending,
    # missing last) is the least significant one, so ties are always broken by id.
    result = _sorted_by(result, "id", as_record, descending=False)
    for order in reversed(criteria.order):
        result = _sorted_by(
            result, order.field, as_record, descending=order.direction is OrderDirection.DESC
        )
    end = None if criteria.limit is None else criteria.offset + criteria.limit
    return result[criteria.offset : end]
