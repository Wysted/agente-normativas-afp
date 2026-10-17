from datetime import UTC, datetime
from typing import Any, ClassVar

import pytest

from agintis.shared_kernel.criteria import (
    AllOf,
    AnyOf,
    Criteria,
    Filter,
    FilterOperator,
    Order,
    OrderDirection,
    UnknownFieldError,
    UnsupportedFilterError,
    ensure_fields,
    fields_of,
    tenant_scoped,
)
from agintis.shared_kernel.criteria.in_memory import apply, matches
from agintis.shared_kernel.errors import DomainError
from agintis.shared_kernel.ids import TenantId

RECORD = {"name": "ana", "age": 30, "tags": "a"}


class TestFilterValidation:
    @pytest.mark.parametrize("operator", [FilterOperator.IN, FilterOperator.NOT_IN])
    @pytest.mark.parametrize("value", ["x", (), ["a"], (1, [2])])
    def test_multi_operators_require_non_empty_tuple_of_scalars(
        self, operator: FilterOperator, value: object
    ) -> None:
        with pytest.raises(ValueError, match=r"tuple|scalars"):
            Filter("f", operator, value)  # type: ignore[arg-type]

    @pytest.mark.parametrize("value", [(1, 2), None, [1]])
    def test_scalar_operators_reject_non_scalars(self, value: object) -> None:
        with pytest.raises(ValueError, match="scalar"):
            Filter("f", FilterOperator.EQ, value)  # type: ignore[arg-type]

    @pytest.mark.parametrize("name", ["", "1a", "a-b", "a b", "a.b", "a;drop", "ñ"])
    def test_invalid_field_names_are_rejected(self, name: str) -> None:
        with pytest.raises(ValueError, match="field"):
            Filter.eq(name, 1)
        with pytest.raises(ValueError, match="field"):
            Order(name)

    @pytest.mark.parametrize("name", ["a", "_a", "A1_b"])
    def test_valid_field_names_are_accepted(self, name: str) -> None:
        assert Filter.eq(name, 1).field == name

    @pytest.mark.parametrize("values", [(1, "a"), (1, 1.5), (True, 1), ("a", 1)])
    def test_multi_values_must_be_homogeneous(self, values: tuple[object, ...]) -> None:
        with pytest.raises(ValueError, match="same type"):
            Filter("f", FilterOperator.IN, values)  # type: ignore[arg-type]

    @pytest.mark.parametrize("values", [(1, 2), (1.5, 2.5), ("a", "b"), (True, False)])
    def test_homogeneous_multi_values_are_accepted(self, values: tuple[object, ...]) -> None:
        assert Filter("f", FilterOperator.NOT_IN, values).value == values  # type: ignore[arg-type]

    def test_in_helper_builds_tuple(self) -> None:
        assert Filter.in_("f", [1, 2]).value == (1, 2)

    def test_composites_must_be_non_empty(self) -> None:
        with pytest.raises(ValueError, match="AllOf"):
            AllOf()
        with pytest.raises(ValueError, match="AnyOf"):
            AnyOf()


class TestCriteriaValidation:
    @pytest.mark.parametrize("limit", [0, -1])
    def test_limit_must_be_positive(self, limit: int) -> None:
        with pytest.raises(ValueError, match="limit"):
            Criteria(limit=limit)

    def test_offset_must_not_be_negative(self) -> None:
        with pytest.raises(ValueError, match="offset"):
            Criteria(offset=-1)

    def test_all_of_shapes(self) -> None:
        a, b = Filter.eq("a", 1), Filter.eq("b", 2)
        assert Criteria.all_of().filter is None
        assert Criteria.all_of(a).filter == a
        assert Criteria.all_of(a, b).filter == AllOf(a, b)

    def test_and_returns_new_criteria_with_extra_first(self) -> None:
        a, b, t = Filter.eq("a", 1), Filter.eq("b", 2), Filter.eq("tenant_id", "t")
        original = Criteria.all_of(a, b)
        combined = original.and_(t)
        assert combined.filter == AllOf(t, a, b)
        assert original.filter == AllOf(a, b)

    def test_and_on_empty_and_single(self) -> None:
        t, a = Filter.eq("t", 1), Filter.eq("a", 2)
        assert Criteria().and_(t).filter == t
        assert Criteria(filter=a).and_(t).filter == AllOf(t, a)

    def test_and_keeps_order_and_pagination(self) -> None:
        base = Criteria(order=(Order("a"),), limit=3, offset=1)
        combined = base.and_(Filter.eq("t", 1))
        assert (combined.order, combined.limit, combined.offset) == (base.order, 3, 1)


class TestMatches:
    @pytest.mark.parametrize(
        ("filter_", "expected"),
        [
            (Filter.eq("age", 30), True),
            (Filter.eq("age", 31), False),
            (Filter.neq("age", 31), True),
            (Filter.neq("age", 30), False),
            (Filter.gt("age", 29), True),
            (Filter.gt("age", 30), False),
            (Filter.gte("age", 30), True),
            (Filter.gte("age", 31), False),
            (Filter.lt("age", 31), True),
            (Filter.lt("age", 30), False),
            (Filter.lte("age", 30), True),
            (Filter.lte("age", 29), False),
            (Filter.in_("name", ["ana", "bob"]), True),
            (Filter.in_("name", ["bob"]), False),
            (Filter.not_in("name", ["bob"]), True),
            (Filter.not_in("name", ["ana"]), False),
        ],
    )
    def test_every_operator(self, filter_: Filter, expected: bool) -> None:
        assert matches(filter_, RECORD) is expected

    def test_missing_field_only_matches_negative_operators(self) -> None:
        assert not matches(Filter.eq("nope", 1), RECORD)
        assert not matches(Filter.gt("nope", 1), RECORD)
        assert not matches(Filter.in_("nope", [1]), RECORD)
        assert matches(Filter.neq("nope", 1), RECORD)
        assert matches(Filter.not_in("nope", [1]), RECORD)

    def test_missing_and_none_are_equivalent_for_negative_operators(self) -> None:
        # Reference semantics: converters must emit `IS NULL OR <>` / `IS NULL OR NOT IN`.
        record = {"x": None}
        assert matches(Filter.neq("x", 1), record)
        assert matches(Filter.not_in("x", [1]), record)
        assert not matches(Filter.eq("x", 1), record)
        assert not matches(Filter.lt("x", 1), record)

    def test_bool_is_distinct_from_int(self) -> None:
        assert not matches(Filter.eq("flag", 1), {"flag": True})
        assert not matches(Filter.eq("n", True), {"n": 1})
        assert matches(Filter.neq("flag", 1), {"flag": True})
        assert not matches(Filter.in_("n", [True]), {"n": 1})
        assert matches(Filter.not_in("n", [True]), {"n": 1})

    def test_int_and_float_compare_numerically(self) -> None:
        assert matches(Filter.eq("n", 1.0), {"n": 1})
        assert matches(Filter.lt("n", 1.5), {"n": 1})

    @pytest.mark.parametrize(
        "filter_",
        [Filter.gt("v", 1), Filter.gte("v", 1), Filter.lt("v", 1), Filter.lte("v", 1)],
    )
    def test_ordering_across_incompatible_types_does_not_match_or_raise(
        self, filter_: Filter
    ) -> None:
        assert not matches(filter_, {"v": "text"})
        assert not matches(filter_, {"v": True})

    def test_naive_vs_aware_datetime_does_not_raise(self) -> None:
        aware = datetime(2026, 1, 1, tzinfo=UTC)
        assert not matches(Filter.gt("at", aware), {"at": datetime(2025, 1, 1)})

    def test_datetime_comparison(self) -> None:
        moment = datetime(2026, 1, 1, tzinfo=UTC)
        record = {"at": moment}
        assert matches(Filter.gte("at", moment), record)
        assert not matches(Filter.gt("at", moment), record)

    def test_nesting(self) -> None:
        expr = AllOf(
            Filter.eq("name", "ana"),
            AnyOf(Filter.eq("age", 99), AllOf(Filter.gte("age", 18), Filter.lt("age", 65))),
        )
        assert matches(expr, RECORD)
        assert not matches(AllOf(expr, Filter.eq("name", "bob")), RECORD)
        assert not matches(AnyOf(Filter.eq("age", 1), Filter.eq("name", "x")), RECORD)


class TestListFields:
    """A list-valued field behaves as the SET of its items (see `filters.py`)."""

    RECORD: ClassVar[dict[str, object]] = {"tags": ["a", "b"], "refs": (7, 31), "none": []}

    @pytest.mark.parametrize(
        ("filter_", "expected"),
        [
            (Filter.eq("tags", "a"), True),
            (Filter.eq("tags", "c"), False),
            (Filter.neq("tags", "a"), False),
            (Filter.neq("tags", "c"), True),
            (Filter.in_("tags", ["c", "b"]), True),
            (Filter.in_("tags", ["c", "d"]), False),
            (Filter.not_in("tags", ["c", "b"]), False),
            (Filter.not_in("tags", ["c", "d"]), True),
            (Filter.eq("refs", 31), True),
            (Filter.eq("refs", 31.0), True),  # int and float compare numerically
            (Filter.in_("refs", [1, 7]), True),
            (Filter.eq("refs", "7"), False),  # strict kinds apply to the items
            (Filter.neq("refs", "7"), True),
            (Filter.eq("tags", 1), False),
            (Filter.gt("refs", 1), False),  # ordering never matches a list
            (Filter.lte("tags", "z"), False),
            (Filter.eq("none", "a"), False),  # an empty list contains nothing...
            (Filter.neq("none", "a"), True),  # ...so it matches the negations
            (Filter.not_in("none", ["a"]), True),
            (Filter.in_("none", ["a"]), False),
        ],
    )
    def test_set_semantics(self, filter_: Filter, expected: bool) -> None:
        assert matches(filter_, self.RECORD) is expected

    def test_a_missing_list_matches_only_the_negations(self) -> None:
        assert matches(Filter.neq("absent", "a"), self.RECORD)
        assert matches(Filter.not_in("absent", ["a"]), self.RECORD)
        assert not matches(Filter.eq("absent", "a"), self.RECORD)


class TestApply:
    RECORDS: tuple[dict[str, Any], ...] = (
        {"id": 1, "group": "a", "score": 3},
        {"id": 2, "group": "b", "score": 1},
        {"id": 3, "group": "a", "score": 2},
        {"id": 4, "group": "a"},
    )

    def ids(self, criteria: Criteria) -> list[int]:
        return [r["id"] for r in apply(criteria, self.RECORDS)]

    def test_no_criteria_returns_everything(self) -> None:
        assert self.ids(Criteria()) == [1, 2, 3, 4]

    def test_filter(self) -> None:
        assert self.ids(Criteria(filter=Filter.eq("group", "a"))) == [1, 3, 4]

    def test_order_asc_and_desc_with_missing_last(self) -> None:
        assert self.ids(Criteria(order=(Order("score"),))) == [2, 3, 1, 4]
        assert self.ids(Criteria(order=(Order("score", OrderDirection.DESC),))) == [1, 3, 2, 4]

    def test_ties_break_by_id_ascending_regardless_of_input_order_and_direction(self) -> None:
        records = [{"id": 3, "g": 1}, {"id": 1, "g": 1}, {"id": 2, "g": 1}, {"id": 4, "g": 0}]
        asc = apply(Criteria(order=(Order("g"),)), records)
        desc = apply(Criteria(order=(Order("g", OrderDirection.DESC),)), records)
        assert [r["id"] for r in asc] == [4, 1, 2, 3]
        assert [r["id"] for r in desc] == [1, 2, 3, 4]
        assert [r["id"] for r in apply(Criteria(limit=2), records)] == [1, 2]

    def test_multi_key_order(self) -> None:
        criteria = Criteria(order=(Order("group"), Order("score", OrderDirection.DESC)))
        assert self.ids(criteria) == [1, 3, 4, 2]

    def test_offset_and_limit(self) -> None:
        assert self.ids(Criteria(offset=1, limit=2)) == [2, 3]
        assert self.ids(Criteria(offset=3)) == [4]
        assert self.ids(Criteria(limit=1)) == [1]

    def test_projection_of_entities(self) -> None:
        out = apply(Criteria(filter=Filter.eq("v", 2)), [1, 2, 3], as_record=lambda i: {"v": i})
        assert out == [2]


class TestFieldHelpers:
    def test_fields_of_collects_nested_fields(self) -> None:
        expr = AllOf(Filter.eq("a", 1), AnyOf(Filter.eq("b", 1), Filter.eq("c", 1)))
        assert fields_of(expr) == {"a", "b", "c"}

    def test_ensure_fields_accepts_known_and_none(self) -> None:
        ensure_fields(None, {"a"})
        ensure_fields(Filter.eq("a", 1), {"a"})
        ensure_fields(Criteria(filter=Filter.eq("a", 1), order=(Order("a"),)), {"a"})

    def test_ensure_fields_rejects_unknown_filter_and_order_fields(self) -> None:
        with pytest.raises(UnknownFieldError, match="zzz"):
            ensure_fields(AnyOf(Filter.eq("a", 1), Filter.eq("zzz", 1)), {"a"})
        with pytest.raises(UnknownFieldError, match="zzz"):
            ensure_fields(Criteria(order=(Order("zzz"),)), {"a"})

    def test_converter_error_types_are_domain_errors(self) -> None:
        assert issubclass(UnsupportedFilterError, DomainError)
        assert issubclass(UnknownFieldError, DomainError)

    def test_tenant_scoped_keeps_caller_expression_as_one_operand(self) -> None:
        tenant = Filter.eq("tenant_id", "t")
        assert tenant_scoped(TenantId("t"), None) == tenant
        any_of = AnyOf(Filter.eq("a", 1), Filter.eq("b", 2))
        assert tenant_scoped(TenantId("t"), any_of) == AllOf(tenant, any_of)
        assert not matches(tenant_scoped(TenantId("t"), any_of), {"tenant_id": "other", "a": 1})
