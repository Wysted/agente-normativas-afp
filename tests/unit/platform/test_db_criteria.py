"""SQL converter: compiled-SQL checks (no database). Behavior is covered by the contract suite."""

from datetime import UTC, datetime
from decimal import Decimal
from enum import IntEnum

import pytest
from sqlalchemy import BigInteger, Column, Integer, MetaData, Numeric, Table, Text
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects.postgresql import JSONB

from agintis.platform.db import (
    ColumnField,
    FieldSpec,
    JsonbField,
    compile_filter,
    compile_order,
)
from agintis.platform.db.criteria import MAX_IN_VALUES
from agintis.shared_kernel.criteria import (
    AllOf,
    AnyOf,
    Filter,
    Order,
    OrderDirection,
    UnknownFieldError,
    UnsupportedFilterError,
)

_t = Table(
    "t",
    MetaData(),
    Column("id", Text),
    Column("name", Text),
    Column("n", Text),
    Column("created", Text),
    Column("ordinal", Integer),
    Column("big", BigInteger),
    Column("meta", JSONB),
)
PG = postgresql.dialect()  # type: ignore[no-untyped-call]
FIELDS: dict[str, FieldSpec] = {
    "id": ColumnField(_t.c.id, "str"),
    "name": ColumnField(_t.c.name, "str"),
    "n": ColumnField(_t.c.n, "number"),
    "created": ColumnField(_t.c.created, "datetime"),
    "ordinal": ColumnField(_t.c.ordinal, "number"),
    "big": ColumnField(_t.c.big, "number"),
    "article": JsonbField(_t.c.meta, "article", "str"),
    "page": JsonbField(_t.c.meta, "page", "number"),
    "flag": JsonbField(_t.c.meta, "flag", "bool"),
    "tags": JsonbField(_t.c.meta, "tags", "str", array=True),
    "refs": JsonbField(_t.c.meta, "refs", "number", array=True),
}


def _sql(expr: Filter | AllOf | AnyOf) -> str:
    return str(compile_filter(expr, FIELDS).compile(dialect=PG))


def _bind_types(expr: Filter | AllOf | AnyOf) -> list[type]:
    compiled = compile_filter(expr, FIELDS).compile(dialect=PG)
    return [type(bind.type) for bind in compiled.binds.values()]


def _params(expr: Filter | AllOf | AnyOf) -> dict[str, object]:
    return dict(compile_filter(expr, FIELDS).compile(dialect=PG).params)


class TestSafety:
    def test_unmapped_field_is_rejected(self) -> None:
        with pytest.raises(UnknownFieldError):
            compile_filter(Filter.eq("password", "x"), FIELDS)

    def test_values_are_bound_never_inlined(self) -> None:
        payload = "x'; DROP TABLE t; --"
        sql = _sql(AllOf(Filter.eq("name", payload), Filter.eq("article", payload)))
        assert "DROP" not in sql
        assert payload in _params(Filter.eq("name", payload)).values()

    def test_jsonb_key_must_be_a_plain_identifier(self) -> None:
        with pytest.raises(ValueError, match="Invalid JSONB key"):
            JsonbField(_t.c.meta, "a'; DROP TABLE t; --", "str")

    def test_non_finite_numbers_are_unsupported(self) -> None:
        for value in (float("nan"), float("inf")):
            with pytest.raises(UnsupportedFilterError):
                compile_filter(Filter.eq("n", value), FIELDS)

    @pytest.mark.parametrize("value", ["a\x00b", "\x00", "lone \ud800 surrogate"])
    def test_strings_postgres_cannot_store_are_unsupported(self, value: str) -> None:
        for expr in (
            Filter.eq("name", value),
            Filter.in_("article", [value]),
            Filter.neq("name", value),
        ):
            with pytest.raises(UnsupportedFilterError):
                compile_filter(expr, FIELDS)

    def test_ordering_on_bool_is_unsupported(self) -> None:
        with pytest.raises(UnsupportedFilterError):
            compile_filter(Filter.gt("flag", True), FIELDS)


class TestOracleSemantics:
    def test_column_negatives_include_null(self) -> None:
        assert _sql(Filter.neq("name", "a")).startswith("t.name IS NULL OR t.name !=")
        assert "IS NULL" in _sql(Filter.not_in("name", ["a", "b"]))

    def test_jsonb_negatives_negate_containment(self) -> None:
        sql = _sql(Filter.neq("article", "12"))
        assert "NOT" in sql
        assert "@>" in sql
        assert {"article": "12"} in _params(Filter.neq("article", "12")).values()

    def test_strict_types_fold_to_constants(self) -> None:
        assert _sql(Filter.eq("n", "5")) == "false"
        assert _sql(Filter.neq("n", "5")) == "true"
        assert _sql(Filter.eq("article", 5)) == "false"
        assert _sql(Filter.gt("article", 5)) == "false"
        assert _sql(Filter.not_in("page", ["a"])) == "true"
        assert _sql(Filter.eq("n", True)) == "false"  # bool is not a number

    def test_naive_datetimes_never_match_timestamptz(self) -> None:
        naive = datetime(2026, 1, 1)
        assert _sql(Filter.gt("created", naive)) == "false"
        assert "false" not in _sql(Filter.gt("created", datetime(2026, 1, 1, tzinfo=UTC)))

    def test_text_ordering_uses_the_c_collation(self) -> None:
        assert 'COLLATE "C"' in _sql(Filter.gt("name", "a"))
        assert 'COLLATE "C"' in _sql(Filter.lt("article", "a"))

    def test_json_numbers_are_guarded_by_their_json_type(self) -> None:
        sql = _sql(Filter.gt("page", 3))
        assert "jsonb_typeof" in sql
        assert "'number'" in sql  # inline, so the predicate can match an expression index
        assert "CAST" in sql

    def test_numbers_bind_exactly(self) -> None:
        from decimal import Decimal

        assert Decimal("0.1") in _params(Filter.gt("page", 0.1)).values()

    def test_ints_against_integer_columns_bind_as_bigint_so_btrees_stay_usable(self) -> None:
        for field in ("ordinal", "big"):
            sql = str(compile_filter(Filter.gte(field, 3), FIELDS).compile(dialect=PG))
            assert "BIGINT" in sql
            assert "NUMERIC" not in sql
            assert "BIGINT" in str(
                compile_filter(Filter.in_(field, [1, 2]), FIELDS).compile(dialect=PG)
            )
        # Floats, and ints against non-integer kinds, keep exact numeric arithmetic.
        assert Numeric in _bind_types(Filter.gte("ordinal", 2.5))
        assert Numeric in _bind_types(Filter.gt("page", 3))
        assert BigInteger not in _bind_types(Filter.gt("page", 3))
        assert Numeric not in _bind_types(Filter.gt("ordinal", 3))

    def test_int_subclasses_bind_as_plain_numbers(self) -> None:
        class Level(IntEnum):
            LOW = 1
            HIGH = 10

        assert Decimal(10) in _params(Filter.gt("page", Level.HIGH)).values()
        assert {"page": 10} in _params(Filter.eq("page", Level.HIGH)).values()
        assert _params(Filter.gt("ordinal", Level.LOW))
        assert _sql(Filter.eq("n", Level.LOW)) != "true"  # an IntEnum is a number, not a str

    def test_in_with_mixed_compatibility_keeps_the_compatible_values(self) -> None:
        # Filter forbids heterogeneous tuples, so a whole IN is either compatible or folded.
        assert _sql(Filter.in_("n", ["a", "b"])) == "false"


class TestListFields:
    def test_equality_is_array_containment_served_by_the_gin_index(self) -> None:
        sql = _sql(Filter.eq("tags", "DL 3500"))
        assert "@>" in sql
        assert {"tags": ["DL 3500"]} in _params(Filter.eq("tags", "DL 3500")).values()

    def test_membership_is_an_or_chain_of_containments(self) -> None:
        expr = Filter.in_("refs", [7, 31])
        sql = _sql(expr)
        assert sql.count("@>") == 2
        assert " OR " in sql
        assert "?|" not in sql  # the overlap operator would bypass jsonb_path_ops
        assert {"refs": [7]} in _params(expr).values()
        assert {"refs": [31]} in _params(expr).values()

    def test_membership_is_capped(self) -> None:
        within = Filter.in_("refs", range(MAX_IN_VALUES))
        assert _sql(within).count("@>") == MAX_IN_VALUES
        with pytest.raises(UnsupportedFilterError, match="limit"):
            _sql(Filter.in_("refs", range(MAX_IN_VALUES + 1)))
        with pytest.raises(UnsupportedFilterError, match="limit"):
            _sql(Filter.not_in("n", range(MAX_IN_VALUES + 1)))

    def test_a_row_in_the_other_shape_does_not_match_the_declared_one(self) -> None:
        """Documented divergence: SQL follows the declared shape, the oracle the stored one."""
        scalar = Filter.eq("refs", 7)  # a list[int] field
        assert _params(scalar)  # compiled as {"refs": [7]}: contains only a stored ARRAY
        assert {"refs": [7]} in _params(scalar).values()
        assert {"refs": 7} not in _params(scalar).values()

    def test_negations_negate_containment_so_missing_and_empty_match(self) -> None:
        assert _sql(Filter.neq("tags", "a")).startswith("NOT")
        assert _sql(Filter.not_in("tags", ["a", "b"])).startswith("NOT")

    def test_strict_kinds_and_ordering_fold_to_constants(self) -> None:
        assert _sql(Filter.eq("tags", 5)) == "false"
        assert _sql(Filter.neq("tags", 5)) == "true"
        assert _sql(Filter.eq("refs", "7")) == "false"
        assert _sql(Filter.gt("refs", 1)) == "false"
        assert _sql(Filter.lte("tags", "z")) == "false"

    def test_values_are_bound(self) -> None:
        payload = "x'}; DROP TABLE t; --"
        assert "DROP" not in _sql(Filter.eq("tags", payload))

    def test_list_fields_cannot_be_ordered_by(self) -> None:
        with pytest.raises(UnsupportedFilterError):
            compile_order((Order("tags"),), FIELDS)


class TestOrder:
    def test_nulls_last_and_id_tiebreak(self) -> None:
        clauses = compile_order((Order("name", OrderDirection.DESC),), FIELDS)
        rendered = [str(c.compile(dialect=PG)) for c in clauses]
        assert rendered[0] == 't.name COLLATE "C" DESC NULLS LAST'
        assert rendered[-1] == 't.id COLLATE "C" ASC NULLS LAST'

    def test_unknown_order_field(self) -> None:
        with pytest.raises(UnknownFieldError):
            compile_order((Order("nope"),), FIELDS)

    def test_ordering_by_a_jsonb_number_uses_the_typed_expression(self) -> None:
        (clause, _) = compile_order((Order("page"),), FIELDS)
        assert "jsonb_typeof" in str(clause.compile(dialect=PG))
