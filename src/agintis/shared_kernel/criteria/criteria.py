"""Criteria: filter + ordering + pagination for repository queries (never for behavior)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

from agintis.shared_kernel.criteria.filters import (
    FIELD_NAME_PATTERN,
    AllOf,
    FilterExpr,
    is_valid_field_name,
)


class OrderDirection(StrEnum):
    ASC = "asc"
    DESC = "desc"


@dataclass(frozen=True)
class Order:
    field: str
    direction: OrderDirection = OrderDirection.ASC

    def __post_init__(self) -> None:
        if not is_valid_field_name(self.field):
            raise ValueError(
                f"Order field {self.field!r} is invalid (must match {FIELD_NAME_PATTERN.pattern})"
            )


@dataclass(frozen=True)
class Criteria:
    filter: FilterExpr | None = None
    order: tuple[Order, ...] = ()
    limit: int | None = None
    offset: int = 0

    def __post_init__(self) -> None:
        if self.limit is not None and self.limit <= 0:
            raise ValueError("limit must be greater than 0")
        if self.offset < 0:
            raise ValueError("offset must be greater than or equal to 0")

    @classmethod
    def all_of(cls, *filters: FilterExpr) -> Criteria:
        """Build a Criteria AND-ing the given expressions (none -> match everything)."""
        match filters:
            case ():
                return cls()
            case (single,):
                return cls(filter=single)
            case _:
                return cls(filter=AllOf(*filters))

    def and_(self, extra: FilterExpr) -> Criteria:
        """Return a new Criteria whose filter is `extra AND self.filter`.

        Converters/adapters use it to inject mandatory filters such as the tenant.
        """
        if self.filter is None:
            return replace(self, filter=extra)
        current = self.filter.exprs if isinstance(self.filter, AllOf) else (self.filter,)
        return replace(self, filter=AllOf(extra, *current))
