"""Field-name validation and tenant scoping helpers shared by repositories and converters."""

from collections.abc import Collection

from agintis.shared_kernel.criteria.criteria import Criteria
from agintis.shared_kernel.criteria.errors import UnknownFieldError
from agintis.shared_kernel.criteria.filters import AllOf, AnyOf, Filter, FilterExpr
from agintis.shared_kernel.ids import TenantId

# Lives in the shared kernel (not in a business module) because every tenant-aware
# repository and every backend converter must agree on the same field name.
TENANT_FIELD = "tenant_id"


def fields_of(expr: FilterExpr) -> set[str]:
    """Return every field name referenced by the expression."""
    match expr:
        case Filter():
            return {expr.field}
        case AllOf() | AnyOf():
            return set().union(*(fields_of(sub) for sub in expr.exprs))


def ensure_fields(expr: FilterExpr | Criteria | None, allowed: Collection[str]) -> None:
    """Raise UnknownFieldError when filter (or Criteria filter/order) uses undeclared fields."""
    used: set[str] = set()
    if isinstance(expr, Criteria):
        if expr.filter is not None:
            used |= fields_of(expr.filter)
        used |= {order.field for order in expr.order}
    elif expr is not None:
        used |= fields_of(expr)
    unknown = used - set(allowed)
    if unknown:
        raise UnknownFieldError(f"Unknown fields: {sorted(unknown)}; allowed: {sorted(allowed)}")


def tenant_scoped(tenant_id: TenantId, expr: FilterExpr | None) -> FilterExpr:
    """Return `tenant_id = X AND expr`. The caller's expression is kept as ONE operand.

    Nesting (instead of flattening) guarantees a top-level AnyOf in `expr` can never
    short-circuit the tenant restriction.
    """
    tenant = Filter.eq(TENANT_FIELD, tenant_id)
    return tenant if expr is None else AllOf(tenant, expr)
