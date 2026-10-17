"""Tenant resolution at the request boundary.

SECURITY: `HeaderTenantResolver` trusts the client-supplied `X-Tenant-Id` header. It is only
safe behind a trusted gateway that authenticates callers and sets/overwrites the header.
`JwtTenantResolver` (platform/http/jwt_tenant.py) takes the tenant from a verified token claim.
"""

import re
from typing import Protocol, cast

from fastapi import Request

from agintis.shared_kernel.errors import AppError, ErrorKind
from agintis.shared_kernel.ids import TenantId

TENANT_HEADER = "X-Tenant-Id"
TENANT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class TenantRequiredError(AppError):
    kind = ErrorKind.BAD_REQUEST
    code = "tenant_required"


class InvalidTenantError(AppError):
    kind = ErrorKind.BAD_REQUEST
    code = "invalid_tenant"


class TenantResolver(Protocol):
    async def resolve(self, request: Request) -> TenantId:
        """Return the caller's tenant or raise an `AppError` (before any stream opens)."""
        ...

    async def aclose(self) -> None: ...


class HeaderTenantResolver:
    """Reads the tenant from `X-Tenant-Id`; `default_tenant` covers a missing header.

    The composition root passes `default_tenant` only in local development.
    """

    def __init__(self, default_tenant: TenantId | None = None, header: str = TENANT_HEADER) -> None:
        self._default = default_tenant
        self._header = header

    async def resolve(self, request: Request) -> TenantId:
        raw = request.headers.get(self._header, "").strip()
        if not raw:
            if self._default is None:
                raise TenantRequiredError(f"Header {self._header} is required")
            return self._default
        if not TENANT_PATTERN.fullmatch(raw):
            raise InvalidTenantError(f"Header {self._header} is not a valid tenant id")
        return TenantId(raw)

    async def aclose(self) -> None:
        return None


async def resolve_tenant(request: Request) -> TenantId:
    """FastAPI dependency: the resolver is published on `app.state` by the app factory."""
    resolver = cast(TenantResolver, request.app.state.tenant_resolver)
    return await resolver.resolve(request)
