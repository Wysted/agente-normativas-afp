import pytest
from starlette.requests import Request

from agintis.platform.http.tenant import (
    HeaderTenantResolver,
    InvalidTenantError,
    TenantRequiredError,
)
from agintis.shared_kernel.ids import TenantId


def _request(headers: dict[str, str]) -> Request:
    scope = {
        "type": "http",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
    }
    return Request(scope)


async def test_header_present() -> None:
    resolver = HeaderTenantResolver()

    assert await resolver.resolve(_request({"X-Tenant-Id": " acme "})) == "acme"


async def test_header_wins_over_default() -> None:
    resolver = HeaderTenantResolver(TenantId("dev"))

    assert await resolver.resolve(_request({"X-Tenant-Id": "acme"})) == "acme"


async def test_missing_uses_default() -> None:
    assert await HeaderTenantResolver(TenantId("dev")).resolve(_request({})) == "dev"


async def test_missing_without_default_is_tenant_required() -> None:
    with pytest.raises(TenantRequiredError) as info:
        await HeaderTenantResolver().resolve(_request({}))
    assert info.value.code == "tenant_required"


@pytest.mark.parametrize("value", ["a b", "../x", "x" * 65, "-lead"])
async def test_malformed_header(value: str) -> None:
    with pytest.raises(InvalidTenantError):
        await HeaderTenantResolver().resolve(_request({"X-Tenant-Id": value}))
