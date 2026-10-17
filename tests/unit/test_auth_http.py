"""JWT mode through the real app: 401/403/503 contract, SSE pre-stream path, header mode intact."""

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from http import HTTPStatus
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import Request

from agintis.main import create_app
from agintis.modules.agents_catalog.api import (
    AgentRegistry,
    AgentsCatalogServices,
    RunAgent,
    RunCompleted,
    Token,
)
from agintis.modules.agents_catalog.http import get_services
from agintis.platform.auth import FileJwks, HttpJwks, JwtVerifier
from agintis.platform.http.jwt_tenant import SUBJECT_STATE_KEY, JwtTenantResolver
from agintis.platform.llm import ProviderErrorClassifier
from tests.fakes.agents import FakeRunner
from tests.support.auth import AUDIENCE, ISSUER, FakeIdp, ManualClock, ScriptedFetcher
from tests.support.settings import make_settings

BODY = {"agent_id": "fake", "message": "hello"}
OK_EVENTS = [Token("Hi"), RunCompleted("stop", None)]


@pytest.fixture(scope="module")
def idp() -> FakeIdp:
    return FakeIdp()


def _services(runner: FakeRunner) -> AgentsCatalogServices:
    registry = AgentRegistry((runner,))
    return AgentsCatalogServices(
        registry=registry,
        run_agent=RunAgent(registry, ProviderErrorClassifier(), id_generator=lambda: "run-1"),
        max_message_chars=100,
    )


@asynccontextmanager
async def _client(
    idp: FakeIdp,
    tmp_path: Path,
    *,
    resolver: JwtTenantResolver | None = None,
    tenant_claim: str = "tenant_id",
    runner: FakeRunner | None = None,
) -> AsyncIterator[httpx.AsyncClient]:
    jwks = tmp_path / "jwks.json"
    jwks.write_text(json.dumps(idp.jwks()))
    auth = {
        "tenant_resolver": "jwt",
        "issuer": ISSUER,
        "audience": AUDIENCE,
        "jwks_file": jwks,
        "tenant_claim": tenant_claim,
    }
    app = create_app(make_settings(auth=auth))
    fake = runner or FakeRunner(OK_EVENTS)
    app.dependency_overrides[get_services] = lambda: _services(fake)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c,
    ):
        if resolver is not None:
            app.state.tenant_resolver = resolver
        yield c


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def test_jwt_mode_boots_the_jwt_resolver(idp: FakeIdp, tmp_path: Path) -> None:
    jwks = tmp_path / "jwks.json"
    jwks.write_text(json.dumps(idp.jwks()))
    app = create_app(
        make_settings(
            auth={
                "tenant_resolver": "jwt",
                "issuer": ISSUER,
                "audience": AUDIENCE,
                "jwks_file": jwks,
            }
        )
    )
    async with app.router.lifespan_context(app):
        assert isinstance(app.state.tenant_resolver, JwtTenantResolver)


class TestSseRoute:
    async def test_streams_with_a_valid_token(self, idp: FakeIdp, tmp_path: Path) -> None:
        async with _client(idp, tmp_path) as c:
            response = await c.post("/v1/runs/stream", json=BODY, headers=_bearer(idp.token()))
        assert response.status_code == HTTPStatus.OK
        assert response.headers["content-type"].startswith("text/event-stream")
        assert "event: run_started" in response.text

    async def test_the_tenant_comes_from_the_token_not_the_header(
        self, idp: FakeIdp, tmp_path: Path
    ) -> None:
        runner = FakeRunner(OK_EVENTS)
        async with _client(idp, tmp_path, runner=runner) as c:
            headers = {**_bearer(idp.token(tenant_id="from-token")), "X-Tenant-Id": "spoofed"}
            await c.post("/v1/runs/stream", json=BODY, headers=headers)
        assert [str(i.tenant_id) for i in runner.inputs] == ["from-token"]

    async def test_missing_token_is_a_401_challenge_without_error(
        self, idp: FakeIdp, tmp_path: Path
    ) -> None:
        async with _client(idp, tmp_path) as c:
            response = await c.post("/v1/runs/stream", json=BODY)
        assert response.status_code == HTTPStatus.UNAUTHORIZED
        assert response.headers["www-authenticate"] == "Bearer"
        assert response.json()["error"]["code"] == "authentication_required"

    async def test_a_tenant_header_alone_does_not_authenticate(
        self, idp: FakeIdp, tmp_path: Path
    ) -> None:
        async with _client(idp, tmp_path) as c:
            response = await c.post("/v1/runs/stream", json=BODY, headers={"X-Tenant-Id": "acme"})
        assert response.status_code == HTTPStatus.UNAUTHORIZED

    @pytest.mark.parametrize("scheme", ["Basic abc", "Bearer", "Bearer a b"])
    async def test_other_schemes_and_empty_bearer(
        self, idp: FakeIdp, tmp_path: Path, scheme: str
    ) -> None:
        async with _client(idp, tmp_path) as c:
            response = await c.post("/v1/runs/stream", json=BODY, headers={"Authorization": scheme})
        assert response.status_code == HTTPStatus.UNAUTHORIZED

    async def test_expired_token_is_a_401_invalid_token_without_the_token_in_the_body(
        self, idp: FakeIdp, tmp_path: Path
    ) -> None:
        token = idp.token(exp=1)
        async with _client(idp, tmp_path) as c:
            response = await c.post("/v1/runs/stream", json=BODY, headers=_bearer(token))
        assert response.status_code == HTTPStatus.UNAUTHORIZED
        assert response.headers["www-authenticate"] == 'Bearer error="invalid_token"'
        assert response.json()["error"]["code"] == "invalid_token"
        assert token not in response.text

    async def test_the_thread_run_route_also_requires_a_token(
        self, idp: FakeIdp, tmp_path: Path
    ) -> None:
        async with _client(idp, tmp_path) as c:
            response = await c.post(f"/v1/threads/{'a' * 32}/runs/stream", json={"message": "hi"})
        assert response.status_code == HTTPStatus.UNAUTHORIZED

    async def test_the_threads_crud_routes_require_a_token(
        self, idp: FakeIdp, tmp_path: Path
    ) -> None:
        async with _client(idp, tmp_path) as c:
            listing = await c.get("/v1/threads")
            created = await c.post("/v1/threads", json={"agent_id": "fake"})
            bad = await c.get("/v1/threads", headers=_bearer("garbage"))
        assert listing.status_code == created.status_code == HTTPStatus.UNAUTHORIZED
        assert bad.status_code == HTTPStatus.UNAUTHORIZED

    async def test_health_stays_public(self, idp: FakeIdp, tmp_path: Path) -> None:
        async with _client(idp, tmp_path) as c:
            assert (await c.get("/health")).status_code == HTTPStatus.OK


class TestTenantClaim:
    async def test_a_nested_claim_path(self, idp: FakeIdp, tmp_path: Path) -> None:
        runner = FakeRunner(OK_EVENTS)
        async with _client(idp, tmp_path, tenant_claim="org.tenant", runner=runner) as c:
            token = idp.token(tenant_id=None, org={"tenant": "nested-co"})
            response = await c.post("/v1/runs/stream", json=BODY, headers=_bearer(token))
        assert response.status_code == HTTPStatus.OK
        assert [str(i.tenant_id) for i in runner.inputs] == ["nested-co"]

    async def test_a_namespaced_claim_with_dots(self, idp: FakeIdp, tmp_path: Path) -> None:
        claim = "https://example.com/tenant"
        async with _client(idp, tmp_path, tenant_claim=claim) as c:
            token = idp.token(tenant_id=None, extra={claim: "ns-co"})
            response = await c.post("/v1/runs/stream", json=BODY, headers=_bearer(token))
        assert response.status_code == HTTPStatus.OK

    async def test_missing_claim_is_a_403(self, idp: FakeIdp, tmp_path: Path) -> None:
        async with _client(idp, tmp_path) as c:
            token = idp.token(tenant_id=None)
            response = await c.post("/v1/runs/stream", json=BODY, headers=_bearer(token))
        assert response.status_code == HTTPStatus.FORBIDDEN
        assert response.json()["error"]["code"] == "tenant_claim_missing"

    @pytest.mark.parametrize("value", ["a b", "../x", "x" * 65, "-lead", 42, ["acme"], ""])
    async def test_an_invalid_tenant_value_is_a_403(
        self, idp: FakeIdp, tmp_path: Path, value: Any
    ) -> None:
        async with _client(idp, tmp_path) as c:
            token = idp.token(tenant_id=value)
            response = await c.post("/v1/runs/stream", json=BODY, headers=_bearer(token))
        assert response.status_code == HTTPStatus.FORBIDDEN
        assert response.json()["error"]["code"] in {"tenant_claim_invalid", "tenant_claim_missing"}
        assert "a b" not in response.text

    async def test_a_missing_claim_is_logged_with_the_claim_name_only(
        self, idp: FakeIdp, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        token = idp.token(tenant_id=None)
        with caplog.at_level(logging.INFO):
            async with _client(idp, tmp_path) as c:
                response = await c.post("/v1/runs/stream", json=BODY, headers=_bearer(token))
        assert response.status_code == HTTPStatus.FORBIDDEN
        text = "\n".join(record.getMessage() for record in caplog.records)
        assert "'tenant_id' is missing" in text
        assert token not in text

    @pytest.mark.parametrize(
        ("value", "reason"),
        [
            ("Sneaky Tenant!", "is not a valid tenant id"),
            ("../etc", "is not a valid tenant id"),
            (42, "is not a string"),
            (["acme"], "is not a string"),
        ],
    )
    async def test_an_invalid_claim_is_logged_without_its_value(
        self,
        idp: FakeIdp,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        value: Any,
        reason: str,
    ) -> None:
        token = idp.token(tenant_id=value)
        with caplog.at_level(logging.INFO):
            async with _client(idp, tmp_path) as c:
                response = await c.post("/v1/runs/stream", json=BODY, headers=_bearer(token))
        assert response.status_code == HTTPStatus.FORBIDDEN
        records = [r for r in caplog.records if "Tenant claim rejected" in r.getMessage()]
        assert [r.levelno for r in records] == [logging.INFO]
        text = "\n".join(record.getMessage() for record in caplog.records)
        assert f"claim 'tenant_id' {reason}" in text
        for leaked in ("Sneaky", "../etc", "42", "acme", token):
            assert leaked not in text


class TestIdpUnavailable:
    async def test_unreachable_jwks_is_a_retryable_503_not_a_401(
        self, idp: FakeIdp, tmp_path: Path
    ) -> None:
        fetcher = ScriptedFetcher()
        fetcher.error = ConnectionError("secret-host.internal refused")
        source = HttpJwks(
            fetcher, ttl_seconds=600, min_refresh_interval_seconds=30, clock=ManualClock()
        )
        verifier = JwtVerifier(source, issuer=ISSUER, audience=AUDIENCE)
        resolver = JwtTenantResolver(verifier, "tenant_id")
        async with _client(idp, tmp_path, resolver=resolver) as c:
            response = await c.post("/v1/runs/stream", json=BODY, headers=_bearer(idp.token()))
        assert response.status_code == HTTPStatus.SERVICE_UNAVAILABLE
        assert response.headers["retry-after"]
        assert response.json()["error"]["code"] == "auth_unavailable"
        assert "secret-host" not in response.text


async def test_the_verified_subject_is_exposed_on_the_request_state(
    idp: FakeIdp, tmp_path: Path
) -> None:
    jwks = tmp_path / "jwks.json"
    jwks.write_text(json.dumps(idp.jwks()))
    resolver = JwtTenantResolver(
        JwtVerifier(FileJwks(jwks), issuer=ISSUER, audience=AUDIENCE), "tenant_id"
    )
    scope = {
        "type": "http",
        "headers": [(b"authorization", f"Bearer {idp.token(sub='user-42')}".encode())],
    }
    request = Request(scope)
    assert await resolver.resolve(request) == "acme"
    assert getattr(request.state, SUBJECT_STATE_KEY) == "user-42"


async def test_rejections_log_a_reason_never_the_token(
    idp: FakeIdp, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    token = idp.token(exp=1)
    with caplog.at_level(logging.DEBUG):
        async with _client(idp, tmp_path) as c:
            await c.post("/v1/runs/stream", json=BODY, headers=_bearer(token))
    text = "\n".join(record.getMessage() for record in caplog.records)
    assert "expired" in text
    assert token not in text
    assert "acme" not in text


@pytest.mark.parametrize(
    ("headers", "reason"),
    [
        ({}, "no bearer credentials"),
        ({"Authorization": "Basic abc"}, "no bearer credentials"),
        ({"Authorization": "Bearer "}, "malformed credentials"),
        ({"Authorization": "Bearer a b"}, "malformed credentials"),
        ({"Authorization": "Bearer " + "x" * 9000}, "malformed credentials"),
    ],
)
async def test_credential_parsing_rejections_are_logged_without_the_token(
    idp: FakeIdp,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    headers: dict[str, str],
    reason: str,
) -> None:
    with caplog.at_level(logging.INFO):
        async with _client(idp, tmp_path) as c:
            response = await c.post("/v1/runs/stream", json=BODY, headers=headers)
    assert response.status_code == 401
    text = "\n".join(record.getMessage() for record in caplog.records)
    assert reason in text
    assert "xxxxx" not in text


class TestHeaderModeIsUnchanged:
    async def test_header_mode_still_works(self) -> None:
        app = create_app(make_settings())
        runner = FakeRunner(OK_EVENTS)
        app.dependency_overrides[get_services] = lambda: _services(runner)
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c,
        ):
            ok = await c.post("/v1/runs/stream", json=BODY, headers={"X-Tenant-Id": "acme"})
            missing = await c.post("/v1/runs/stream", json=BODY)
        assert ok.status_code == HTTPStatus.OK
        assert missing.status_code == HTTPStatus.BAD_REQUEST
        assert missing.json()["error"]["code"] == "tenant_required"
