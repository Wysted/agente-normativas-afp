import json
import logging
from pathlib import Path

import pytest
from pydantic import ValidationError

from agintis.bootstrap.settings import AuthSettings
from agintis.bootstrap.tenancy import build_tenant_resolver
from agintis.platform.auth import AuthConfigError
from agintis.platform.http.jwt_tenant import JwtTenantResolver
from agintis.platform.http.tenant import HeaderTenantResolver
from tests.support.auth import AUDIENCE, ISSUER, FakeIdp
from tests.support.settings import make_settings

JWT = {"tenant_resolver": "jwt", "issuer": ISSUER, "audience": AUDIENCE}


def _auth(**values: object) -> AuthSettings:
    return AuthSettings(_env_file=None, **values)  # type: ignore[arg-type]


def test_defaults_keep_the_header_behaviour() -> None:
    auth = _auth()
    assert auth.tenant_resolver == "header"
    assert auth.algorithms == ["RS256", "ES256"]
    assert auth.tenant_claim == "tenant_id"


def test_jwt_needs_issuer_audience_and_exactly_one_jwks_source() -> None:
    with pytest.raises(ValidationError, match="AUTH_ISSUER, AUTH_AUDIENCE"):
        _auth(tenant_resolver="jwt", jwks_url="https://idp/jwks")
    with pytest.raises(ValidationError, match="exactly one"):
        _auth(**JWT)
    with pytest.raises(ValidationError, match="exactly one"):
        _auth(**JWT, jwks_url="https://idp/jwks", jwks_file="x.json")
    assert _auth(**JWT, jwks_url="https://idp/jwks").jwks_url == "https://idp/jwks"


def test_empty_strings_mean_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTH_JWKS_URL", "")
    monkeypatch.setenv("AUTH_JWKS_FILE", "")
    monkeypatch.setenv("AUTH_ISSUER", "")
    auth = _auth()
    assert auth.jwks_url is None
    assert auth.jwks_file is None
    assert auth.issuer is None


@pytest.mark.parametrize("value", ["none", "HS256", "RS256,HS384", "RS256,none", ""])
def test_symmetric_and_none_algorithms_are_refused(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("AUTH_ALGORITHMS", value)
    with pytest.raises(ValidationError, match="asymmetric"):
        _auth()


def test_algorithms_are_read_comma_separated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTH_ALGORITHMS", "RS256, ES384")
    assert _auth().algorithms == ["RS256", "ES384"]


def test_the_refresh_interval_cannot_exceed_the_ttl() -> None:
    with pytest.raises(ValidationError, match="MIN_REFRESH"):
        _auth(jwks_cache_ttl_seconds=10, jwks_min_refresh_seconds=20)


def test_new_bounds_and_defaults() -> None:
    auth = _auth()
    assert auth.jwks_stale_grace_seconds == 3600
    assert auth.max_token_lifetime_seconds == 86400
    assert auth.jwks_allow_insecure is False
    for bad in (
        {"jwks_cache_ttl_seconds": 86401},
        {"jwks_min_refresh_seconds": 4},
        {"jwks_stale_grace_seconds": -1},
        {"jwks_stale_grace_seconds": 86401},
        {"max_token_lifetime_seconds": 10},
    ):
        with pytest.raises(ValidationError):
            _auth(**bad)
    assert _auth(jwks_stale_grace_seconds=0).jwks_stale_grace_seconds == 0


def test_the_jwks_url_is_validated_and_kept_verbatim() -> None:
    with pytest.raises(ValidationError):
        _auth(jwks_url="ftp://idp/jwks")
    assert _auth(jwks_url="https://idp:443/jwks/").jwks_url == "https://idp:443/jwks/"


class TestBuildTenantResolver:
    def test_header_mode_warns_that_it_trusts_the_client(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            resolver = build_tenant_resolver(make_settings())
        assert isinstance(resolver, HeaderTenantResolver)
        assert any("trusted gateway" in r.getMessage() for r in caplog.records)

    def test_jwt_mode_builds_the_jwt_resolver_without_the_header_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        jwks = tmp_path / "jwks.json"
        jwks.write_text(json.dumps(FakeIdp().jwks()))
        settings = make_settings(auth={**JWT, "jwks_file": jwks})
        with caplog.at_level(logging.WARNING):
            resolver = build_tenant_resolver(settings)
        assert isinstance(resolver, JwtTenantResolver)
        assert not any("trusted gateway" in r.getMessage() for r in caplog.records)

    def test_an_unreadable_jwks_file_fails_the_boot(self, tmp_path: Path) -> None:
        settings = make_settings(auth={**JWT, "jwks_file": tmp_path / "missing.json"})
        with pytest.raises(AuthConfigError):
            build_tenant_resolver(settings)

    def test_a_plain_http_jwks_url_is_refused_at_boot(self) -> None:
        settings = make_settings(auth={**JWT, "jwks_url": "http://idp.internal/jwks"})
        with pytest.raises(AuthConfigError, match="must be https"):
            build_tenant_resolver(settings)

    async def test_plain_http_is_accepted_with_the_opt_in_and_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        settings = make_settings(
            auth={**JWT, "jwks_url": "http://idp.internal/jwks", "jwks_allow_insecure": True}
        )
        with caplog.at_level(logging.WARNING):
            resolver = build_tenant_resolver(settings)
        assert any("not https" in r.getMessage() for r in caplog.records)
        await resolver.aclose()

    async def test_plain_http_is_accepted_with_an_explicit_local_env(self) -> None:
        settings = make_settings(
            app_env="local", auth={**JWT, "jwks_url": "http://idp.internal/jwks"}
        )
        await build_tenant_resolver(settings).aclose()

    @pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "[::1]"])
    async def test_loopback_http_is_always_accepted(self, host: str) -> None:
        settings = make_settings(auth={**JWT, "jwks_url": f"http://{host}:8080/jwks"})
        await build_tenant_resolver(settings).aclose()

    async def test_https_is_accepted_without_warnings(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        settings = make_settings(auth={**JWT, "jwks_url": "https://idp.example/jwks"})
        with caplog.at_level(logging.WARNING):
            await build_tenant_resolver(settings).aclose()
        assert not caplog.records

    def test_a_jwks_file_always_warns_at_boot(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        jwks = tmp_path / "jwks.json"
        jwks.write_text(json.dumps(FakeIdp().jwks()))
        with caplog.at_level(logging.WARNING):
            build_tenant_resolver(make_settings(auth={**JWT, "jwks_file": jwks}))
        assert any("static key set" in r.getMessage() for r in caplog.records)

    def test_dev_kids_in_a_jwks_file_are_refused_outside_explicit_local(
        self, tmp_path: Path
    ) -> None:
        jwks = tmp_path / "jwks.json"
        jwks.write_text(json.dumps(FakeIdp("dev-rs256").jwks()))
        auth = {**JWT, "jwks_file": jwks}
        with pytest.raises(AuthConfigError, match="development keys"):
            build_tenant_resolver(make_settings(auth=auth))  # implicit "local" does not count
        with pytest.raises(AuthConfigError, match="development keys"):
            build_tenant_resolver(make_settings(app_env="prod", auth=auth))
        assert isinstance(
            build_tenant_resolver(make_settings(app_env="local", auth=auth)), JwtTenantResolver
        )

    def test_default_tenant_is_ignored_in_jwt_mode(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        jwks = tmp_path / "jwks.json"
        jwks.write_text(json.dumps(FakeIdp().jwks()))
        settings = make_settings(
            app_env="local", http={"default_tenant": "dev"}, auth={**JWT, "jwks_file": jwks}
        )
        with caplog.at_level(logging.WARNING):
            build_tenant_resolver(settings)
        assert any("HTTP_DEFAULT_TENANT is ignored" in r.getMessage() for r in caplog.records)
