"""JWT verification and JWKS caching (no network: scripted fetchers, manual clock)."""

import base64
import hashlib
import hmac
import json
import time
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives import serialization

from agintis.platform.auth import (
    AuthConfigError,
    AuthUnavailableError,
    FileJwks,
    HttpJwks,
    InvalidTokenError,
    JwtVerifier,
    parse_jwks,
)
from agintis.platform.auth.jwks import JwksSource
from tests.support.auth import AUDIENCE, ISSUER, FakeIdp, ManualClock, ScriptedFetcher

TTL = 600.0
MIN_REFRESH = 30.0


@pytest.fixture(scope="module")
def idp() -> FakeIdp:
    return FakeIdp()


@pytest.fixture(scope="module")
def other_idp() -> FakeIdp:
    return FakeIdp("key-2")


def _file_source(tmp_path: Path, *idps: FakeIdp) -> FileJwks:
    path = tmp_path / "jwks.json"
    path.write_text(json.dumps({"keys": [i.jwk() for i in idps]}))
    return FileJwks(path)


def _verifier(source: JwksSource, **kwargs: Any) -> JwtVerifier:
    return JwtVerifier(source, issuer=ISSUER, audience=AUDIENCE, **kwargs)


async def _reason(verifier: JwtVerifier, token: str) -> str:
    with pytest.raises(InvalidTokenError) as info:
        await verifier.verify(token)
    assert not token or token not in info.value.message
    assert info.value.headers["WWW-Authenticate"] == 'Bearer error="invalid_token"'
    return info.value.reason


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unsigned(claims: dict[str, Any], alg: str, kid: str, signature: bytes = b"") -> str:
    header = _b64(json.dumps({"alg": alg, "kid": kid, "typ": "JWT"}).encode())
    return f"{header}.{_b64(json.dumps(claims).encode())}.{_b64(signature)}"


class TestVerification:
    async def test_a_valid_token_yields_its_claims(self, idp: FakeIdp, tmp_path: Path) -> None:
        verified = await _verifier(_file_source(tmp_path, idp)).verify(idp.token())
        assert verified.claims["tenant_id"] == "acme"
        assert verified.subject == "user-1"
        assert "acme" not in repr(verified)  # claims never reach a repr/log

    async def test_ec_keys_and_algorithms_work(self, tmp_path: Path) -> None:
        ec_idp = FakeIdp("ec-1", algorithm="ES256")
        verifier = _verifier(_file_source(tmp_path, ec_idp))
        assert (await verifier.verify(ec_idp.token())).subject == "user-1"

    async def test_expired(self, idp: FakeIdp, tmp_path: Path) -> None:
        verifier = _verifier(_file_source(tmp_path, idp), leeway_seconds=5)
        assert await _reason(verifier, idp.token(exp=int(time.time()) - 60)) == "expired"

    async def test_leeway_accepts_a_just_expired_token(self, idp: FakeIdp, tmp_path: Path) -> None:
        verifier = _verifier(_file_source(tmp_path, idp), leeway_seconds=30)
        await verifier.verify(idp.token(exp=int(time.time()) - 5))

    async def test_not_yet_valid(self, idp: FakeIdp, tmp_path: Path) -> None:
        verifier = _verifier(_file_source(tmp_path, idp))
        await _reason(verifier, idp.token(nbf=int(time.time()) + 600))

    @pytest.mark.parametrize("claim", ["iss", "aud", "exp"])
    async def test_registered_claims_are_required(
        self, idp: FakeIdp, tmp_path: Path, claim: str
    ) -> None:
        await _reason(_verifier(_file_source(tmp_path, idp)), idp.token(**{claim: None}))

    async def test_wrong_issuer(self, idp: FakeIdp, tmp_path: Path) -> None:
        await _reason(_verifier(_file_source(tmp_path, idp)), idp.token(iss="https://evil"))

    async def test_wrong_audience(self, idp: FakeIdp, tmp_path: Path) -> None:
        await _reason(_verifier(_file_source(tmp_path, idp)), idp.token(aud="someone-else"))

    async def test_audience_may_be_a_list(self, idp: FakeIdp, tmp_path: Path) -> None:
        verifier = _verifier(_file_source(tmp_path, idp))
        await verifier.verify(idp.token(aud=["other", AUDIENCE]))

    async def test_signature_of_another_key(
        self, idp: FakeIdp, other_idp: FakeIdp, tmp_path: Path
    ) -> None:
        verifier = _verifier(_file_source(tmp_path, idp))
        assert await _reason(verifier, other_idp.token(kid=idp.kid)) == "InvalidSignatureError"

    async def test_alg_none_is_rejected(self, idp: FakeIdp, tmp_path: Path) -> None:
        token = _unsigned(idp.claims(), "none", idp.kid)
        assert (
            await _reason(_verifier(_file_source(tmp_path, idp)), token) == "algorithm not allowed"
        )

    async def test_hs256_signed_with_the_public_key_is_rejected(
        self, idp: FakeIdp, tmp_path: Path
    ) -> None:
        """Algorithm confusion: the public key (or its PEM) used as an HMAC secret."""
        pem = idp.private_key.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        )
        header = _b64(json.dumps({"alg": "HS256", "kid": idp.kid, "typ": "JWT"}).encode())
        body = _b64(json.dumps(idp.claims()).encode())
        signature = hmac.new(pem, f"{header}.{body}".encode(), hashlib.sha256).digest()
        token = f"{header}.{body}.{_b64(signature)}"
        verifier = _verifier(_file_source(tmp_path, idp))
        assert await _reason(verifier, token) == "algorithm not allowed"

    async def test_an_algorithm_outside_the_allowlist_is_rejected(
        self, idp: FakeIdp, tmp_path: Path
    ) -> None:
        verifier = _verifier(_file_source(tmp_path, idp), algorithms=["ES256"])
        assert await _reason(verifier, idp.token()) == "algorithm not allowed"

    async def test_the_key_must_match_the_algorithm(self, tmp_path: Path) -> None:
        ec_idp = FakeIdp("ec-1", algorithm="ES256")
        jwk = ec_idp.jwk(with_alg=False)
        path = tmp_path / "jwks.json"
        path.write_text(json.dumps({"keys": [jwk]}))
        rsa_signed = FakeIdp("ec-1").token()  # RS256 token naming the EC key's kid
        verifier = _verifier(FileJwks(path))
        await _reason(verifier, rsa_signed)

    async def test_a_key_pinned_to_another_algorithm_is_rejected(self, tmp_path: Path) -> None:
        pinned = FakeIdp("k", algorithm="ES256")
        verifier = _verifier(_file_source(tmp_path, pinned), algorithms=["RS256", "ES256"])
        assert await _reason(verifier, FakeIdp("k").token()) == "algorithm differs from the key's"

    @pytest.mark.parametrize("token", ["", "abc", "a.b.c", "a.b", "....", "é.x.y"])
    async def test_malformed_tokens(self, idp: FakeIdp, tmp_path: Path, token: str) -> None:
        await _reason(_verifier(_file_source(tmp_path, idp)), token)

    async def test_a_token_without_kid_is_rejected(self, idp: FakeIdp, tmp_path: Path) -> None:
        import jwt

        token = jwt.encode(idp.claims(), idp.private_key, algorithm="RS256")
        assert await _reason(_verifier(_file_source(tmp_path, idp)), token) == "missing kid"

    async def test_unknown_kid(self, idp: FakeIdp, tmp_path: Path) -> None:
        assert await _reason(_verifier(_file_source(tmp_path, idp)), idp.token(kid="x")) == (
            "unknown kid"
        )

    @pytest.mark.parametrize("algorithms", [["none"], ["HS256"], ["RS256", "HS512"], []])
    def test_the_allowlist_itself_is_validated(self, idp: FakeIdp, algorithms: list[str]) -> None:
        with pytest.raises(ValueError, match="allowlist"):
            JwtVerifier(FileJwks.__new__(FileJwks), issuer="i", audience="a", algorithms=algorithms)


class TestFileJwks:
    def test_unusable_files_fail_at_startup(self, tmp_path: Path) -> None:
        path = tmp_path / "jwks.json"
        with pytest.raises(AuthConfigError, match="Cannot load"):
            FileJwks(path)
        path.write_text("not json")
        with pytest.raises(AuthConfigError):
            FileJwks(path)
        path.write_text('{"keys": []}')
        with pytest.raises(AuthConfigError, match="no usable"):
            FileJwks(path)

    def test_parse_skips_entries_it_cannot_use(self, idp: FakeIdp) -> None:
        enc = {**idp.jwk(), "kid": "enc", "use": "enc"}
        nokid = {k: v for k, v in idp.jwk().items() if k != "kid"}
        junk = {"kty": "oct", "kid": "sym", "k": "AAAA"}
        keys = parse_jwks({"keys": [enc, nokid, junk, "x", idp.jwk(), {**idp.jwk(), "kid": "dup"}]})
        assert set(keys) == {idp.kid, "dup"}


def _http_jwks(fetcher: ScriptedFetcher, clock: ManualClock) -> HttpJwks:
    return HttpJwks(fetcher, ttl_seconds=TTL, min_refresh_interval_seconds=MIN_REFRESH, clock=clock)


class TestHttpJwks:
    async def test_keys_are_fetched_once_and_cached(self, idp: FakeIdp) -> None:
        fetcher, clock = ScriptedFetcher(idp.jwks()), ManualClock()
        source = _http_jwks(fetcher, clock)
        for _ in range(3):
            assert await source.get_key(idp.kid) is not None
        assert fetcher.calls == 1

    async def test_the_cache_expires_after_the_ttl(self, idp: FakeIdp) -> None:
        fetcher, clock = ScriptedFetcher(idp.jwks()), ManualClock()
        source = _http_jwks(fetcher, clock)
        await source.get_key(idp.kid)
        clock.advance(TTL + 1)
        await source.get_key(idp.kid)
        assert fetcher.calls == 2

    async def test_an_unknown_kid_refreshes_once_then_accepts_the_rotated_key(
        self, idp: FakeIdp, other_idp: FakeIdp
    ) -> None:
        fetcher, clock = ScriptedFetcher(idp.jwks()), ManualClock()
        source = _http_jwks(fetcher, clock)
        await source.get_key(idp.kid)
        fetcher.document = {"keys": [idp.jwk(), other_idp.jwk()]}  # the IdP rotated
        clock.advance(MIN_REFRESH + 1)
        assert await source.get_key(other_idp.kid) is not None
        assert fetcher.calls == 2

    async def test_an_unknown_kid_that_stays_unknown_is_rejected_after_one_refresh(
        self, idp: FakeIdp
    ) -> None:
        fetcher, clock = ScriptedFetcher(idp.jwks()), ManualClock()
        source = _http_jwks(fetcher, clock)
        await source.get_key(idp.kid)
        clock.advance(MIN_REFRESH + 1)
        assert await source.get_key("nope") is None
        assert fetcher.calls == 2

    async def test_random_kids_cannot_flood_the_idp(self, idp: FakeIdp) -> None:
        fetcher, clock = ScriptedFetcher(idp.jwks()), ManualClock()
        source = _http_jwks(fetcher, clock)
        await source.get_key(idp.kid)
        for i in range(50):
            assert await source.get_key(f"random-{i}") is None  # inside the interval
        assert fetcher.calls == 1
        clock.advance(MIN_REFRESH + 1)
        for i in range(50):
            assert await source.get_key(f"again-{i}") is None
        assert fetcher.calls == 2

    async def test_concurrent_callers_share_one_fetch(self, idp: FakeIdp) -> None:
        import asyncio

        fetcher, clock = ScriptedFetcher(idp.jwks()), ManualClock()
        source = _http_jwks(fetcher, clock)
        results = await asyncio.gather(*(source.get_key(idp.kid) for _ in range(20)))
        assert all(result is not None for result in results)
        assert fetcher.calls == 1

    async def test_an_unreachable_idp_is_unavailable_not_invalid(self, idp: FakeIdp) -> None:
        fetcher, clock = ScriptedFetcher(idp.jwks()), ManualClock()
        fetcher.error = ConnectionError("idp down")
        source = _http_jwks(fetcher, clock)
        with pytest.raises(AuthUnavailableError) as info:
            await source.get_key(idp.kid)
        assert "idp down" not in info.value.message
        assert info.value.headers["Retry-After"]
        with pytest.raises(AuthUnavailableError):  # inside the interval: no new fetch
            await source.get_key(idp.kid)
        assert fetcher.calls == 1
        fetcher.error = None
        clock.advance(MIN_REFRESH + 1)
        assert await source.get_key(idp.kid) is not None

    async def test_expired_keys_are_not_served_when_the_refresh_fails(self, idp: FakeIdp) -> None:
        fetcher, clock = ScriptedFetcher(idp.jwks()), ManualClock()
        source = _http_jwks(fetcher, clock)
        await source.get_key(idp.kid)
        fetcher.error = TimeoutError()
        clock.advance(TTL + 1)
        with pytest.raises(AuthUnavailableError):
            await source.get_key(idp.kid)

    async def test_a_document_without_usable_keys_is_unavailable(self) -> None:
        source = _http_jwks(ScriptedFetcher({"keys": []}), ManualClock())
        with pytest.raises(AuthUnavailableError):
            await source.get_key("x")

    async def test_a_known_kid_survives_a_failed_refresh_inside_the_ttl(self, idp: FakeIdp) -> None:
        fetcher, clock = ScriptedFetcher(idp.jwks()), ManualClock()
        source = _http_jwks(fetcher, clock)
        await source.get_key(idp.kid)
        fetcher.error = ConnectionError()
        clock.advance(MIN_REFRESH + 1)
        with pytest.raises(AuthUnavailableError):
            await source.get_key("unknown")  # the refresh for the unknown kid fails: 503
        assert await source.get_key(idp.kid) is not None  # known kid, cache still fresh
