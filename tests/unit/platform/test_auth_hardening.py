"""Review hardening of the JWT/JWKS layer: odd JWKS entries, fetch deadlines, stale grace,
the HTTP fetcher, unusual JWS headers and time-claim types (no real network)."""

import asyncio
import base64
import json
import logging
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx
import pytest

from agintis.platform.auth import (
    AuthUnavailableError,
    FileJwks,
    HttpJwks,
    HttpxJwksFetcher,
    InvalidTokenError,
    JwtVerifier,
    parse_jwks,
)
from tests.support.auth import AUDIENCE, ISSUER, FakeIdp, ManualClock, ScriptedFetcher

TTL = 600.0
MIN_REFRESH = 30.0
GRACE = 3600.0


@pytest.fixture(scope="module")
def idp() -> FakeIdp:
    return FakeIdp()


@pytest.fixture(scope="module")
def other_idp() -> FakeIdp:
    return FakeIdp("key-2")


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


class TestOddJwksEntries:
    def _mixed(self, idp: FakeIdp) -> dict[str, Any]:
        return {
            "keys": [
                {**idp.jwk(), "kid": "none", "alg": "none"},
                {**idp.jwk(), "kid": "hs", "alg": "HS256"},
                {**idp.jwk(), "kid": "weird", "alg": "XYZ"},
                {**idp.jwk(), "kid": "intalg", "alg": 5},
                {"kty": "weird", "kid": "kty"},
                {"kty": "RSA", "kid": "broken", "n": "!!", "e": "AQAB"},
                {"kty": "oct", "kid": "sym", "k": "AAAA", "alg": "HS256"},
                idp.jwk(),
            ]
        }

    def test_one_odd_entry_never_kills_the_key_set(
        self, idp: FakeIdp, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            keys = parse_jwks(self._mixed(idp))
        assert set(keys) == {idp.kid}
        text = " ".join(r.getMessage() for r in caplog.records)
        assert "none" in text
        assert "broken" in text

    def test_a_file_with_an_odd_entry_boots(self, idp: FakeIdp, tmp_path: Path) -> None:
        path = tmp_path / "jwks.json"
        path.write_text(json.dumps(self._mixed(idp)))
        assert FileJwks(path).kids == {idp.kid}

    def test_a_non_object_document_is_a_value_error(self) -> None:
        with pytest.raises(ValueError, match="not an object"):
            parse_jwks([])  # type: ignore[arg-type]

    async def test_http_jwks_survives_an_odd_entry(self, idp: FakeIdp) -> None:
        source = HttpJwks(
            ScriptedFetcher(self._mixed(idp)),
            ttl_seconds=TTL,
            min_refresh_interval_seconds=MIN_REFRESH,
            clock=ManualClock(),
        )
        assert await source.get_key(idp.kid) is not None


class SlowFetcher:
    def __init__(self, document: Mapping[str, Any], delay: float) -> None:
        self.document = document
        self.delay = delay
        self.calls = 0

    async def __call__(self) -> Mapping[str, Any]:
        self.calls += 1
        await asyncio.sleep(self.delay)
        return self.document


class TestFetchDeadline:
    async def test_a_slow_fetch_is_cut_at_the_deadline(self, idp: FakeIdp) -> None:
        fetcher = SlowFetcher(idp.jwks(), delay=5)
        source = HttpJwks(
            fetcher,
            ttl_seconds=TTL,
            min_refresh_interval_seconds=MIN_REFRESH,
            fetch_timeout_seconds=0.05,
            clock=ManualClock(),
        )
        started = time.monotonic()
        with pytest.raises(AuthUnavailableError):
            await source.get_key(idp.kid)
        assert time.monotonic() - started < 1

    async def test_waiters_do_not_queue_behind_a_stuck_leader(self, idp: FakeIdp) -> None:
        fetcher = SlowFetcher(idp.jwks(), delay=5)
        source = HttpJwks(
            fetcher,
            ttl_seconds=TTL,
            min_refresh_interval_seconds=MIN_REFRESH,
            fetch_timeout_seconds=0.05,
            clock=ManualClock(),
        )
        started = time.monotonic()
        results = await asyncio.gather(
            *(source.get_key(idp.kid) for _ in range(20)), return_exceptions=True
        )
        assert all(isinstance(r, AuthUnavailableError) for r in results)
        assert time.monotonic() - started < 1
        assert fetcher.calls == 1  # one bounded fetch, not one per waiter


def _grace_source(fetcher: ScriptedFetcher, clock: ManualClock, grace: float = GRACE) -> HttpJwks:
    return HttpJwks(
        fetcher,
        ttl_seconds=TTL,
        min_refresh_interval_seconds=MIN_REFRESH,
        stale_grace_seconds=grace,
        clock=clock,
    )


class TestStaleGrace:
    @pytest.mark.parametrize(
        "error",
        [
            ConnectionError("down"),
            TimeoutError(),
            httpx.HTTPStatusError(
                "500", request=httpx.Request("GET", "https://i"), response=httpx.Response(500)
            ),
            ValueError("malformed"),
        ],
    )
    async def test_cached_keys_are_served_while_the_refresh_fails(
        self, idp: FakeIdp, error: Exception, caplog: pytest.LogCaptureFixture
    ) -> None:
        fetcher, clock = ScriptedFetcher(idp.jwks()), ManualClock()
        source = _grace_source(fetcher, clock)
        await source.get_key(idp.kid)
        fetcher.error = error
        clock.advance(TTL + 1)
        with caplog.at_level(logging.WARNING):
            assert await source.get_key(idp.kid) is not None
        assert any("STALE" in r.getMessage() for r in caplog.records)

    async def test_a_malformed_or_empty_document_counts_as_a_failure(self, idp: FakeIdp) -> None:
        fetcher, clock = ScriptedFetcher(idp.jwks()), ManualClock()
        source = _grace_source(fetcher, clock)
        await source.get_key(idp.kid)
        clock.advance(TTL + 1)
        fetcher.document = {"keys": []}
        assert await source.get_key(idp.kid) is not None
        fetcher.document = {"nope": 1}
        clock.advance(MIN_REFRESH + 1)
        assert await source.get_key(idp.kid) is not None

    async def test_stale_keys_are_never_used_when_the_refresh_succeeds(
        self, idp: FakeIdp, other_idp: FakeIdp
    ) -> None:
        fetcher, clock = ScriptedFetcher(idp.jwks()), ManualClock()
        source = _grace_source(fetcher, clock)
        await source.get_key(idp.kid)
        fetcher.document = other_idp.jwks()  # the IdP dropped the first key
        clock.advance(TTL + 1)
        assert await source.get_key(idp.kid) is None
        assert await source.get_key(other_idp.kid) is not None

    async def test_after_the_grace_it_is_a_503(self, idp: FakeIdp) -> None:
        fetcher, clock = ScriptedFetcher(idp.jwks()), ManualClock()
        source = _grace_source(fetcher, clock)
        await source.get_key(idp.kid)
        fetcher.error = ConnectionError()
        clock.advance(TTL + GRACE - 1)
        assert await source.get_key(idp.kid) is not None
        clock.advance(2)
        with pytest.raises(AuthUnavailableError):
            await source.get_key(idp.kid)

    async def test_grace_zero_disables_it(self, idp: FakeIdp) -> None:
        fetcher, clock = ScriptedFetcher(idp.jwks()), ManualClock()
        source = _grace_source(fetcher, clock, grace=0)
        await source.get_key(idp.kid)
        fetcher.error = ConnectionError()
        clock.advance(TTL + 1)
        with pytest.raises(AuthUnavailableError):
            await source.get_key(idp.kid)

    async def test_an_unknown_kid_is_a_503_not_a_401_while_stale(self, idp: FakeIdp) -> None:
        fetcher, clock = ScriptedFetcher(idp.jwks()), ManualClock()
        source = _grace_source(fetcher, clock)
        await source.get_key(idp.kid)
        fetcher.error = ConnectionError()
        clock.advance(TTL + 1)
        with pytest.raises(AuthUnavailableError):
            await source.get_key("rotated")

    async def test_the_warning_is_rate_limited_and_fetches_stay_rate_limited(
        self, idp: FakeIdp, caplog: pytest.LogCaptureFixture
    ) -> None:
        fetcher, clock = ScriptedFetcher(idp.jwks()), ManualClock()
        source = _grace_source(fetcher, clock)
        await source.get_key(idp.kid)
        fetcher.error = ConnectionError()
        clock.advance(TTL + 1)
        with caplog.at_level(logging.WARNING):
            for _ in range(30):
                assert await source.get_key(idp.kid) is not None
        stale = [r for r in caplog.records if "STALE" in r.getMessage()]
        assert len(stale) == 1
        assert fetcher.calls == 2  # initial + one failed refresh inside the interval

    async def test_a_recovered_idp_ends_the_stale_period(self, idp: FakeIdp) -> None:
        fetcher, clock = ScriptedFetcher(idp.jwks()), ManualClock()
        source = _grace_source(fetcher, clock)
        await source.get_key(idp.kid)
        fetcher.error = ConnectionError()
        clock.advance(TTL + 1)
        await source.get_key(idp.kid)
        fetcher.error = None
        clock.advance(MIN_REFRESH + 1)
        assert await source.get_key(idp.kid) is not None
        assert fetcher.calls == 3

    async def test_a_slow_leader_serves_stale_to_waiters(self, idp: FakeIdp) -> None:
        clock = ManualClock()
        fetcher = SlowFetcher(idp.jwks(), delay=0)
        source = HttpJwks(
            fetcher,
            ttl_seconds=TTL,
            min_refresh_interval_seconds=MIN_REFRESH,
            fetch_timeout_seconds=0.05,
            stale_grace_seconds=GRACE,
            clock=clock,
        )
        await source.get_key(idp.kid)
        fetcher.delay = 5
        clock.advance(TTL + 1)
        results = await asyncio.gather(*(source.get_key(idp.kid) for _ in range(5)))
        assert all(r is not None for r in results)


def _transport(handler: Any) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


class TestHttpxFetcher:
    def _fetcher(self, handler: Any) -> HttpxJwksFetcher:
        return HttpxJwksFetcher("https://idp.example/jwks", 2.0, transport=_transport(handler))

    async def test_returns_the_json_object(self, idp: FakeIdp) -> None:
        fetcher = self._fetcher(lambda r: httpx.Response(200, json=idp.jwks()))
        assert (await fetcher())["keys"][0]["kid"] == idp.kid
        await fetcher.aclose()

    @pytest.mark.parametrize("status", [301, 302, 307, 308])
    async def test_redirects_are_not_followed(self, status: int) -> None:
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(str(request.url))
            return httpx.Response(status, headers={"Location": "https://evil.example/jwks"})

        with pytest.raises(httpx.HTTPStatusError):
            await self._fetcher(handler)()
        assert seen == ["https://idp.example/jwks"]

    @pytest.mark.parametrize("status", [400, 401, 404, 500, 503])
    async def test_non_2xx_fails(self, status: int) -> None:
        with pytest.raises(httpx.HTTPStatusError):
            await self._fetcher(lambda r: httpx.Response(status, json={"keys": []}))()

    async def test_a_body_over_one_mib_is_refused(self) -> None:
        big = b" " * (1024 * 1024 + 1)
        with pytest.raises(ValueError, match="too large"):
            await self._fetcher(lambda r: httpx.Response(200, content=big))()

    @pytest.mark.parametrize("body", [b"[1]", b'"x"', b"7", b"null"])
    async def test_non_object_json_is_refused(self, body: bytes) -> None:
        with pytest.raises(ValueError, match="not a JSON object"):
            await self._fetcher(lambda r: httpx.Response(200, content=body))()

    async def test_invalid_json_is_refused(self) -> None:
        with pytest.raises(json.JSONDecodeError):
            await self._fetcher(lambda r: httpx.Response(200, content=b"<html>"))()

    async def test_every_failure_is_unavailable_through_http_jwks(self) -> None:
        fetcher = self._fetcher(lambda r: httpx.Response(302))
        source = HttpJwks(
            fetcher, ttl_seconds=TTL, min_refresh_interval_seconds=MIN_REFRESH, clock=ManualClock()
        )
        with pytest.raises(AuthUnavailableError):
            await source.get_key("x")


def _verifier(idp: FakeIdp, tmp_path: Path, **kwargs: Any) -> JwtVerifier:
    path = tmp_path / "jwks.json"
    path.write_text(json.dumps(idp.jwks()))
    return JwtVerifier(FileJwks(path), issuer=ISSUER, audience=AUDIENCE, **kwargs)


class TestJwsHeaders:
    async def test_an_unknown_critical_header_is_rejected(
        self, idp: FakeIdp, tmp_path: Path
    ) -> None:
        token = idp.token(headers={"crit": ["x-custom"], "x-custom": "1"})
        with pytest.raises(InvalidTokenError):
            await _verifier(idp, tmp_path).verify(token)

    async def test_an_embedded_jwk_is_ignored(
        self, idp: FakeIdp, other_idp: FakeIdp, tmp_path: Path
    ) -> None:
        """Signed by the attacker's key, which the header offers as `jwk`: the JWKS key wins."""
        attacker = FakeIdp(idp.kid)
        token = attacker.token(headers={"jwk": attacker.jwk()})
        with pytest.raises(InvalidTokenError) as info:
            await _verifier(idp, tmp_path).verify(token)
        assert info.value.reason == "InvalidSignatureError"

    async def test_an_embedded_jku_is_ignored(self, idp: FakeIdp, tmp_path: Path) -> None:
        attacker = FakeIdp(idp.kid)
        token = attacker.token(headers={"jku": "https://evil.example/jwks.json"})
        with pytest.raises(InvalidTokenError) as info:
            await _verifier(idp, tmp_path).verify(token)
        assert info.value.reason == "InvalidSignatureError"

    async def test_a_valid_token_with_a_jku_header_is_still_accepted(
        self, idp: FakeIdp, tmp_path: Path
    ) -> None:
        token = idp.token(headers={"jku": "https://evil.example/jwks.json"})
        assert (await _verifier(idp, tmp_path).verify(token)).subject == "user-1"


class TestTimeClaims:
    @pytest.mark.parametrize("claim", ["exp", "nbf", "iat"])
    @pytest.mark.parametrize("value", ["9999999999", True, [1], {"a": 1}])
    async def test_non_numeric_time_claims_are_rejected(
        self, idp: FakeIdp, tmp_path: Path, claim: str, value: object
    ) -> None:
        with pytest.raises(InvalidTokenError):
            await _verifier(idp, tmp_path).verify(idp.token(**{claim: value}))  # type: ignore[arg-type]

    async def test_a_huge_exp_is_rejected(self, idp: FakeIdp, tmp_path: Path) -> None:
        with pytest.raises(InvalidTokenError):
            await _verifier(idp, tmp_path).verify(idp.token(exp=int(time.time()) + 10**9))

    async def test_a_long_lifetime_with_iat_is_rejected(self, idp: FakeIdp, tmp_path: Path) -> None:
        now = int(time.time())
        verifier = _verifier(idp, tmp_path, max_token_lifetime_seconds=3600)
        with pytest.raises(InvalidTokenError) as info:
            await verifier.verify(idp.token(iat=now - 100, exp=now + 7200))
        assert info.value.reason == "lifetime too long"

    async def test_without_iat_the_remaining_time_is_capped(
        self, idp: FakeIdp, tmp_path: Path
    ) -> None:
        now = int(time.time())
        verifier = _verifier(idp, tmp_path, max_token_lifetime_seconds=3600)
        with pytest.raises(InvalidTokenError):
            await verifier.verify(idp.token(iat=None, exp=now + 7200))
        await verifier.verify(idp.token(iat=None, exp=now + 1800))

    async def test_a_forged_future_iat_does_not_hide_a_far_exp(
        self, idp: FakeIdp, tmp_path: Path
    ) -> None:
        now = int(time.time())
        verifier = _verifier(idp, tmp_path, max_token_lifetime_seconds=3600)
        with pytest.raises(InvalidTokenError):
            await verifier.verify(idp.token(iat=now + 10**6, exp=now + 10**6 + 60))

    async def test_a_float_exp_within_bounds_is_fine(self, idp: FakeIdp, tmp_path: Path) -> None:
        await _verifier(idp, tmp_path).verify(idp.token(exp=time.time() + 300.5))
