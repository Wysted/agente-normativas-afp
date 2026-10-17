"""JWT test helpers: a throwaway IdP (keys, JWKS, tokens) and a scripted JWKS fetcher."""

import json
import time
from collections.abc import Mapping
from typing import Any

import jwt
from cryptography.hazmat.primitives.asymmetric import ec, rsa

ISSUER = "https://idp.example"
AUDIENCE = "agintis-api"
KID = "key-1"


class FakeIdp:
    """Signs tokens and publishes the matching public JWKS (RSA by default, EC on request)."""

    def __init__(self, kid: str = KID, *, algorithm: str = "RS256") -> None:
        self.kid = kid
        self.algorithm = algorithm
        self.private_key: Any = (
            rsa.generate_private_key(65537, 2048)
            if algorithm.startswith(("RS", "PS"))
            else ec.generate_private_key(ec.SECP256R1())
        )

    def jwk(self, *, with_alg: bool = True) -> dict[str, Any]:
        public = self.private_key.public_key()
        algorithms = jwt.algorithms
        raw = (
            algorithms.RSAAlgorithm.to_jwk(public)
            if isinstance(self.private_key, rsa.RSAPrivateKey)
            else algorithms.ECAlgorithm.to_jwk(public)
        )
        jwk: dict[str, Any] = json.loads(raw)
        jwk.update({"kid": self.kid, "use": "sig"})
        if with_alg:
            jwk["alg"] = self.algorithm
        return jwk

    def jwks(self) -> dict[str, Any]:
        return {"keys": [self.jwk()]}

    def claims(self, **overrides: Any) -> dict[str, Any]:
        now = int(time.time())
        claims: dict[str, Any] = {
            "iss": ISSUER,
            "aud": AUDIENCE,
            "sub": "user-1",
            "iat": now,
            "exp": now + 300,
            "tenant_id": "acme",
        }
        claims.update(overrides)
        return {key: value for key, value in claims.items() if value is not None}

    def token(
        self,
        *,
        algorithm: str | None = None,
        kid: str | None = None,
        headers: Mapping[str, Any] | None = None,
        extra: Mapping[str, Any] | None = None,
        **overrides: Any,
    ) -> str:
        """A token signed by this IdP; an override set to None drops the claim.

        `extra` carries claims whose names are not identifiers (`https://x/tenant`).
        """
        header = {"kid": kid or self.kid, **(headers or {})}
        return jwt.encode(
            {**self.claims(**overrides), **(extra or {})},
            self.private_key,
            algorithm=algorithm or self.algorithm,
            headers=header,
        )


class ScriptedFetcher:
    """A JWKS fetcher answering from a mutable document, or failing; counts calls."""

    def __init__(self, document: Mapping[str, Any] | None = None) -> None:
        self.document: Mapping[str, Any] | None = document
        self.error: Exception | None = None
        self.calls = 0

    async def __call__(self) -> Mapping[str, Any]:
        self.calls += 1
        if self.error is not None:
            raise self.error
        assert self.document is not None
        return self.document


class ManualClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds
