"""Tenant from a verified JWT bearer token (`Authorization: Bearer <jwt>`).

Behaviours worth knowing:
- FastAPI parses the JSON body BEFORE running dependencies, so a malformed body without a token
  answers 422, not 401 (a well-formed body reaches this resolver first).
- `GET /v1/agents` (a catalog, not tenant-scoped) and `/health` do not resolve a tenant: public.
- There is no `typ` check; `AUTH_AUDIENCE` must not be the client_id (see `JwtVerifier`).
"""

import logging
from collections.abc import Mapping
from typing import Any

from fastapi import Request

from agintis.platform.auth import (
    InvalidTokenError,
    JwtVerifier,
    MissingCredentialsError,
    TenantClaimError,
)
from agintis.platform.http.tenant import TENANT_PATTERN
from agintis.shared_kernel.ids import TenantId

logger = logging.getLogger(__name__)

AUTHORIZATION_HEADER = "Authorization"
BEARER_SCHEME = "bearer"
MAX_TOKEN_CHARS = 8192
# `request.state` attribute with the verified `sub` (None when absent): for future use.
SUBJECT_STATE_KEY = "auth_subject"


def claim_at(claims: Mapping[str, Any], path: str) -> Any:
    """Value at `path`: the exact top-level key first (namespaced claims such as
    `https://x/tenant`), then the dotted path through nested objects. None when absent.

    Order matters: a token whose top-level key equals the configured path wins over the nested
    object of the same dotted name; the exact key is what an IdP namespaces, so it is preferred."""
    if path in claims:
        return claims[path]
    node: Any = claims
    for part in path.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return None
        node = node[part]
    return node


def bearer_token(request: Request) -> str:
    header = request.headers.get(AUTHORIZATION_HEADER, "").strip()
    scheme, _, token = header.partition(" ")
    if scheme.lower() != BEARER_SCHEME:
        raise MissingCredentialsError
    token = token.strip()
    if not token or len(token) > MAX_TOKEN_CHARS or " " in token:
        raise InvalidTokenError("malformed credentials")
    return token


class JwtTenantResolver:
    """Verifies the bearer token and reads the tenant from `tenant_claim`.

    Never logs tokens or claim values: rejections log a fixed reason (plus the configured claim
    NAME for tenant claim rejections), successes nothing.
    """

    def __init__(self, verifier: JwtVerifier, tenant_claim: str) -> None:
        self._verifier = verifier
        self._tenant_claim = tenant_claim

    async def resolve(self, request: Request) -> TenantId:
        try:
            token = bearer_token(request)
        except MissingCredentialsError:
            logger.info("Request rejected: no bearer credentials")
            raise
        except InvalidTokenError as exc:
            logger.info("Bearer token rejected: %s", exc.reason)
            raise
        try:
            verified = await self._verifier.verify(token)
        except InvalidTokenError as exc:
            logger.info("Bearer token rejected: %s", exc.reason)
            raise
        raw = claim_at(verified.claims, self._tenant_claim)
        if raw is None:
            logger.info("Tenant claim rejected: claim %r is missing", self._tenant_claim)
            raise TenantClaimError("The token has no tenant claim", code="tenant_claim_missing")
        if not isinstance(raw, str):
            logger.info("Tenant claim rejected: claim %r is not a string", self._tenant_claim)
            raise TenantClaimError("The token's tenant claim is not a valid tenant id")
        if not TENANT_PATTERN.fullmatch(raw):
            logger.info(
                "Tenant claim rejected: claim %r is not a valid tenant id", self._tenant_claim
            )
            raise TenantClaimError("The token's tenant claim is not a valid tenant id")
        setattr(request.state, SUBJECT_STATE_KEY, verified.subject)
        return TenantId(raw)

    async def aclose(self) -> None:
        await self._verifier.aclose()
