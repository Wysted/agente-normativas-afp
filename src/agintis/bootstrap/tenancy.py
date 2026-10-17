"""Builds the `TenantResolver` selected by `AUTH_*` (header trust or JWT verification)."""

import logging
from urllib.parse import urlsplit

from agintis.bootstrap.settings import Settings
from agintis.platform.auth import (
    DEV_KID_PREFIX,
    AuthConfigError,
    FileJwks,
    HttpJwks,
    HttpxJwksFetcher,
    JwksSource,
    JwtVerifier,
)
from agintis.platform.http.jwt_tenant import JwtTenantResolver
from agintis.platform.http.tenant import HeaderTenantResolver, TenantResolver
from agintis.shared_kernel.ids import TenantId

logger = logging.getLogger(__name__)

LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def _jwks_source(settings: Settings) -> JwksSource:
    auth = settings.auth
    local = settings.explicit_local_env
    if auth.jwks_file is not None:
        source = FileJwks(auth.jwks_file)
        logger.warning(
            "AUTH_JWKS_FILE is a static key set: no rotation or revocation until the file "
            "is replaced and the app restarted. Prefer AUTH_JWKS_URL in production."
        )
        dev_kids = sorted(kid for kid in source.kids if kid.startswith(DEV_KID_PREFIX))
        if dev_kids and not local:
            raise AuthConfigError(
                f"AUTH_JWKS_FILE contains development keys ({', '.join(dev_kids)}): their private "
                f"keys sit on a developer machine. Allowed only with an explicit APP_ENV=local."
            )
        return source
    assert auth.jwks_url is not None  # guaranteed by AuthSettings validation
    parts = urlsplit(auth.jwks_url)
    if parts.scheme != "https" and parts.hostname not in LOCAL_HOSTS:
        if not (auth.jwks_allow_insecure or local):
            raise AuthConfigError(
                "AUTH_JWKS_URL must be https: signing keys fetched over plain http can be "
                "replaced by anyone on the path. Set AUTH_JWKS_ALLOW_INSECURE=true to accept "
                "it (or use an explicit APP_ENV=local)."
            )
        logger.warning(
            "AUTH_JWKS_URL is not https: signing keys fetched over plain http can be replaced "
            "by anyone on the path."
        )
    fetcher = HttpxJwksFetcher(auth.jwks_url, auth.jwks_timeout_seconds)
    return HttpJwks(
        fetcher,
        ttl_seconds=auth.jwks_cache_ttl_seconds,
        min_refresh_interval_seconds=auth.jwks_min_refresh_seconds,
        fetch_timeout_seconds=auth.jwks_timeout_seconds,
        stale_grace_seconds=auth.jwks_stale_grace_seconds,
        closer=fetcher.aclose,
    )


def build_tenant_resolver(settings: Settings) -> TenantResolver:
    """Sync and network-free (a JWKS file is read here, so a bad one fails the boot).

    Built only on the HTTP path (`Container.build(with_tenant_resolver=True)`): CLIs have no use
    for it and must not read JWKS files or log auth warnings.
    """
    auth = settings.auth
    http = settings.http
    if auth.tenant_resolver == "jwt":
        assert auth.issuer is not None
        assert auth.audience is not None
        if http.default_tenant:
            logger.warning("HTTP_DEFAULT_TENANT is ignored: AUTH_TENANT_RESOLVER is 'jwt'")
        verifier = JwtVerifier(
            _jwks_source(settings),
            issuer=auth.issuer,
            audience=auth.audience,
            algorithms=auth.algorithms,
            leeway_seconds=auth.leeway_seconds,
            max_token_lifetime_seconds=auth.max_token_lifetime_seconds,
        )
        return JwtTenantResolver(verifier, auth.tenant_claim)

    logger.warning(
        "AUTH_TENANT_RESOLVER=header: the tenant is read from the X-Tenant-Id request header "
        "and the client is trusted. Use only behind a trusted gateway that authenticates "
        "callers and sets the header; otherwise set AUTH_TENANT_RESOLVER=jwt."
    )
    # The default tenant is a local-development convenience (fail-open): it needs an
    # explicit APP_ENV=local and is announced at boot.
    default_tenant = http.default_tenant if settings.default_tenant_enabled else None
    if default_tenant:
        logger.warning(
            "HTTP_DEFAULT_TENANT=%r is active (APP_ENV=local): requests without X-Tenant-Id "
            "act as that tenant. Never use this outside local development.",
            default_tenant,
        )
    elif http.default_tenant:
        logger.info("HTTP_DEFAULT_TENANT ignored: it needs an explicit APP_ENV=local")
    return HeaderTenantResolver(TenantId(default_tenant) if default_tenant else None)
