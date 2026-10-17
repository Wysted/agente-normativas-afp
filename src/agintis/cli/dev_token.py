"""DEV ONLY: mint a signed JWT with a local keypair, to exercise `AUTH_TENANT_RESOLVER=jwt`.

    uv run python -m agintis.cli.dev_token --tenant acme [--sub dev-user] [--ttl 3600]

On first use it creates a keypair under `.dev/auth/` (gitignored; the private key is written with
mode 0600 and never leaves that directory) and (re)writes `.dev/auth/jwks.json`, the PUBLIC key set
(with `kid`). Point the app at it and the very same verification path as production runs:

    AUTH_TENANT_RESOLVER=jwt  AUTH_ISSUER=<issuer>  AUTH_AUDIENCE=<audience>
    AUTH_JWKS_FILE=.dev/auth/jwks.json

The token goes to stdout (nothing else), hints to stderr. Issuer, audience and the tenant claim
default to the AUTH_* settings (and to `agintis-dev` / `agintis` / `tenant_id` when unset). Never
use these keys in a real deployment: the private key sits on this machine's disk.

Exit codes: 0 ok; 2 usage error.
"""

import argparse
import json
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import jwt
from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from pydantic import ValidationError

from agintis.bootstrap.settings import AuthSettings
from agintis.cli._common import EXIT_OK, EXIT_USAGE, UsageError
from agintis.platform.auth import DEFAULT_MAX_TOKEN_LIFETIME_SECONDS, DEV_KID_PREFIX
from agintis.platform.http.tenant import TENANT_PATTERN

DevKey = rsa.RSAPrivateKey | ec.EllipticCurvePrivateKey

DEFAULT_DIR = Path(".dev/auth")
DEFAULT_ISSUER = "agintis-dev"
DEFAULT_AUDIENCE = "agintis"
DEFAULT_SUBJECT = "dev-user"
DEFAULT_TTL_SECONDS = 3600
MAX_TTL_SECONDS = DEFAULT_MAX_TOKEN_LIFETIME_SECONDS  # the verifier's default cap
JWKS_FILENAME = "jwks.json"
PRIVATE_FILE_MODE = 0o600
PRIVATE_DIR_MODE = 0o700
RSA_KEY_BITS = 2048
RSA_PUBLIC_EXPONENT = 65537
SUPPORTED_ALGORITHMS = ("RS256", "ES256")


def kid_for(algorithm: str) -> str:
    return f"{DEV_KID_PREFIX}{algorithm.lower()}"


def _generate(algorithm: str) -> DevKey:
    if algorithm == "RS256":
        return rsa.generate_private_key(RSA_PUBLIC_EXPONENT, RSA_KEY_BITS)
    return ec.generate_private_key(ec.SECP256R1())


def _pem_path(directory: Path, algorithm: str) -> Path:
    return directory / f"{algorithm.lower()}.private.pem"


def _ensure_directory(directory: Path) -> None:
    """Create the key directory (0700) when missing. An EXISTING directory is never chmod-ed:
    `--dir` may point anywhere and its permissions are not ours to change."""
    try:
        directory.parent.mkdir(parents=True, exist_ok=True)
        directory.mkdir(mode=PRIVATE_DIR_MODE)
    except FileExistsError:
        if not directory.is_dir():
            raise UsageError(f"{directory} exists and is not a directory") from None
        return  # created by someone else (or earlier): leave its mode alone
    directory.chmod(PRIVATE_DIR_MODE)  # mkdir's mode is subject to the umask


def load_or_create_key(directory: Path, algorithm: str) -> DevKey:
    """The dev private key of `algorithm`, created (mode 0600, exclusive) when missing."""
    _ensure_directory(directory)
    path = _pem_path(directory, algorithm)
    if not path.exists():
        key = _generate(algorithm)
        pem = key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, PRIVATE_FILE_MODE)
        except FileExistsError:
            pass  # lost a race with another process: use its key
        else:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(pem)
    path.chmod(PRIVATE_FILE_MODE)
    return _load(path)


def _load(path: Path) -> DevKey:
    try:
        loaded = serialization.load_pem_private_key(path.read_bytes(), password=None)
    except (ValueError, TypeError, UnsupportedAlgorithm) as exc:
        raise UsageError(
            f"{path} is empty, corrupt or encrypted (not an unencrypted PEM private key); "
            f"delete it to generate a new one"
        ) from exc
    except OSError as exc:
        raise UsageError(f"cannot read {path}: {exc.strerror}") from exc
    if not isinstance(loaded, rsa.RSAPrivateKey | ec.EllipticCurvePrivateKey):
        raise UsageError(f"{path} is not an RSA or EC private key")
    return loaded


def _public_jwk(algorithm: str, private_key: DevKey) -> dict[str, Any]:
    public = private_key.public_key()
    if isinstance(public, rsa.RSAPublicKey):
        jwk: dict[str, Any] = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(public))
    elif isinstance(public, ec.EllipticCurvePublicKey):
        jwk = json.loads(jwt.algorithms.ECAlgorithm.to_jwk(public))
    else:  # pragma: no cover - only RSA and EC keys are generated here
        raise UsageError("unsupported key type")
    jwk.update({"kid": kid_for(algorithm), "use": "sig", "alg": algorithm})
    return jwk


def write_jwks(directory: Path) -> Path:
    """The public key set of every dev key in `directory` (the app's AUTH_JWKS_FILE)."""
    keys = [
        _public_jwk(alg, _load(path))
        for alg in SUPPORTED_ALGORITHMS
        if (path := _pem_path(directory, alg)).exists()
    ]
    target = directory / JWKS_FILENAME
    target.write_text(json.dumps({"keys": keys}, indent=2), encoding="utf-8")
    return target


def _nested(path: str, value: str) -> dict[str, Any]:
    """`org.tenant` -> {"org": {"tenant": value}}."""
    claims: dict[str, Any] = {}
    node = claims
    *parents, leaf = path.split(".")
    for part in parents:
        node = node.setdefault(part, {})
    node[leaf] = value
    return claims


def mint(
    *,
    directory: Path,
    algorithm: str,
    tenant: str,
    subject: str,
    ttl_seconds: int,
    issuer: str,
    audience: str,
    tenant_claim: str,
) -> tuple[str, Path]:
    """A signed token and the JWKS file that verifies it."""
    key = load_or_create_key(directory, algorithm)
    jwks_path = write_jwks(directory)
    now = int(time.time())
    claims: dict[str, Any] = {
        "iss": issuer,
        "aud": audience,
        "sub": subject,
        "iat": now,
        "nbf": now,
        "exp": now + ttl_seconds,
    }
    claims.update(_nested(tenant_claim, tenant))  # a dotted claim path nests the value
    token = jwt.encode(claims, key, algorithm=algorithm, headers={"kid": kid_for(algorithm)})
    return token, jwks_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m agintis.cli.dev_token",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--tenant", required=True, help="tenant id placed in the tenant claim")
    parser.add_argument("--sub", default=DEFAULT_SUBJECT, help="subject (default: %(default)s)")
    parser.add_argument(
        "--ttl", type=int, default=DEFAULT_TTL_SECONDS, help="validity in seconds (default: 3600)"
    )
    parser.add_argument("--alg", choices=SUPPORTED_ALGORITHMS, default="RS256")
    parser.add_argument("--issuer", help="default: AUTH_ISSUER or 'agintis-dev'")
    parser.add_argument("--audience", help="default: AUTH_AUDIENCE or 'agintis'")
    parser.add_argument("--tenant-claim", help="default: AUTH_TENANT_CLAIM (dotted path allowed)")
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR, help="key directory")
    return parser


def run(argv: Sequence[str] | None) -> None:
    args = build_parser().parse_args(argv)
    if not TENANT_PATTERN.fullmatch(args.tenant):
        raise UsageError(f"--tenant {args.tenant!r} is not a valid tenant id")
    if not 0 < args.ttl <= MAX_TTL_SECONDS:
        raise UsageError(f"--ttl must be between 1 and {MAX_TTL_SECONDS} seconds")
    try:
        auth = AuthSettings()
    except ValidationError as exc:
        raise UsageError(f"invalid AUTH_* settings: {exc}") from exc
    issuer = args.issuer or auth.issuer or DEFAULT_ISSUER
    audience = args.audience or auth.audience or DEFAULT_AUDIENCE
    token, jwks_path = mint(
        directory=args.dir,
        algorithm=args.alg,
        tenant=args.tenant,
        subject=args.sub,
        ttl_seconds=args.ttl,
        issuer=issuer,
        audience=audience,
        tenant_claim=args.tenant_claim or auth.tenant_claim,
    )
    print(
        f"dev only. JWKS file: {jwks_path}\n"
        f"  AUTH_TENANT_RESOLVER=jwt AUTH_ISSUER={issuer} AUTH_AUDIENCE={audience} "
        f"AUTH_JWKS_FILE={jwks_path}",
        file=sys.stderr,
    )
    print(token)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        run(argv)
    except UsageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
