import json
import stat
from pathlib import Path

import jwt
import pytest

from agintis.cli import dev_token
from agintis.cli._common import EXIT_USAGE
from agintis.platform.auth import FileJwks, JwtVerifier


@pytest.fixture(autouse=True)
def _isolated_auth_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in ("ISSUER", "AUDIENCE", "TENANT_CLAIM", "TENANT_RESOLVER", "JWKS_FILE", "JWKS_URL"):
        monkeypatch.delenv(f"AUTH_{name}", raising=False)
    monkeypatch.chdir(tmp_path)  # AuthSettings reads `.env` from the cwd: none here


def _run(capsys: pytest.CaptureFixture[str], *args: str) -> str:
    assert dev_token.main(list(args)) == 0
    return capsys.readouterr().out.strip()


@pytest.mark.parametrize("alg", ["RS256", "ES256"])
async def test_the_minted_token_verifies_against_the_generated_jwks(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], alg: str
) -> None:
    keys = tmp_path / "keys"
    token = _run(capsys, "--tenant", "acme", "--sub", "me", "--alg", alg, "--dir", str(keys))
    verifier = JwtVerifier(
        FileJwks(keys / "jwks.json"),
        issuer=dev_token.DEFAULT_ISSUER,
        audience=dev_token.DEFAULT_AUDIENCE,
        algorithms=[alg],
    )
    verified = await verifier.verify(token)
    assert verified.claims["tenant_id"] == "acme"
    assert verified.subject == "me"
    assert jwt.get_unverified_header(token)["kid"] == dev_token.kid_for(alg)


def test_private_keys_are_private_and_the_jwks_is_public_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    keys = tmp_path / "keys"
    _run(capsys, "--tenant", "acme", "--dir", str(keys))
    pem = keys / "rs256.private.pem"
    assert stat.S_IMODE(pem.stat().st_mode) == 0o600
    assert stat.S_IMODE(keys.stat().st_mode) == 0o700
    document = json.loads((keys / "jwks.json").read_text())
    assert all(not {"d", "p", "q"} & set(key) for key in document["keys"])
    assert all(key["kid"] and key["use"] == "sig" for key in document["keys"])


def test_the_key_is_created_once_and_reused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    keys = tmp_path / "keys"
    _run(capsys, "--tenant", "acme", "--dir", str(keys))
    first = (keys / "rs256.private.pem").read_bytes()
    _run(capsys, "--tenant", "other", "--dir", str(keys))
    assert (keys / "rs256.private.pem").read_bytes() == first


def test_both_algorithms_share_one_jwks(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    keys = tmp_path / "keys"
    _run(capsys, "--tenant", "acme", "--alg", "RS256", "--dir", str(keys))
    _run(capsys, "--tenant", "acme", "--alg", "ES256", "--dir", str(keys))
    kids = {key["kid"] for key in json.loads((keys / "jwks.json").read_text())["keys"]}
    assert kids == {"dev-rs256", "dev-es256"}


async def test_issuer_audience_and_a_nested_claim_come_from_options_and_settings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AUTH_ISSUER", "https://idp.test")
    monkeypatch.setenv("AUTH_AUDIENCE", "api-x")
    monkeypatch.setenv("AUTH_TENANT_CLAIM", "org.tenant")
    keys = tmp_path / "keys"
    token = _run(capsys, "--tenant", "nested", "--dir", str(keys))
    verifier = JwtVerifier(
        FileJwks(keys / "jwks.json"), issuer="https://idp.test", audience="api-x"
    )
    claims = (await verifier.verify(token)).claims
    assert claims["org"] == {"tenant": "nested"}


def test_ttl_sets_the_expiry(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    token = _run(capsys, "--tenant", "acme", "--ttl", "120", "--dir", str(tmp_path / "k"))
    claims = jwt.decode(token, options={"verify_signature": False})
    assert claims["exp"] - claims["iat"] == 120


@pytest.mark.parametrize(
    "args",
    [["--tenant", "a b"], ["--tenant", "acme", "--ttl", "0"], ["--tenant", "acme", "--ttl", "-5"]],
)
def test_usage_errors(args: list[str], capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    assert dev_token.main([*args, "--dir", str(tmp_path / "k")]) == EXIT_USAGE
    assert not (tmp_path / "k").exists()
    assert capsys.readouterr().out == ""


def test_stdout_is_only_the_token_and_stderr_never_has_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert dev_token.main(["--tenant", "acme", "--dir", str(tmp_path / "k")]) == 0
    captured = capsys.readouterr()
    token = captured.out.strip()
    assert token.count(".") == 2
    assert token not in captured.err


def test_dev_keys_are_gitignored() -> None:
    root = Path(__file__).resolve().parents[3]
    assert ".dev/" in (root / ".gitignore").read_text().splitlines()


def test_an_existing_directory_keeps_its_mode(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o755)
    _run(capsys, "--tenant", "acme", "--dir", str(shared))
    assert stat.S_IMODE(shared.stat().st_mode) == 0o755
    assert stat.S_IMODE((shared / "rs256.private.pem").stat().st_mode) == 0o600


def test_a_created_directory_is_private_even_with_nested_parents(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    keys = tmp_path / "a" / "b" / "keys"
    _run(capsys, "--tenant", "acme", "--dir", str(keys))
    assert stat.S_IMODE(keys.stat().st_mode) == 0o700


@pytest.mark.parametrize(
    "content", [b"", b"garbage", b"-----BEGIN PRIVATE KEY-----\nAAAA\n-----END PRIVATE KEY-----\n"]
)
def test_an_empty_or_corrupt_pem_is_a_clean_usage_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], content: bytes
) -> None:
    keys = tmp_path / "keys"
    keys.mkdir()
    (keys / "rs256.private.pem").write_bytes(content)
    assert dev_token.main(["--tenant", "acme", "--dir", str(keys)]) == EXIT_USAGE
    captured = capsys.readouterr()
    assert "delete it" in captured.err
    assert captured.out == ""


def test_a_file_where_the_directory_should_be_is_a_usage_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "keys").write_text("x")
    assert dev_token.main(["--tenant", "acme", "--dir", str(tmp_path / "keys")]) == EXIT_USAGE


def test_losing_the_creation_race_uses_the_other_process_key(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    keys = tmp_path / "keys"
    _run(capsys, "--tenant", "acme", "--dir", str(keys))
    winner = (keys / "rs256.private.pem").read_bytes()
    real_exists = Path.exists
    # The CLI sees "missing" for the pem, then O_EXCL finds it: the existing key must win.
    monkeypatch.setattr(
        Path, "exists", lambda self: False if self.name.endswith(".pem") else real_exists(self)
    )
    _run(capsys, "--tenant", "acme", "--dir", str(keys))
    assert (keys / "rs256.private.pem").read_bytes() == winner


def test_the_dev_kid_prefix_is_the_one_the_boot_refuses() -> None:
    from agintis.platform.auth import DEV_KID_PREFIX

    assert dev_token.kid_for("RS256").startswith(DEV_KID_PREFIX)


def test_dev_keys_are_dockerignored() -> None:
    root = Path(__file__).resolve().parents[3]
    assert ".dev" in (root / ".dockerignore").read_text().splitlines()
