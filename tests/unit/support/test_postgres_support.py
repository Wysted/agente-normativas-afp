"""The test-database helpers must never be able to drop a non-test database."""

from pathlib import Path

import pytest
from sqlalchemy.engine import make_url

from tests.support import postgres as support


@pytest.mark.parametrize(
    "name", ["agintis_test", "agintis_test_e2e", "agintis_test_hnsw", "x_test_a_b", "a1_test"]
)
def test_disposable_names_are_accepted(name: str) -> None:
    support.assert_disposable_name(name)


@pytest.mark.parametrize(
    "name",
    [
        "agintis",
        "postgres",
        "test",
        "agintis_testing",
        "agintis_test_",
        "prod_test;drop",
        "",
        "a b",
    ],
)
def test_other_names_are_refused(name: str) -> None:
    with pytest.raises(RuntimeError, match="Refusing to drop"):
        support.assert_disposable_name(name)


def test_drop_and_recreate_refuse_before_touching_the_server() -> None:
    url = make_url("postgresql+psycopg://u:p@127.0.0.1:1/agintis")  # unreachable: must not connect
    with pytest.raises(RuntimeError, match="Refusing to drop"):
        support.drop_database(url, "agintis")
    with pytest.raises(RuntimeError, match="Refusing to drop"):
        support.recreate_test_database(url, "agintis")


def test_admin_url_prefers_test_database_url(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dev = make_url("postgresql+psycopg://dev:p@devhost:5432/agintis")
    monkeypatch.setattr(support, "dev_database_url", lambda: dev)
    monkeypatch.chdir(tmp_path)  # no .env to read
    monkeypatch.delenv("TEST_DATABASE_URL", raising=False)
    assert support.admin_database_url() == dev
    monkeypatch.setenv("TEST_DATABASE_URL", "postgresql+psycopg://ci:p@cihost:5433/postgres")
    assert support.admin_database_url().host == "cihost"
