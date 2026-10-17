from collections.abc import AsyncIterator
from http import HTTPStatus

import httpx
import pytest
from pydantic import ValidationError

from agintis.bootstrap.container import Container
from agintis.bootstrap.settings import Settings
from agintis.main import create_app
from tests.support.settings import make_settings

_settings = make_settings


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(_settings())
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c,
    ):
        yield c


async def test_health_returns_ok(client: httpx.AsyncClient) -> None:
    response = await client.get("/health")

    assert response.status_code == HTTPStatus.OK
    assert response.json() == {"status": "ok"}


async def test_lifespan_sets_container() -> None:
    app = create_app(_settings())
    async with app.router.lifespan_context(app):
        assert isinstance(app.state.container, Container)


def test_settings_without_database_url_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(ValidationError):
        Settings.isolated()


def test_create_app_without_database_url_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(ValidationError):
        create_app(Settings.isolated())


def test_settings_reject_non_psycopg_scheme() -> None:
    with pytest.raises(ValidationError):
        Settings.isolated(database_url="postgresql+asyncpg://u:p@h/db")


def test_libpq_url_drops_driver_suffix() -> None:
    assert (
        _settings().libpq_url.get_secret_value() == "postgresql://u:secret-pw@localhost:5432/test"
    )


def test_container_repr_does_not_leak_password() -> None:
    assert "secret-pw" not in repr(Container.build(_settings()))
