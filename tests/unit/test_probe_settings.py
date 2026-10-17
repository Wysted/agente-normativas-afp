"""KNOWLEDGE_PROBE_CONCURRENCY: validated against the pool and wired into the probe."""

import pytest
from pydantic import ValidationError

from agintis.bootstrap.container import Container
from tests.support.settings import make_settings


def test_the_default_leaves_a_connection_free_in_the_default_pool() -> None:
    settings = make_settings()

    assert settings.knowledge.probe_concurrency == 2
    assert settings.database.pool_size == 3  # 2 <= 3 - 1


def test_it_cannot_take_the_whole_pool() -> None:
    with pytest.raises(ValidationError, match="DATABASE_POOL_SIZE - 1"):
        make_settings(knowledge={"probe_concurrency": 3})  # pool 3 -> at most 2


def test_a_bigger_pool_allows_more() -> None:
    settings = make_settings(knowledge={"probe_concurrency": 5}, database={"pool_size": 6})

    assert settings.knowledge.probe_concurrency == 5


def test_a_one_connection_pool_still_allows_one_probe_at_a_time() -> None:
    settings = make_settings(knowledge={"probe_concurrency": 1}, database={"pool_size": 1})

    assert settings.knowledge.probe_concurrency == 1


def test_it_must_be_at_least_one() -> None:
    with pytest.raises(ValidationError):
        make_settings(knowledge={"probe_concurrency": 0})


def test_the_container_wires_the_setting_into_the_probe() -> None:
    container = Container.build(make_settings(knowledge={"probe_concurrency": 1}))

    assert container.probe_profiles._max_concurrency == 1
