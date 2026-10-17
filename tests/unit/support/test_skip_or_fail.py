import pytest
from _pytest.outcomes import Failed, Skipped

from tests.conftest import REQUIRE_SERVICES_ENV, skip_or_fail


def test_skips_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(REQUIRE_SERVICES_ENV, raising=False)
    with pytest.raises(Skipped):
        skip_or_fail("db down")


def test_fails_when_services_are_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(REQUIRE_SERVICES_ENV, "1")
    with pytest.raises(Failed, match="db down"):
        skip_or_fail("db down")
