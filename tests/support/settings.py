"""Settings builder for tests: never reads `.env` (nested groups included)."""

from typing import Any

from agintis.bootstrap.settings import Settings

TEST_DATABASE_URL = "postgresql+psycopg://u:secret-pw@localhost:5432/test"


def make_settings(**overrides: Any) -> Settings:
    """Isolated settings with a test DB URL, boot checks off (no tokenizer/TEI access) and the
    checkpointer disabled (no pool)."""
    overrides.setdefault("database_url", TEST_DATABASE_URL)
    overrides.setdefault("startup_checks", False)
    # Threads are off by default in tests: no pool is opened against the fake URL above.
    overrides.setdefault("checkpointer", {"enabled": False})
    return Settings.isolated(**overrides)
