"""docs/configuration.md rows must match the environment variables `Settings` accepts."""

import re
from pathlib import Path

from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings

from agintis.bootstrap.settings import Settings

DOC = Path(__file__).resolve().parents[2] / "docs" / "configuration.md"


def _is_group(field: FieldInfo) -> bool:
    annotation = field.annotation
    return isinstance(annotation, type) and issubclass(annotation, BaseSettings)


def _env_name(prefix: str, name: str, field: FieldInfo) -> str:
    alias = field.validation_alias
    return (alias if isinstance(alias, str) else f"{prefix}{name}").upper()


def settings_env_names() -> set[str]:
    """Flat env names: top-level fields as-is, nested groups with their own `env_prefix`."""
    names: set[str] = set()
    for name, field in Settings.model_fields.items():
        if _is_group(field):
            group = field.annotation
            assert isinstance(group, type)
            assert issubclass(group, BaseModel)
            prefix = str(group.model_config.get("env_prefix", ""))
            names.update(_env_name(prefix, n, f) for n, f in group.model_fields.items())
        else:
            names.add(_env_name("", name, field))
    return names


def documented_settings_names() -> set[str]:
    """Names in the table rows (`| `NAME` |`) of the sections before "Compose-only variables"."""
    text = DOC.read_text(encoding="utf-8").split("## Compose-only variables")[0]
    return set(re.findall(r"^\| `([A-Z][A-Z0-9_]*)` \|", text, flags=re.MULTILINE))


def test_documented_rows_match_settings_env_vars() -> None:
    names = settings_env_names()
    documented = documented_settings_names()
    assert len(names) > 50, "the walk found suspiciously few variables"

    assert not names - documented, f"missing rows: {', '.join(sorted(names - documented))}"
    assert not documented - names, f"stale rows: {', '.join(sorted(documented - names))}"
