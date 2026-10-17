"""Load RagProfiles from YAML files (one profile per file) in a configured directory.

Pure data in, frozen `RagProfile` out: no strategy is resolved here (and never by import path).
Strategy keys and their parameters are checked afterwards by `validate_profiles` against the
whitelisted registries, so a profile naming an unknown chunker fails at boot with that key.
"""

import logging
from pathlib import Path
from typing import Any, Final

import yaml
from pydantic import ValidationError

from agintis.modules.knowledge.application import InvalidProfileError
from agintis.modules.knowledge.domain import RagProfile
from agintis.modules.knowledge.profiles.params import describe_errors
from agintis.modules.knowledge.profiles.schema import ProfileDocument
from agintis.platform.yaml_config import (
    MAX_YAML_FILE_BYTES,
    YAML_SUFFIXES,
    StrictYamlLoader,
    YamlConfigError,
    load_yaml_file,
    yaml_files,
)

logger = logging.getLogger(__name__)

# Kept under their old names: profile files share the platform's limits and strict YAML loader.
PROFILE_FILE_SUFFIXES = YAML_SUFFIXES
MAX_PROFILE_FILE_BYTES: Final = MAX_YAML_FILE_BYTES


def parse_profile(raw: Any, *, origin: str) -> RagProfile:
    """Validate an already-parsed YAML document (a mapping) and build the profile."""
    if not isinstance(raw, dict):
        raise InvalidProfileError(f"{origin}: a profile file must be a YAML mapping")
    try:
        return ProfileDocument.model_validate(raw).to_domain()
    except ValidationError as exc:
        raise InvalidProfileError(f"{origin}: {describe_errors(exc)}") from exc
    except ValueError as exc:  # domain invariants (RagProfile, MetadataField, CleanerSpec)
        raise InvalidProfileError(f"{origin}: {exc}") from exc
    except RecursionError as exc:
        raise InvalidProfileError(f"{origin}: profile is nested too deeply") from exc


def load_yaml_text(text: str, *, origin: str) -> Any:
    """Parse profile YAML with `StrictYamlLoader`; `InvalidProfileError` on any problem."""
    try:
        return yaml.load(text, Loader=StrictYamlLoader)
    except yaml.YAMLError as exc:
        raise InvalidProfileError(f"{origin}: cannot read profile file: {exc}") from exc
    except RecursionError as exc:
        raise InvalidProfileError(f"{origin}: profile is nested too deeply") from exc


def load_profile(path: Path) -> RagProfile:
    try:
        raw = load_yaml_file(path)
    except YamlConfigError as exc:
        raise InvalidProfileError(str(exc)) from exc
    return parse_profile(raw, origin=path.name)


def load_profiles_dir(directory: Path) -> list[RagProfile]:
    """Every `*.yaml` / `*.yml` file of `directory` (not recursive), in file name order."""
    if not directory.is_dir():
        raise InvalidProfileError(f"Profiles directory {str(directory)!r} does not exist")
    profiles: list[RagProfile] = []
    seen: dict[str, str] = {}
    for path in yaml_files(directory):
        profile = load_profile(path)
        if profile.name in seen:
            raise InvalidProfileError(
                f"{path.name}: profile {profile.name!r} is already defined in {seen[profile.name]}"
            )
        seen[profile.name] = path.name
        profiles.append(profile)
    return profiles
