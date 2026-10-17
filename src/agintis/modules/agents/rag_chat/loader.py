"""Load agent specs from a directory of YAML files (one spec per file)."""

from pathlib import Path

from agintis.modules.agents.rag_chat.spec import (
    AgentSpec,
    InvalidAgentSpecError,
    parse_agent_spec,
)
from agintis.platform.yaml_config import YamlConfigError, load_yaml_file, yaml_files


def load_agent_specs(directory: Path) -> list[AgentSpec]:
    """Every `*.yaml` / `*.yml` file of `directory` (not recursive), in file name order.

    An invalid file, or two files declaring the same agent id, raises `InvalidAgentSpecError`
    naming the file. An empty directory yields no specs.
    """
    if not directory.is_dir():
        raise InvalidAgentSpecError(f"Agents directory {str(directory)!r} does not exist")
    specs: list[AgentSpec] = []
    seen: dict[str, str] = {}
    for path in yaml_files(directory):
        try:
            raw = load_yaml_file(path)
        except YamlConfigError as exc:
            raise InvalidAgentSpecError(str(exc)) from exc
        spec = parse_agent_spec(raw, origin=path.name)
        if spec.id in seen:
            raise InvalidAgentSpecError(
                f"{path.name}: agent {spec.id!r} is already defined in {seen[spec.id]}"
            )
        seen[spec.id] = path.name
        specs.append(spec)
    return specs
