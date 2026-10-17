"""Strict YAML loading for configuration files (agent specs; plain data in, plain data out).

`SafeLoader` that also rejects what configuration never needs and attackers like: anchors and
aliases (an alias bomb expands a few hundred bytes into gigabytes) and duplicate mapping keys
(PyYAML keeps the last one silently, changing behaviour without a trace).
"""

import logging
from pathlib import Path
from typing import Any, Final

import yaml

logger = logging.getLogger(__name__)

YAML_SUFFIXES: Final = (".yaml", ".yml")
# A config file is a few KiB; the cap keeps a stray or hostile file from being read whole.
MAX_YAML_FILE_BYTES: Final = 256 * 1024


class YamlConfigError(ValueError):
    """A configuration file cannot be read or parsed (the message names the file)."""


class StrictYamlLoader(yaml.SafeLoader):
    def compose_node(self, parent: yaml.Node | None, index: int) -> yaml.Node | None:
        event = self.peek_event()  # type: ignore[no-untyped-call]
        if getattr(event, "anchor", None) is not None:  # anchors and aliases both carry one
            raise yaml.composer.ComposerError(
                None, None, "YAML anchors and aliases are not allowed", event.start_mark
            )
        return super().compose_node(parent, index)

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        seen: set[object] = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=True)
            try:
                duplicate = key in seen
                seen.add(key)
            except TypeError:  # unhashable: the base class reports it
                continue
            if duplicate:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found a duplicate key {key!r}",
                    key_node.start_mark,
                )
        return super().construct_mapping(node, deep=deep)


def load_yaml_file(path: Path) -> Any:
    """Parse one file with `StrictYamlLoader`; `YamlConfigError` on any problem."""
    try:
        with path.open("rb") as handle:
            data = handle.read(MAX_YAML_FILE_BYTES + 1)
        if len(data) > MAX_YAML_FILE_BYTES:
            raise YamlConfigError(f"{path.name}: file is larger than {MAX_YAML_FILE_BYTES} bytes")
        return yaml.load(data.decode("utf-8"), Loader=StrictYamlLoader)
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise YamlConfigError(f"{path.name}: cannot read file: {exc}") from exc
    except RecursionError as exc:
        raise YamlConfigError(f"{path.name}: file is nested too deeply") from exc


def yaml_files(directory: Path) -> list[Path]:
    """Regular `*.yaml` / `*.yml` files of `directory` (not recursive), sorted by name.

    A symlink is followed only when its target stays inside the directory; one pointing
    elsewhere is skipped with a warning.
    """
    root = directory.resolve()
    files: list[Path] = []
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in YAML_SUFFIXES or not path.is_file():
            continue
        if not path.resolve().is_relative_to(root):
            logger.warning("Ignoring file %s: it resolves outside %s", path.name, root)
            continue
        files.append(path)
    return files
