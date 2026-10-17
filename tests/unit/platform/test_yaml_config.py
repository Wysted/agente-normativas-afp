from pathlib import Path

import pytest

from agintis.platform.yaml_config import (
    MAX_YAML_FILE_BYTES,
    YamlConfigError,
    load_yaml_file,
    yaml_files,
)


def _file(tmp_path: Path, text: str, name: str = "c.yaml") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_plain_data(tmp_path: Path) -> None:
    assert load_yaml_file(_file(tmp_path, "a: 1\nb: [x, y]\n")) == {"a": 1, "b": ["x", "y"]}


@pytest.mark.parametrize(
    "text", ["a: [", "a: 1\na: 2\n", "a: &x 1\nb: *x\n", "a: !!python/name:os.system"]
)
def test_rejects_what_configuration_never_needs(tmp_path: Path, text: str) -> None:
    with pytest.raises(YamlConfigError, match=r"c\.yaml"):
        load_yaml_file(_file(tmp_path, text))


def test_rejects_oversized_and_undecodable_files(tmp_path: Path) -> None:
    with pytest.raises(YamlConfigError, match="larger"):
        load_yaml_file(_file(tmp_path, "a: " + "x" * MAX_YAML_FILE_BYTES))
    bad = tmp_path / "bin.yaml"
    bad.write_bytes(b"\xff\xfe\x00")
    with pytest.raises(YamlConfigError):
        load_yaml_file(bad)


def test_yaml_files_are_sorted_and_symlinks_must_stay_inside(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    target = _file(outside, "a: 1", "secret.yaml")
    inside = tmp_path / "inside"
    inside.mkdir()
    _file(inside, "a: 1", "b.yaml")
    _file(inside, "a: 1", "a.yml")
    _file(inside, "a: 1", "skip.txt")
    (inside / "link.yaml").symlink_to(target)
    (inside / "ok.yaml").symlink_to(inside / "b.yaml")

    assert [p.name for p in yaml_files(inside)] == ["a.yml", "b.yaml", "ok.yaml"]
