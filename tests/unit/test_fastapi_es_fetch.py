"""Pure logic of the fastapi_es acquisition script (loaded by path: `case/` is not a package)."""

import importlib.util
from pathlib import Path

from tests.support.postgres import PROJECT_ROOT

_SPEC = importlib.util.spec_from_file_location(
    "fastapi_es_fetch", PROJECT_ROOT / "case" / "fastapi_es" / "fetch" / "fetch.py"
)
assert _SPEC is not None
assert _SPEC.loader is not None
fetch = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fetch)

CODE = "a\nb\nc\nd"


def test_select_lines_ranges_and_gap_marker() -> None:
    assert fetch.select_lines(CODE, "1:3") == "a\nb\nc"
    assert fetch.select_lines(CODE, "2") == "b"
    assert fetch.select_lines(CODE, "1,3:4") == "a\n# ...\nc\nd"


def test_select_lines_invalid_is_none() -> None:
    for ln in ("0", "5:2", "1:9", "x", "1,"):
        assert fetch.select_lines(CODE, ln) is None


def test_include_path_guard_counts_escapes_as_missing(tmp_path: Path) -> None:
    (tmp_path / "docs_src").mkdir()
    (tmp_path / "docs_src" / "ok.py").write_text("x = 1\n")
    (tmp_path / "secret.py").write_text("s = 1\n")
    text = "\n".join(
        [
            "{* ../../docs_src/ok.py *}",
            "{* ../../docs_src/../secret.py *}",
            "{* ../../docs_src//etc/passwd *}",
            "  {* indented *}",
        ]
    )
    out, inlined, missing, unmatched, _ = fetch.inline_includes(text, tmp_path)
    assert (inlined, missing, unmatched) == (1, 2, 1)
    assert "x = 1" in out
    assert "secret" not in out.replace("../secret.py", "")
