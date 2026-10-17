"""Pieces shared by the measurement CLIs (`eval`, `calibrate_evidence`, the answer-judge CLIs)."""

import hashlib
import json
import os
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import IO

DEFAULT_TENANT = "default"
EXIT_OK = 0
EXIT_USAGE = 2
EXIT_ABORTED = 3
GIT_TIMEOUT_SECONDS = 5
PRIVATE_MODE = 0o600  # files that hold retrieved text and model output


class UsageError(ValueError):
    """Inconsistent arguments or input files (exit code 2)."""


def git_revision(cwd: Path | None = None) -> str | None:
    """`<short sha>` (`-dirty` with uncommitted changes), None outside a git checkout."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            timeout=GIT_TIMEOUT_SECONDS,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            timeout=GIT_TIMEOUT_SECONDS,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return f"{sha}-dirty" if dirty else sha


def check_output_path(path: Path | None, flag: str) -> None:
    """`UsageError` unless `path` (when given) can be written: not a directory, parent exists."""
    if path is None:
        return
    if path.is_dir():
        raise UsageError(f"{flag} {path} is a directory")
    if not path.parent.is_dir():
        raise UsageError(f"{flag}: the directory {path.parent} does not exist")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_lines(path: Path) -> list[tuple[str, str]]:
    """`(where, line)` of the non-blank lines of a UTF-8 file; `where` is `name:line number`."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise UsageError(f"{path}: cannot read ({type(exc).__name__}: {exc})") from exc
    return [(f"{path.name}:{n}", line) for n, line in enumerate(lines, start=1) if line.strip()]


def read_jsonl(path: Path) -> list[tuple[str, object]]:
    """`(where, parsed JSON)` of every non-blank line (`UsageError` on unreadable / invalid)."""
    rows: list[tuple[str, object]] = []
    for where, line in read_lines(path):
        try:
            rows.append((where, json.loads(line)))
        except ValueError as exc:
            raise UsageError(f"{where}: invalid JSON ({exc})") from exc
    return rows


class PartialWriter:
    """Appends lines to `PATH.partial` (0600, also over an existing looser file), one flushed
    line each, and renames it to PATH on `commit`. Without a commit the partial file stays: it is
    what an abort already paid for."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.partial = path.with_name(path.name + ".partial")
        self._file: IO[str] | None = None

    def _open(self) -> IO[str]:
        if self._file is None:
            fd = os.open(self.partial, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, PRIVATE_MODE)
            os.fchmod(fd, PRIVATE_MODE)
            self._file = os.fdopen(fd, "w", encoding="utf-8")
        return self._file

    def write_text(self, text: str) -> None:
        handle = self._open()
        handle.write(text)
        handle.flush()

    def write_line(self, line: str) -> None:
        self.write_text(line + "\n")

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def commit(self) -> None:
        self._open()  # nothing written: still leave an (empty) file behind
        self.close()
        os.replace(self.partial, self.path)


def write_private(path: Path, text: str) -> None:
    """The whole file at once through `PATH.partial` (0600), renamed to PATH when complete."""
    writer = PartialWriter(path)
    try:
        writer.write_text(text)
        writer.commit()
    finally:
        writer.close()


def write_private_lines(path: Path, lines: Iterable[str]) -> None:
    write_private(path, "".join(line + "\n" for line in lines))
