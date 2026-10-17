#!/usr/bin/env bash
# Local and CI quality gate: lint, format, types, import contracts, unit tests.
# Integration tests are not part of it (they need Postgres + TEI): uv run pytest -m integration
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

step() {
    printf '\n==> %s\n' "$1"
    shift
    "$@"
}

step "ruff check" uv run ruff check .
step "ruff format --check" uv run ruff format --check .
step "mypy" uv run mypy
step "import-linter" uv run lint-imports
step "pytest (unit)" uv run pytest -q

printf '\nAll checks passed.\n'
