"""Arguments and output of the index sync CLI (the database part is covered by integration)."""

import pytest

from agintis.cli.indexes import (
    EXIT_ABORTED,
    EXIT_USAGE,
    build_parser,
    format_pending_sql,
    format_plan,
    main,
)
from agintis.modules.knowledge.api import (
    ApplyResult,
    IndexEntry,
    IndexPlan,
    IndexStatus,
    IndexSyncError,
)
from agintis.modules.knowledge.domain import MetadataField
from agintis.modules.knowledge.infrastructure.postgres import expected_metadata_indexes


def _plan() -> IndexPlan:
    (spec,) = expected_metadata_indexes([MetadataField("a", "str", filterable=True, indexed=True)])
    return IndexPlan(
        (
            IndexEntry(spec.name, IndexStatus.MISSING, spec),
            IndexEntry("ix_chunks_meta_old", IndexStatus.ORPHAN),
        )
    )


def test_defaults_are_a_dry_run_without_prune() -> None:
    args = build_parser().parse_args(["--profiles", "dir"])
    assert (args.apply, args.prune, args.confirm_prune) == (False, False, False)


def test_the_lock_timeout_is_configurable_and_validated(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert build_parser().parse_args(["--profiles", "d"]).lock_timeout == 30.0
    assert (
        build_parser().parse_args(["--profiles", "d", "--lock-timeout", "2.5"]).lock_timeout == 2.5
    )
    assert main(["--profiles", "d", "--lock-timeout", "0"]) == EXIT_USAGE
    assert "--lock-timeout" in capsys.readouterr().err


def test_every_status_has_an_action_and_the_columns_stay_aligned() -> None:
    entries = tuple(IndexEntry(f"ix_{s.value}", s) for s in IndexStatus)
    lines = format_plan(IndexPlan(entries)).splitlines()
    assert len(lines) == len(IndexStatus) + 1
    assert len({line.index(line.split()[2]) for line in lines[1:]}) == 1  # action column aligned
    assert "building" in format_plan(IndexPlan(entries))


def test_a_failed_run_prints_what_it_had_already_done(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def failing(_: object) -> object:
        raise IndexSyncError("boom", ApplyResult(("ix_a",), ("ix_old",), ("ix_a",), ("ix_b",)))

    monkeypatch.setattr("agintis.cli.indexes.run", failing)
    assert main(["--profiles", "dir"]) == EXIT_ABORTED
    captured = capsys.readouterr()
    assert "aborted: boom" in captured.err
    assert "created: ix_a" in captured.out
    assert "dropped: ix_old" in captured.out
    assert "WITHOUT fingerprint comment" in captured.out
    assert "skipped" in captured.out


def test_prune_needs_the_explicit_confirmation(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--profiles", "dir", "--prune"]) == EXIT_USAGE
    assert "--confirm-prune" in capsys.readouterr().err
    assert main(["--profiles", "dir", "--confirm-prune"]) == EXIT_USAGE


def test_the_plan_lists_every_index_with_its_action() -> None:
    lines = format_plan(_plan()).splitlines()
    assert lines[0].split() == ["index", "status", "action"]
    assert "ix_chunks_meta_a" in lines[1]
    assert "missing" in lines[1]
    assert "create" in lines[1]
    assert "orphan" in lines[2]
    assert "--confirm-prune" in lines[2]


def test_only_missing_indexes_have_pending_sql() -> None:
    (statement,) = format_pending_sql(_plan())
    assert statement.startswith("CREATE INDEX CONCURRENTLY IF NOT EXISTS")
    assert statement.endswith(";")


def test_an_empty_plan_says_so() -> None:
    assert "nothing to do" in format_plan(IndexPlan(()))
