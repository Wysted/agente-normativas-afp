import json
import stat
from pathlib import Path
from typing import Any

import pytest

from agintis.cli import agreement, perturb
from agintis.cli._common import EXIT_OK, EXIT_USAGE
from agintis.cli.perturb import EXIT_SKIPPED
from tests.unit.agent_kit.judge.support import make_record

ANSWER = "El plazo es de 24 meses [1]. La tasa es del 10% [2]."


def _jsonl(path: Path, *rows: Any) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _spec(**patch: Any) -> dict[str, Any]:
    return {
        "record_id": "q1",
        "split": "dev",
        "type": "number_date",
        "original": "24 meses",
        "replacement": "12 meses",
        "cited": [1],
        "why": "w",
        **patch,
    }


def _answers(tmp_path: Path) -> Path:
    return _jsonl(
        tmp_path / "answers.jsonl",
        make_record(ANSWER, id="q1").to_json(),
        make_record(ANSWER, id="q2").to_json(),
    )


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _run(tmp_path: Path, specs: list[dict[str, Any]], *extra: str) -> int:
    return perturb.main(
        ["--answers", str(_answers(tmp_path)),
         "--specs", str(_jsonl(tmp_path / "specs.jsonl", *specs)),
         "--out", str(tmp_path / "out.jsonl"),
         "--labels-out", str(tmp_path / "labels.jsonl"), *extra]
    )  # fmt: skip


def test_end_to_end_writes_originals_perturbed_and_labels(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = _run(
        tmp_path,
        [
            _spec(),
            _spec(
                record_id="q2",
                type="quantifier",
                original="10%",
                replacement="20%",
                split="test",
                cited=[2],
            ),
        ],
    )

    assert code == EXIT_OK
    out = _read(tmp_path / "out.jsonl")
    assert [r["id"] for r in out] == ["q1", "q2", "q1~p1", "q2~p1"]
    assert out[0]["answer"] == ANSWER
    assert out[2]["answer"].count("12 meses") == 1
    assert out[2]["meta"]["type"] == "number_date"
    labels = _read(tmp_path / "labels.jsonl")
    assert {(r["id"], r["judge"], r["label"]) for r in labels} == {
        ("q1", "faithfulness", True),
        ("q1~p1", "faithfulness", False),
        ("q1~p1", "citation_support", False),
        ("q2", "faithfulness", True),
        ("q2~p1", "faithfulness", False),
        ("q2~p1", "citation_support", False),
    }
    assert all(r["labeler"] == "perturbation" and r["answer_sha256"] for r in labels)
    for name in ("out.jsonl", "labels.jsonl"):
        assert stat.S_IMODE((tmp_path / name).stat().st_mode) == 0o600
        assert not (tmp_path / f"{name}.partial").exists()
    err = capsys.readouterr().err
    assert "dev: 1" in err
    assert "test: 1" in err
    assert "number_date" in err


def test_split_filter_and_no_originals(tmp_path: Path) -> None:
    specs = [
        _spec(),
        _spec(record_id="q2", split="test", original="10%", replacement="2%", cited=[2]),
    ]
    code = _run(tmp_path, specs, "--split", "test", "--no-include-originals")

    assert code == EXIT_OK
    assert [r["id"] for r in _read(tmp_path / "out.jsonl")] == ["q2~p1"]
    assert {r["id"] for r in _read(tmp_path / "labels.jsonl")} == {"q2~p1"}


def test_a_record_in_two_splits_is_a_usage_error_even_with_a_split_filter(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    specs = [_spec(), _spec(split="test", original="10%", replacement="2%", cited=[2])]
    assert _run(tmp_path, specs, "--split", "test") == EXIT_USAGE
    assert "splits" in capsys.readouterr().err
    assert not (tmp_path / "out.jsonl").exists()


def test_skipped_specs_exit_1_but_the_valid_ones_are_written(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    specs = [_spec(), _spec(original="nope"), _spec(record_id="zz"), _spec(type="x")]
    code = _run(tmp_path, specs)

    assert code == EXIT_SKIPPED
    assert [r["id"] for r in _read(tmp_path / "out.jsonl")] == ["q1", "q1~p1"]
    err = capsys.readouterr().err
    for reason in ("not found", "unknown record_id", "unknown type"):
        assert reason in err


def test_a_malformed_spec_line_is_skipped_not_fatal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    specs = tmp_path / "specs.jsonl"
    specs.write_text(json.dumps(_spec()) + "\nnot json\n", encoding="utf-8")

    code = perturb.main(
        ["--answers", str(_answers(tmp_path)), "--specs", str(specs),
         "--out", str(tmp_path / "o.jsonl"), "--labels-out", str(tmp_path / "l.jsonl")]
    )  # fmt: skip

    assert code == EXIT_SKIPPED
    assert [r["id"] for r in _read(tmp_path / "o.jsonl")] == ["q1", "q1~p1"]
    err = capsys.readouterr().err
    assert "specs.jsonl:2: malformed spec" in err


def test_an_empty_spec_file_is_a_usage_error(tmp_path: Path) -> None:
    assert _run(tmp_path, []) == EXIT_USAGE


def test_usage_errors_exit_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    answers = _answers(tmp_path)
    specs = _jsonl(tmp_path / "s.jsonl", _spec())
    base = ["--answers", str(answers), "--specs", str(specs)]

    code = perturb.main([*base, "--out", str(tmp_path), "--labels-out", str(tmp_path / "l")])
    assert code == EXIT_USAGE
    assert "is a directory" in capsys.readouterr().err
    code = perturb.main(
        ["--answers", str(tmp_path / "nope"), "--specs", str(specs),
         "--out", str(tmp_path / "o"), "--labels-out", str(tmp_path / "l")]
    )  # fmt: skip
    assert code == EXIT_USAGE
    code = perturb.main([*base, "--out", str(tmp_path / "x"), "--labels-out", str(tmp_path / "x")])
    assert code == EXIT_USAGE
    code = perturb.main([*base, "--split", "test", "--out", str(tmp_path / "o"),
                         "--labels-out", str(tmp_path / "l")])  # fmt: skip
    assert code == EXIT_USAGE  # no spec of that split
    assert "no specs for split" in capsys.readouterr().err


def test_the_output_chains_into_cli_agreement_by_type(tmp_path: Path) -> None:
    _run(
        tmp_path,
        [_spec(), _spec(type="quantifier", original="10%", replacement="20%", cited=[2])],
    )
    # a judge that passes everything: accepts the original, misses both perturbations
    records = [
        {
            "id": r["id"],
            "answer_sha256": r["answer_sha256"],
            "judgments": {
                "faithfulness": {"status": "ok", "verdict": r["id"] != "q1~p2"},
                "citation_support": {"status": "ok", "verdict": False},
            },
        }
        for r in _read(tmp_path / "out.jsonl")
    ]
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"meta": {}, "records": records}), encoding="utf-8")
    out = tmp_path / "agreement.json"

    code = agreement.main(
        ["--labels", str(tmp_path / "labels.jsonl"), "--judgments", str(report),
         "--by-meta", "type", "--answers", str(tmp_path / "out.jsonl"), "--out", str(out)]
    )  # fmt: skip

    assert code == EXIT_OK
    entry = json.loads(out.read_text(encoding="utf-8"))["judges"]["faithfulness"]
    assert entry["judge_vs_consensus"]["sensitivity"] == 1.0
    assert entry["judge_vs_consensus"]["specificity"] == 0.5
    groups = entry["by_group"]
    assert groups["(none)"]["sensitivity"] == 1.0  # the original
    assert groups["number_date"]["specificity"] == 0.0  # missed
    assert groups["quantifier"]["specificity"] == 1.0  # caught


def test_a_malformed_row_takes_its_numbering_slot(tmp_path: Path) -> None:
    specs = tmp_path / "specs.jsonl"
    bad = {"record_id": "q1", "split": "dev"}  # no 'type', no 'original'
    specs.write_text(json.dumps(bad) + "\n" + json.dumps(_spec()) + "\n", encoding="utf-8")

    code = perturb.main(
        ["--answers", str(_answers(tmp_path)), "--specs", str(specs),
         "--out", str(tmp_path / "o.jsonl"), "--labels-out", str(tmp_path / "l.jsonl")]
    )  # fmt: skip

    assert code == EXIT_SKIPPED
    assert [r["id"] for r in _read(tmp_path / "o.jsonl")] == ["q1", "q1~p2"]


def _truth(tmp_path: Path, *rows: dict[str, Any]) -> Path:
    return _jsonl(tmp_path / "truth.jsonl", *rows)


def test_citation_truth_labels_only_the_originals_it_confirms_with_a_matching_hash(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sha = make_record(ANSWER, id="q1").answer_sha256
    truth = _truth(
        tmp_path,
        {"id": "q1", "judge": "citation_support", "label": True, "answer_sha256": sha},
        {"id": "q2", "judge": "citation_support", "label": True},  # no hash: confirms nothing
        {"id": "q1", "judge": "faithfulness", "label": False, "answer_sha256": "other"},
    )
    specs = [_spec(), _spec(record_id="q2")]

    code = _run(tmp_path, specs, "--citation-truth", str(truth))

    assert code == EXIT_OK
    labels = {(r["id"], r["judge"]): r["label"] for r in _read(tmp_path / "labels.jsonl")}
    assert labels[("q1", "citation_support")] is True
    assert ("q2", "citation_support") not in labels
    assert "citation_support true for 1 originals" in capsys.readouterr().err


def test_citation_truth_ignores_a_stale_or_contradicted_row(tmp_path: Path) -> None:
    sha = make_record(ANSWER, id="q1").answer_sha256
    truth = _truth(
        tmp_path,
        {"id": "q1", "judge": "citation_support", "label": True, "answer_sha256": sha},
        {"id": "q1", "judge": "citation_support", "label": False, "answer_sha256": sha},
        {"id": "q2", "judge": "citation_support", "label": True, "answer_sha256": "stale"},
    )

    _run(tmp_path, [_spec(), _spec(record_id="q2")], "--citation-truth", str(truth))

    judges = {(r["id"], r["judge"]) for r in _read(tmp_path / "labels.jsonl")}
    assert ("q1", "citation_support") not in judges
    assert ("q2", "citation_support") not in judges


def test_an_unreadable_citation_truth_file_is_a_usage_error(tmp_path: Path) -> None:
    code = _run(tmp_path, [_spec()], "--citation-truth", str(tmp_path / "nope.jsonl"))
    assert code == EXIT_USAGE


def test_agreement_refuses_an_answers_dump_the_report_was_not_made_on(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _run(tmp_path, [_spec()])
    out = _read(tmp_path / "out.jsonl")
    records = [
        {
            "id": r["id"],
            "answer_sha256": r["answer_sha256"],
            "judgments": {"faithfulness": {"status": "ok", "verdict": True}},
        }
        for r in out
    ]
    records[1]["answer_sha256"] = "0" * 64  # the report judged another answer
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"meta": {}, "records": records}), encoding="utf-8")

    code = agreement.main(
        ["--labels", str(tmp_path / "labels.jsonl"), "--judgments", str(report),
         "--by-meta", "type", "--answers", str(tmp_path / "out.jsonl")]
    )  # fmt: skip

    assert code == EXIT_USAGE
    assert "answer_sha256 differs" in capsys.readouterr().err

    records.pop()  # a record of the dump the report does not have
    report.write_text(json.dumps({"meta": {}, "records": records}), encoding="utf-8")
    code = agreement.main(
        ["--labels", str(tmp_path / "labels.jsonl"), "--judgments", str(report),
         "--by-meta", "type", "--answers", str(tmp_path / "out.jsonl")]
    )  # fmt: skip
    assert code == EXIT_USAGE
    assert "not in the report" in capsys.readouterr().err


def test_the_consensus_out_rows_carry_the_hash_of_the_judged_answer(tmp_path: Path) -> None:
    _run(tmp_path, [_spec()])
    out = _read(tmp_path / "out.jsonl")
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "meta": {},
                "records": [
                    {
                        "id": r["id"],
                        "answer_sha256": r["answer_sha256"],
                        "judgments": {
                            "faithfulness": {"status": "ok", "verdict": True},
                            "citation_support": {"status": "ok", "verdict": False},
                        },
                    }
                    for r in out
                ],
            }
        ),
        encoding="utf-8",
    )
    consensus = tmp_path / "consensus.jsonl"

    code = agreement.main(
        ["--labels", str(tmp_path / "labels.jsonl"), "--judgments", str(report),
         "--consensus-out", str(consensus)]
    )  # fmt: skip

    assert code == EXIT_OK
    hashes = {r["id"]: r["answer_sha256"] for r in out}
    rows = _read(consensus)
    assert rows
    assert all(row["answer_sha256"] == hashes[row["id"]] for row in rows)
