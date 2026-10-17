import json
from pathlib import Path
from typing import Any

import pytest

from agintis.cli import agreement
from agintis.cli._common import EXIT_OK, EXIT_USAGE, UsageError
from agintis.cli.agreement import build_report, load_labels, load_report


def _jsonl(path: Path, *rows: Any) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _report(path: Path, *, hashed: bool = True) -> Path:
    # 10 records: the judge says true for a-e and false for f-j; one record is undecided.
    records = [
        {
            "id": item,
            **({"answer_sha256": f"hash-{item}"} if hashed else {}),
            "judgments": {
                "faithfulness": {
                    "status": "ok" if item != "j" else "not_applicable",
                    "verdict": item in "abcde" if item != "j" else None,
                }
            },
        }
        for item in "abcdefghij"
    ]
    path.write_text(
        json.dumps(
            {
                "meta": {
                    "judge_model": "judge (x:y)",
                    "judge_version": "8",
                    "judge_prompts_sha256": "p",
                    "answers_sha256": "h",
                },
                "records": records,
            }
        ),
        encoding="utf-8",
    )
    return path


def _label(item: str, label: bool, labeler: str | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {"id": item, "judge": "faithfulness", "label": label}
    if labeler:
        row["labeler"] = labeler
    return row


def test_main_prints_and_writes_agreement_and_the_consensus_set(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ana = {i: i in "abcdh" for i in "abcdefghi"}  # a-d true, h true; the rest false
    ben = {**ana, "d": False}  # disagrees with ana on d
    rows = [
        {**_label(i, v, who), "answer_sha256": f"hash-{i}"}
        for who, labeled in (("ana", ana), ("ben", ben))
        for i, v in labeled.items()
        if (who, i) != ("ben", "i")
    ]
    out, consensus = tmp_path / "agreement.json", tmp_path / "consensus.jsonl"

    code = agreement.main(
        [
            "--labels", str(_jsonl(tmp_path / "labels.jsonl", *rows)),
            "--judgments", str(_report(tmp_path / "report.json")),
            "--out", str(out),
            "--consensus-out", str(consensus),
        ]
    )  # fmt: skip

    assert code == EXIT_OK
    entry = json.loads(out.read_text(encoding="utf-8"))["judges"]["faithfulness"]
    assert entry["labelers"] == ["ana", "ben"]
    assert entry["disagreed_items"] == ["d"]
    assert entry["incomplete_items"] == ["i"]
    assert entry["inter_labeler"][0]["n"] == 8  # items both labeled
    # consensus: a b c true, e f g false, h true (d disagreed, i incomplete)
    comparison = entry["judge_vs_consensus"]
    # judge: a-e true, f-j false. Reference: a b c h true / e f g false.
    # a b c: tp (3); h: fn (ref true, judge false); e: fp (ref false, judge true); f g: tn (2)
    assert comparison["confusion"] == {"tp": 3, "fp": 1, "fn": 1, "tn": 2}
    assert comparison["n"] == 7
    rows_out = [json.loads(line) for line in consensus.read_text().splitlines()]
    assert {r["id"]: r["label"] for r in rows_out} == {
        "a": True, "b": True, "c": True, "e": False, "f": False, "g": False, "h": True,
    }  # fmt: skip
    assert all(r["answer_sha256"] == f"hash-{r['id']}" for r in rows_out)
    text = capsys.readouterr().out
    assert "faithfulness" in text
    assert "inter-labeler ana vs ben" in text
    assert "kappa" in text


def test_a_single_labeler_without_a_labeler_field_works(tmp_path: Path) -> None:
    labels = load_labels(_jsonl(tmp_path / "l.jsonl", _label("a", True), _label("f", False)))
    report = build_report(labels, load_report(_report(tmp_path / "r.json")), allow_unbound=True)

    entry = report["judges"]["faithfulness"]
    assert entry["labelers"] == ["default"]
    assert entry["inter_labeler"] is None
    assert entry["judge_vs_consensus"]["agreement"] == 1.0
    assert report["meta"]["judge_model"] == "judge (x:y)"
    assert report["meta"]["judge_version"] == "8"
    assert report["meta"]["judge_prompts_sha256"] == "p"


def test_labels_for_a_judge_the_report_lacks_are_a_usage_error(tmp_path: Path) -> None:
    labels = load_labels(_jsonl(tmp_path / "l.jsonl", {"id": "a", "judge": "vibes", "label": True}))
    with pytest.raises(UsageError, match="vibes"):
        build_report(labels, load_report(_report(tmp_path / "r.json")), allow_unbound=True)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("not json\n", "invalid JSON"),
        ('{"id": "a", "judge": "j", "label": "yes"}\n', "'label'"),
        ("", "no labels"),
    ],
)
def test_bad_label_files_are_usage_errors(tmp_path: Path, content: str, message: str) -> None:
    path = tmp_path / "l.jsonl"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(UsageError, match=message):
        load_labels(path)


def test_a_bad_report_is_a_usage_error(tmp_path: Path) -> None:
    path = tmp_path / "r.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(UsageError, match="JSON object"):
        load_report(path)
    path.write_text("{", encoding="utf-8")
    with pytest.raises(UsageError, match="cannot read"):
        load_report(path)
    with pytest.raises(UsageError, match="'records'"):
        build_report(load_labels(_jsonl(tmp_path / "l.jsonl", _label("a", True))), {})


def test_main_maps_usage_errors_to_exit_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = agreement.main(
        ["--labels", str(tmp_path / "nope.jsonl"), "--judgments", str(_report(tmp_path / "r.json"))]
    )
    assert code == EXIT_USAGE
    assert "cannot read" in capsys.readouterr().err
    code = agreement.main(
        ["--labels", str(_jsonl(tmp_path / "l.jsonl", _label("a", True))),
         "--judgments", str(tmp_path / "r.json"), "--out", str(tmp_path)]
    )  # fmt: skip
    assert code == EXIT_USAGE
    assert "is a directory" in capsys.readouterr().err


# --- review fixes -------------------------------------------------------------------------------


def _hashed_report(path: Path) -> Path:
    records = [
        {
            "id": item,
            "answer_sha256": f"hash-{item}",
            "judgments": {"faithfulness": {"status": "ok", "verdict": item in "abc"}},
        }
        for item in "abcd"
    ]
    path.write_text(json.dumps({"meta": {}, "records": records}), encoding="utf-8")
    return path


def test_labels_for_a_changed_answer_are_excluded_with_a_warning(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = [
        {**_label("a", True), "answer_sha256": "hash-a"},
        {**_label("b", True), "answer_sha256": "STALE"},  # labeled against another answer
        {**_label("c", False)},  # no hash: kept, but unbound
        {**_label("d", False), "answer_sha256": "hash-d"},
    ]
    out = tmp_path / "out.json"

    code = agreement.main(
        ["--allow-unbound", "--labels", str(_jsonl(tmp_path / "l.jsonl", *rows)),
         "--judgments", str(_hashed_report(tmp_path / "r.json")), "--out", str(out)]
    )  # fmt: skip

    assert code == EXIT_OK
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["excluded"]["stale"] == [
        {"id": "b", "judge": "faithfulness", "labeler": "default"}
    ]
    assert report["excluded"]["unbound"] == 1
    assert report["judges"]["faithfulness"]["labeled_items"] == 3  # b is gone
    err = capsys.readouterr().err
    assert "1 label rows EXCLUDED" in err
    assert "carry no answer_sha256" in err


def test_the_consensus_file_leaves_out_the_excluded_rows(tmp_path: Path) -> None:
    rows = [
        {**_label("a", True), "answer_sha256": "hash-a"},
        {**_label("b", True), "answer_sha256": "STALE"},
    ]
    consensus = tmp_path / "c.jsonl"
    agreement.main(
        ["--labels", str(_jsonl(tmp_path / "l.jsonl", *rows)),
         "--judgments", str(_hashed_report(tmp_path / "r.json")), "--consensus-out", str(consensus)]
    )  # fmt: skip
    assert [json.loads(x)["id"] for x in consensus.read_text().splitlines()] == ["a"]


def test_the_consensus_flag_selects_the_headline_mode_and_both_are_reported(
    tmp_path: Path,
) -> None:
    rows = [_label(i, v, who) for i, v, who in [
        ("a", True, "ana"), ("a", True, "ben"), ("a", True, "cris"),
        ("b", True, "ana"), ("b", True, "ben"), ("b", False, "cris"),
        ("c", False, "ana"), ("c", False, "ben"), ("c", False, "cris"),
    ]]  # fmt: skip
    outputs = {}
    for mode in ("unanimous", "majority"):
        out = tmp_path / f"{mode}.json"
        agreement.main(
            ["--allow-unbound", "--labels", str(_jsonl(tmp_path / "l.jsonl", *rows)),
             "--judgments", str(_hashed_report(tmp_path / "r.json")),
             "--consensus", mode, "--out", str(out)]
        )  # fmt: skip
        outputs[mode] = json.loads(out.read_text(encoding="utf-8"))["judges"]["faithfulness"]

    assert outputs["unanimous"]["consensus_items"] == 2
    assert outputs["majority"]["consensus_items"] == 3
    assert outputs["majority"]["consensus_mode"] == "majority"
    assert set(outputs["unanimous"]["by_mode"]) == {"unanimous", "majority"}
    assert outputs["majority"]["fleiss"]["n"] == 3


def test_the_report_flag_is_gone_in_favour_of_judgments(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        agreement.main(["--labels", str(tmp_path / "l"), "--report", str(tmp_path / "r")])


def test_unbound_label_rows_are_rejected_unless_explicitly_allowed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = [{**_label("a", True), "answer_sha256": "hash-a"}, _label("b", True)]  # b: no hash
    args = ["--labels", str(_jsonl(tmp_path / "l.jsonl", *rows)),
            "--judgments", str(_hashed_report(tmp_path / "r.json"))]  # fmt: skip

    assert agreement.main(args) == EXIT_USAGE
    assert "1 label rows carry no answer_sha256" in capsys.readouterr().err
    assert agreement.main([*args, "--allow-unbound"]) == EXIT_OK


def test_a_judgments_report_without_hashes_fails_unless_explicitly_allowed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = [{**_label("a", True), "answer_sha256": "hash-a"}]
    args = ["--labels", str(_jsonl(tmp_path / "l.jsonl", *rows)),
            "--judgments", str(_report(tmp_path / "r.json", hashed=False))]  # fmt: skip

    assert agreement.main(args) == EXIT_USAGE
    assert "carry no answer_sha256" in capsys.readouterr().err
    assert agreement.main([*args, "--allow-unbound"]) == EXIT_OK  # hashed rows cannot be checked


def test_fully_bound_labels_need_no_flag(tmp_path: Path) -> None:
    rows = [{**_label(i, True), "answer_sha256": f"hash-{i}"} for i in "abc"]
    code = agreement.main(
        ["--labels", str(_jsonl(tmp_path / "l.jsonl", *rows)),
         "--judgments", str(_hashed_report(tmp_path / "r.json"))]
    )  # fmt: skip
    assert code == EXIT_OK


def test_consensus_out_is_refused_together_with_allow_unbound(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # an unhashed label row would get the report's hash stamped on it by the consensus file
    consensus = tmp_path / "c.jsonl"
    code = agreement.main(
        ["--allow-unbound", "--labels", str(_jsonl(tmp_path / "l.jsonl", _label("a", True))),
         "--judgments", str(_hashed_report(tmp_path / "r.json")), "--consensus-out", str(consensus)]
    )  # fmt: skip
    assert code == EXIT_USAGE
    assert "--allow-unbound" in capsys.readouterr().err
    assert not consensus.exists()
