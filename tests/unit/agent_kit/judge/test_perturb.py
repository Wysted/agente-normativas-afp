import pytest

from agintis.agent_kit.judge import (
    PERTURBATION_LABELER,
    AnswerRecord,
    MalformedSpec,
    PerturbationSpec,
    ScopeVerdict,
    SkipReason,
    SpecFormatError,
    SplitLeakError,
    apply_spec,
    build_perturbed,
    perturbation_labels,
    record_from_json,
    spec_from_json,
)
from agintis.agent_kit.judge.perturb import check_spec
from tests.unit.agent_kit.judge.support import make_record

ANSWER = "El plazo es de 24 meses [1]. La tasa es del 10% [2]."


def _spec(**patch: object) -> PerturbationSpec:
    row = {
        "record_id": "q1",
        "split": "dev",
        "type": "number_date",
        "original": "24 meses",
        "replacement": "12 meses",
        "cited": [1],
        "why": "changes the term",
    }
    return spec_from_json({**row, **patch})


def test_apply_replaces_the_span_once_and_keeps_the_rest() -> None:
    source = make_record(ANSWER)
    perturbed = apply_spec(source, _spec(), 1)

    assert perturbed.answer == "El plazo es de 12 meses [1]. La tasa es del 10% [2]."
    assert perturbed.id == "q1~p1"
    assert (perturbed.question, perturbed.expected_scope, perturbed.sources) == (
        source.question,
        source.expected_scope,
        source.sources,
    )
    assert perturbed.meta["git"] == "abc"  # the source meta is kept
    assert perturbed.meta["source_id"] == "q1"
    assert perturbed.meta["type"] == "number_date"
    assert perturbed.meta["original"] == "24 meses"
    assert perturbed.meta["replacement"] == "12 meses"
    assert perturbed.meta["why"] == "changes the term"
    assert source.answer == ANSWER  # the source is not mutated


def test_the_hash_is_recomputed_and_the_record_loads_back() -> None:
    source = make_record(ANSWER)
    perturbed = apply_spec(source, _spec(), 2)

    assert perturbed.answer_sha256 != source.answer_sha256
    assert record_from_json(perturbed.to_json()) == perturbed


@pytest.mark.parametrize(
    ("patch", "reason"),
    [
        ({"original": "no such span"}, SkipReason.MISSING),
        ({"original": "es "}, SkipReason.AMBIGUOUS),  # "es de" and "es del"
        ({"record_id": "ghost"}, SkipReason.UNKNOWN_RECORD),
        ({"type": "vibes"}, SkipReason.UNKNOWN_TYPE),
        ({"replacement": "24 meses"}, SkipReason.NO_CHANGE),
    ],
)
def test_invalid_specs_are_skipped_with_a_reason(
    patch: dict[str, object], reason: SkipReason
) -> None:
    dump = build_perturbed([make_record(ANSWER)], [("s:1", _spec(**patch))])

    assert dump.perturbed == ()
    assert [s.reason for s in dump.skipped] == [reason]
    assert dump.skipped[0].where == "s:1"


def test_apply_refuses_a_spec_that_cannot_be_applied() -> None:
    with pytest.raises(ValueError, match="cannot be applied"):
        apply_spec(make_record(ANSWER), _spec(original="nope"), 1)


@pytest.mark.parametrize(
    "patch",
    [
        {"split": "train"},
        {"original": ""},
        {"replacement": 3},
        {"cited": "1"},
        {"cited": [0]},
        {"cited": [True]},
        {"why": 1},
    ],
)
def test_malformed_spec_rows_raise(patch: dict[str, object]) -> None:
    with pytest.raises(SpecFormatError):
        _spec(**patch)


def test_an_empty_replacement_is_a_valid_deletion() -> None:
    perturbed = apply_spec(make_record(ANSWER), _spec(replacement=""), 1)
    assert "24 meses" not in perturbed.answer


def test_k_counts_the_specs_of_a_record_over_the_whole_input_whatever_the_split() -> None:
    records = [
        make_record(ANSWER, id="q1"),
        make_record(ANSWER, id="q2"),
        make_record(ANSWER, id="q3"),
    ]
    specs = [
        ("a", _spec(record_id="q1")),
        ("b", _spec(record_id="q1", original="10%", replacement="20%", cited=[2])),
        ("c", _spec(record_id="q3", split="test", original="10%", replacement="20%", cited=[2])),
        ("d", _spec(record_id="q2", original="nope")),  # skipped but it still takes k=1
        ("e", _spec(record_id="q2", original="10%", replacement="30%", cited=[2])),
    ]

    dump = build_perturbed(records, specs, split="dev")

    assert [r.id for r in dump.perturbed] == ["q1~p1", "q1~p2", "q2~p2"]
    assert [r.id for r in dump.originals] == ["q1", "q2"]  # q3 is of the other split
    assert {r.meta["split"] for r in dump.perturbed} == {"dev"}
    # a skipped spec adds no original
    only_skipped = build_perturbed(records, [("d", _spec(record_id="q2", original="nope"))])
    assert only_skipped.originals == ()


@pytest.mark.parametrize("split", [None, "dev", "test"])
def test_a_record_perturbed_in_two_splits_is_a_leak_whatever_the_filter(split: str | None) -> None:
    records = [make_record(ANSWER, id="q1")]
    specs = [
        ("a", _spec(split="dev")),
        ("b", _spec(split="test", original="10%", replacement="20%", cited=[2])),
    ]
    with pytest.raises(SplitLeakError, match="'dev' and 'test'"):
        build_perturbed(records, specs, split=split)


@pytest.mark.parametrize("fragment", ["[1", "1]", "plazo [1", "1] y"])
def test_half_a_marker_in_original_or_replacement_is_skipped(fragment: str) -> None:
    record = make_record(ANSWER, id="q1")
    for spec in (_spec(original=fragment), _spec(replacement=fragment)):
        problem = check_spec(spec, record)
        assert problem is not None
        assert problem[0] is SkipReason.TOUCHES_MARKER


def test_labels_follow_the_emission_rules() -> None:
    records = [make_record(ANSWER, id="q1"), make_record(ANSWER, id="q2")]
    dump = build_perturbed(
        records,
        [
            ("a", _spec(record_id="q1", cited=[1])),
            ("b", _spec(record_id="q2", cited=[], original="10%", replacement="20%")),
        ],
    )

    rows = perturbation_labels(dump.originals, dump.perturbed)

    got = {(r.id, r.judge): r.label for r in rows}
    assert got == {
        ("q1", "faithfulness"): True,
        ("q2", "faithfulness"): True,
        ("q1~p1", "faithfulness"): False,
        ("q1~p1", "citation_support"): False,  # cited sentence no longer holds
        ("q2~p1", "faithfulness"): False,  # no cited: no citation label
    }
    assert {r.labeler for r in rows} == {PERTURBATION_LABELER}
    by_record: dict[str, AnswerRecord] = {r.id: r for r in [*dump.originals, *dump.perturbed]}
    assert all(r.answer_sha256 == by_record[r.id].answer_sha256 for r in rows)
    assert not any(r.judge in ("answer_relevance", "refusal_correctness") for r in rows)


def test_a_malformed_row_keeps_its_numbering_slot_and_is_reported() -> None:
    records = [make_record(ANSWER, id="q1")]
    specs: list[tuple[str, PerturbationSpec | MalformedSpec]] = [
        ("a", MalformedSpec("q1", "'original' must be a non-empty string")),
        ("b", MalformedSpec(None, "a spec must be a JSON object")),  # no id: no slot
        ("c", _spec(record_id="q1")),
    ]

    dump = build_perturbed(records, specs)

    assert [r.id for r in dump.perturbed] == ["q1~p2"]
    assert [(s.where, s.reason) for s in dump.skipped] == [
        ("a", SkipReason.MALFORMED),
        ("b", SkipReason.MALFORMED),
    ]


@pytest.mark.parametrize(
    ("record_patch", "spec_patch", "reason"),
    [
        ({"error": "RuntimeError"}, {}, SkipReason.RECORD_ERROR),
        ({"sources": ()}, {}, SkipReason.NO_SOURCES),
        ({"scope": ScopeVerdict("out", 0.1)}, {}, SkipReason.RECORD_REFUSED),
        ({}, {"original": "[1]", "replacement": "[2]"}, SkipReason.TOUCHES_MARKER),
        ({}, {"original": "meses [1]"}, SkipReason.TOUCHES_MARKER),
        ({}, {"replacement": "12 meses [3]"}, SkipReason.TOUCHES_MARKER),
        ({}, {"cited": [2]}, SkipReason.CITED_MISMATCH),  # "24 meses" sits in the [1] sentence
        ({}, {"cited": [1, 2]}, SkipReason.CITED_MISMATCH),
        ({}, {"original": "[1]. La tasa"}, SkipReason.TOUCHES_MARKER),
    ],
)
def test_the_new_guards_skip_unusable_specs(
    record_patch: dict[str, object], spec_patch: dict[str, object], reason: SkipReason
) -> None:
    record = make_record(ANSWER, **record_patch)  # type: ignore[arg-type]
    dump = build_perturbed([record], [("s:1", _spec(**spec_patch))])
    assert [s.reason for s in dump.skipped] == [reason]


def test_a_span_that_crosses_a_sentence_boundary_is_skipped_when_it_names_cited() -> None:
    answer = "Regla uno cita [1]. Sigue la segunda parte. Fin [2]."
    dump = build_perturbed(
        [make_record(answer)],
        [("s:1", _spec(original=". Sigue la", replacement=". Omite la", cited=[1]))],
    )
    assert [s.reason for s in dump.skipped] == [SkipReason.NOT_IN_SENTENCE]


def test_a_span_inside_a_sentence_that_cites_the_numbers_is_accepted() -> None:
    answer = "El plazo es de 24 meses [1][2]. La tasa es del 10% [2]."
    dump = build_perturbed(
        [make_record(answer)], [("s:1", _spec(cited=[1, 2])), ("s:2", _spec(cited=[]))]
    )
    assert [r.id for r in dump.perturbed] == ["q1~p1", "q1~p2"]
    assert not any(s.reason is SkipReason.CITED_MISMATCH for s in dump.skipped)


def test_condition_deletion_is_a_known_type() -> None:
    answer = "El titular podrá ceder el puesto siempre que la Dirección lo autorice [1]."
    spec = _spec(
        type="condition_deletion", original=" siempre que la Dirección lo autorice", replacement=""
    )
    dump = build_perturbed([make_record(answer)], [("s:1", spec)])
    assert dump.skipped == ()
    assert "siempre que" not in dump.perturbed[0].answer


def test_originals_get_a_citation_label_only_when_the_truth_confirms_them() -> None:
    records = [make_record(ANSWER, id="q1"), make_record(ANSWER, id="q2")]
    dump = build_perturbed(records, [("a", _spec(record_id="q1")), ("b", _spec(record_id="q2"))])

    rows = perturbation_labels(dump.originals, dump.perturbed, citation_truth={"q2"})

    got = {(r.id, r.judge): r.label for r in rows}
    assert ("q1", "citation_support") not in got
    assert got[("q2", "citation_support")] is True
    assert ("q2~p1", "citation_support") in got  # perturbed rows are unchanged
