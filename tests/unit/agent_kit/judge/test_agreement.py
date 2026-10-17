import pytest

from agintis.agent_kit.judge import (
    MIN_RELIABLE_ITEMS,
    Confusion,
    LabelFormatError,
    LabelRow,
    agreement_report,
    bind_labels,
    consensus_labels,
    consensus_rows,
    fleiss_for,
    fleiss_kappa,
    hashes_from_report,
    inter_labeler,
    parse_label_rows,
    verdicts_from_report,
)


def kappa(first: list[bool], second: list[bool]) -> float | None:
    return Confusion.of(first, second).kappa


def _ratings(yy: int, yn: int, ny: int, nn: int) -> tuple[list[bool], list[bool]]:
    first = [True] * yy + [True] * yn + [False] * ny + [False] * nn
    second = [True] * yy + [False] * yn + [True] * ny + [False] * nn
    return first, second


# --- Cohen's kappa ------------------------------------------------------------------------------


def test_kappa_wikipedia_example_one() -> None:
    # 50 items: 20 yes/yes, 5 yes/no, 10 no/yes, 15 no/no -> po 0.7, pe 0.5, kappa 0.4
    first, second = _ratings(20, 5, 10, 15)
    assert kappa(first, second) == pytest.approx(0.4)
    confusion = Confusion.of(first, second)
    assert (confusion.tp, confusion.fn, confusion.fp, confusion.tn) == (20, 5, 10, 15)
    assert confusion.agreement == pytest.approx(0.7)


def test_kappa_second_worked_example() -> None:
    # 100 items: 45/15/25/15 -> po 0.6, pe 0.6*0.7 + 0.4*0.3 = 0.54, kappa 0.06/0.46
    first, second = _ratings(45, 15, 25, 15)
    assert kappa(first, second) == pytest.approx(0.06 / 0.46)


def test_kappa_perfect_agreement_is_one() -> None:
    first, second = _ratings(10, 0, 0, 10)
    assert kappa(first, second) == pytest.approx(1.0)


def test_kappa_complete_disagreement_is_minus_one() -> None:
    first, second = _ratings(0, 5, 5, 0)
    assert kappa(first, second) == pytest.approx(-1.0)


def test_kappa_at_chance_level_is_zero() -> None:
    # Independent 50/50 raters: po == pe == 0.5.
    first, second = _ratings(5, 5, 5, 5)
    assert kappa(first, second) == pytest.approx(0.0)


def test_kappa_is_undefined_without_variance_or_items() -> None:
    assert kappa([True, True, True], [True, True, True]) is None
    assert kappa([False, False], [False, False]) is None
    assert kappa([], []) is None
    assert Confusion().agreement is None


def test_kappa_penalises_a_skewed_judge_that_agrees_mostly_by_chance() -> None:
    # 95% agreement looks great, but one rater says "true" nearly always: kappa is far lower.
    first, second = _ratings(90, 0, 5, 5)
    assert Confusion.of(first, second).agreement == pytest.approx(0.95)
    assert kappa(first, second) == pytest.approx(9 / 14)  # po .95, pe .9*.95 + .1*.05 = .86


def test_ratings_of_different_length_are_rejected() -> None:
    with pytest.raises(ValueError, match="same length"):
        kappa([True], [True, False])


# --- label rows ---------------------------------------------------------------------------------


def _rows(*rows: dict[str, object]) -> list[tuple[str, object]]:
    return [(f"l:{i}", row) for i, row in enumerate(rows, start=1)]


def test_label_rows_parse_with_a_default_labeler() -> None:
    parsed = parse_label_rows(
        _rows(
            {"id": "a", "judge": "faithfulness", "label": True},
            {"id": "a", "judge": "faithfulness", "label": False, "labeler": "ana"},
        )
    )
    assert [(r.labeler, r.label) for r in parsed] == [("default", True), ("ana", False)]


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ({"id": "a", "judge": "j", "label": "yes"}, "'label'"),
        ({"id": "a", "judge": "j", "label": 1}, "'label'"),
        ({"id": "", "judge": "j", "label": True}, "'id'"),
        ({"id": "a", "label": True}, "'judge'"),
        ({"id": "a", "judge": "j", "label": True, "labeler": ""}, "'labeler'"),
        ([1], "JSON object"),
    ],
)
def test_malformed_label_rows_are_rejected(row: object, message: str) -> None:
    with pytest.raises(LabelFormatError, match=message):
        parse_label_rows([("l:1", row)])


def test_a_repeated_item_for_the_same_labeler_is_rejected() -> None:
    row = {"id": "a", "judge": "j", "label": True}
    with pytest.raises(LabelFormatError, match="already labeled"):
        parse_label_rows(_rows(row, dict(row)))


# --- consensus and inter-labeler ----------------------------------------------------------------


def _labels(judge: str, **by_labeler: dict[str, bool]) -> list[LabelRow]:
    return [
        LabelRow(item, judge, label, labeler)
        for labeler, labels in by_labeler.items()
        for item, label in labels.items()
    ]


def test_consensus_keeps_only_unanimous_items_labeled_by_everyone() -> None:
    rows = _labels(
        "j",
        ana={"a": True, "b": False, "c": True, "d": True},
        ben={"a": True, "b": True, "c": True},  # no d
    )
    consensus = consensus_labels(rows)

    assert consensus.labelers == ("ana", "ben")
    assert dict(consensus.labels) == {"a": True, "c": True}
    assert consensus.disagreed == ("b",)
    assert consensus.incomplete == ("d",)


def test_a_single_labeler_is_its_own_consensus() -> None:
    consensus = consensus_labels(_labels("j", default={"a": True, "b": False}))
    assert dict(consensus.labels) == {"a": True, "b": False}
    assert consensus.disagreed == ()


def test_inter_labeler_kappa_per_pair_on_common_items() -> None:
    items = [f"i{n}" for n in range(50)]
    first, second = _ratings(20, 5, 10, 15)
    rows = _labels(
        "j",
        ana=dict(zip(items, first, strict=True)),
        ben=dict(zip(items, second, strict=True)),
        cris={"i0": True},
    )
    pairs = {tuple(p["labelers"]): p for p in inter_labeler(rows)}

    ana_ben = pairs[("ana", "ben")]
    assert ana_ben["n"] == 50
    assert ana_ben["kappa"] == pytest.approx(0.4)
    assert pairs[("ana", "cris")]["n"] == 1
    assert pairs[("ana", "cris")]["kappa"] is None  # a single item has no variance


def test_consensus_rows_are_sorted_and_flat() -> None:
    rows = _labels("j2", default={"b": True}) + _labels("j1", default={"z": False, "a": True})
    assert consensus_rows(rows) == [
        {"id": "a", "judge": "j1", "label": True},
        {"id": "z", "judge": "j1", "label": False},
        {"id": "b", "judge": "j2", "label": True},
    ]


# --- judge vs consensus -------------------------------------------------------------------------


REPORT = {
    "records": [
        {"id": "a", "judgments": {"j": {"status": "ok", "verdict": True}}},
        {"id": "b", "judgments": {"j": {"status": "ok", "verdict": False}}},
        {"id": "c", "judgments": {"j": {"status": "ok", "verdict": True}}},
        {"id": "d", "judgments": {"j": {"status": "not_applicable", "verdict": None}}},
        {"id": "e", "judgments": {"j": {"status": "error", "verdict": None}}},
    ]
}


def test_report_compares_the_judge_with_the_consensus_and_lists_what_it_skipped() -> None:
    labels = _labels(
        "j",
        default={"a": True, "b": True, "c": False, "d": True, "e": True, "ghost": True},
    )
    entry = agreement_report(labels, verdicts_from_report(REPORT))["judges"]["j"]

    comparison = entry["judge_vs_consensus"]
    assert comparison["n"] == 3
    # a: ref true / judge true -> tp; b: ref true / judge false -> fn; c: ref false / true -> fp
    assert comparison["confusion"] == {"tp": 1, "fp": 1, "fn": 1, "tn": 0}
    assert comparison["agreement"] == pytest.approx(1 / 3)
    assert entry["skipped"] == {"no_judgment": ["ghost"], "not_decided": ["d", "e"]}
    assert entry["inter_labeler"] is None


def test_report_only_counts_consensus_items() -> None:
    labels = _labels(
        "j",
        ana={"a": True, "b": True, "c": False},
        ben={"a": True, "b": False, "c": False},
    )
    entry = agreement_report(labels, verdicts_from_report(REPORT))["judges"]["j"]

    assert entry["consensus_items"] == 2  # a and c; b is a disagreement
    assert entry["disagreed_items"] == ["b"]
    assert entry["judge_vs_consensus"]["n"] == 2
    assert entry["inter_labeler"][0]["labelers"] == ["ana", "ben"]


def test_a_malformed_judgments_report_is_rejected() -> None:
    with pytest.raises(LabelFormatError, match="records"):
        verdicts_from_report({})
    with pytest.raises(LabelFormatError, match="malformed"):
        verdicts_from_report({"records": [{"id": "a"}]})


# --- review fixes: more metrics, modes, binding ---------------------------------------------------


def test_positive_negative_agreement_pabak_and_ac1() -> None:
    # 20 / 5 / 10 / 15 (tp / fn / fp / tn): po 0.7
    confusion = Confusion(tp=20, fn=5, fp=10, tn=15)
    assert confusion.positive_agreement == pytest.approx(40 / 55)  # 2tp / (2tp + fp + fn)
    assert confusion.negative_agreement == pytest.approx(30 / 45)
    assert confusion.pabak == pytest.approx(0.4)  # 2 * 0.7 - 1
    # pi = (25/50 + 30/50) / 2 = 0.55 -> pe = 2 * 0.55 * 0.45 = 0.495 -> (0.7 - 0.495) / 0.505
    assert confusion.gwet_ac1 == pytest.approx(0.205 / 0.505)


def test_ac1_stays_high_where_kappa_collapses_on_a_skewed_sample() -> None:
    # 96 tp, 2 fp, 2 fn, 0 tn: raters agree 96%, but almost everything is `true`.
    confusion = Confusion(tp=96, fp=2, fn=2, tn=0)
    assert confusion.kappa is not None
    assert confusion.kappa < 0
    assert confusion.gwet_ac1 is not None
    assert confusion.gwet_ac1 > 0.9
    assert confusion.negative_agreement == 0.0


def test_undefined_agreement_metrics_are_none() -> None:
    empty = Confusion()
    assert (empty.gwet_ac1, empty.pabak, empty.positive_agreement) == (None, None, None)
    assert Confusion(tn=3).positive_agreement is None
    assert Confusion(tp=3).negative_agreement is None


def test_a_small_sample_is_flagged() -> None:
    assert Confusion(tp=5).to_json()["small_sample"] is True
    assert Confusion(tp=MIN_RELIABLE_ITEMS).to_json()["small_sample"] is False


def test_fleiss_kappa_hand_computed_example() -> None:
    # 4 items, 3 raters: TTT, TTF, FFF, TFF -> P = 1, 1/3, 1, 1/3 (mean 2/3), pe = 0.5 -> 1/3
    items = [
        [True, True, True],
        [True, True, False],
        [False, False, False],
        [True, False, False],
    ]
    assert fleiss_kappa(items) == pytest.approx(1 / 3)


def test_fleiss_kappa_extremes_and_degenerate_cases() -> None:
    assert fleiss_kappa([[True, True, True], [False, False, False]]) == pytest.approx(1.0)
    assert fleiss_kappa([[True, True, True]] * 3) is None
    assert fleiss_kappa([]) is None
    with pytest.raises(ValueError, match="same number"):
        fleiss_kappa([[True, False], [True, False, True]])


def test_fleiss_is_reported_only_with_three_or_more_labelers() -> None:
    two = _labels("j", ana={"a": True}, ben={"a": True})
    assert fleiss_for(two) is None
    three = _labels("j", ana={"a": True, "b": False}, ben={"a": True, "b": True}, cris={"a": True})
    entry = fleiss_for(three)
    assert entry is not None
    assert entry["labelers"] == ["ana", "ben", "cris"]
    assert entry["n"] == 1  # only `a` was labeled by all three


def test_majority_needs_strictly_more_than_half_of_the_labelers() -> None:
    rows = _labels(
        "j",
        ana={"a": True, "b": True, "c": True, "d": True},
        ben={"a": True, "b": True, "c": False, "d": False},
        cris={"a": True, "b": False, "c": False},
    )
    unanimous = consensus_labels(rows, "unanimous")
    majority = consensus_labels(rows, "majority")

    assert dict(unanimous.labels) == {"a": True}
    assert dict(majority.labels) == {"a": True, "b": True, "c": False}
    assert majority.incomplete == ("d",)  # 1 true, 1 false, one labeler missing: no majority yet


def test_an_even_count_tie_goes_to_disagreed() -> None:
    rows = _labels("j", ana={"a": True}, ben={"a": False})
    assert consensus_labels(rows, "majority").disagreed == ("a",)
    assert dict(consensus_labels(rows, "majority").labels) == {}


def test_a_single_labeler_is_the_same_in_both_modes() -> None:
    rows = _labels("j", default={"a": True, "b": False})
    assert consensus_labels(rows, "majority").labels == consensus_labels(rows, "unanimous").labels


def test_the_report_carries_both_modes_and_the_selected_one_on_top() -> None:
    rows = _labels(
        "j",
        ana={"a": True, "b": True, "c": False},
        ben={"a": True, "b": False, "c": False},
        cris={"a": True, "b": True, "c": False},
    )
    entry = agreement_report(rows, verdicts_from_report(REPORT), mode="majority")["judges"]["j"]

    assert set(entry["by_mode"]) == {"unanimous", "majority"}
    assert entry["by_mode"]["unanimous"]["consensus_items"] == 2  # a, c
    assert entry["by_mode"]["majority"]["consensus_items"] == 3  # a, b (2 of 3), c
    assert entry["consensus_mode"] == "majority"
    assert entry["consensus_items"] == 3
    assert entry["fleiss"]["n"] == 3
    assert len(entry["inter_labeler"]) == 3


def test_labels_bound_to_another_answer_are_excluded_before_anything_is_counted() -> None:
    rows = [
        LabelRow("a", "j", True, "ana", "hash-a"),
        LabelRow("b", "j", True, "ana", "OLD-hash"),  # the answer changed since
        LabelRow("c", "j", False, "ana", None),  # unbound: kept, counted
    ]
    bound = bind_labels(rows, {"a": "hash-a", "b": "hash-b", "c": "hash-c"})
    assert [r.id for r in bound.kept] == ["a", "c"]
    assert [r.id for r in bound.stale] == ["b"]
    assert bound.unbound == 1

    report = agreement_report(rows, verdicts_from_report(REPORT), {"a": "hash-a", "b": "hash-b"})
    assert report["excluded"] == {
        "stale": [{"id": "b", "judge": "j", "labeler": "ana"}],
        "unbound": 1,
    }
    assert report["judges"]["j"]["labeled_items"] == 2


def test_a_label_for_an_id_the_report_has_no_hash_for_is_kept() -> None:
    bound = bind_labels([LabelRow("zzz", "j", True, "ana", "h")], {"a": "hash-a"})
    assert len(bound.kept) == 1


def test_hashes_come_from_the_report_records() -> None:
    report = {"records": [{"id": "a", "answer_sha256": "h", "judgments": {}}, {"id": "b"}]}
    assert hashes_from_report(report) == {"a": "h"}


def test_label_rows_accept_an_optional_hash_and_reject_a_bad_one() -> None:
    row = {"id": "a", "judge": "j", "label": True, "answer_sha256": "abc"}
    assert parse_label_rows([("l:1", row)])[0].answer_sha256 == "abc"
    with pytest.raises(LabelFormatError, match="answer_sha256"):
        parse_label_rows([("l:1", {**row, "answer_sha256": 5})])


def test_sensitivity_specificity_and_pass_rate() -> None:
    c = Confusion(tp=3, fp=1, fn=1, tn=3)
    assert (c.sensitivity, c.specificity, c.pass_rate) == (0.75, 0.75, 0.5)
    assert Confusion(tn=2).sensitivity is None
    assert Confusion(tp=2).specificity is None
    assert Confusion().pass_rate is None


def test_agreement_report_breaks_the_comparison_down_by_group() -> None:
    verdicts = {
        "a": {"j": ("ok", True)},
        "b": {"j": ("ok", True)},
        "c": {"j": ("ok", False)},
        "d": {"j": ("ok", True)},
    }
    labels = [
        LabelRow("a", "j", True),
        LabelRow("b", "j", False),
        LabelRow("c", "j", False),
        LabelRow("d", "j", False),
    ]

    entry = agreement_report(labels, verdicts, groups={"b": "neg", "c": "neg", "d": "qty"})[
        "judges"
    ]["j"]

    groups = entry["by_group"]
    assert groups["(none)"]["confusion"] == {"tp": 1, "fp": 0, "fn": 0, "tn": 0}
    assert groups["neg"]["specificity"] == 0.5
    assert groups["qty"]["pass_rate"] == 1.0
    assert agreement_report(labels, verdicts)["judges"]["j"]["by_group"] is None
