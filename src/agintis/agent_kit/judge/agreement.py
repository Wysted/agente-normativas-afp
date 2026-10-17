"""Agreement between a judge and reference labels: Cohen's kappa, confusion matrices, consensus.

Pure functions over booleans (no numpy / sklearn). Labels come from one or several human (or
model) labelers as rows `{id, judge, label: bool[, labeler]}`:

- the JUDGE is compared with the CONSENSUS labels, in two modes reported side by side:
  `unanimous` (an item counts only when EVERY labeler of that judge labeled it and they all agree;
  a single labeler is its own consensus) and `majority` (strictly more than half of the judge's
  labelers give the same label; a tie, or no label reaching half, leaves the item out);
- the LABELERS are compared with each other: Cohen's kappa for every pair on the items both
  labeled and, with three or more labelers, Fleiss' kappa on the items all of them labeled (the
  quality of the reference itself);
- labels are BOUND to what was judged: a row carrying the `answer_sha256` of the dump record is
  dropped when the judged record has another hash (the answer or its sources changed since the
  labeler read it); rows without a hash are kept and counted as unbound.

Confusion cells are `(reference, judge)`: `tp` both true, `fp` reference false / judge true, `fn`
reference true / judge false, `tn` both false. Next to Cohen's kappa each comparison carries
positive and negative agreement (2tp / (2tp + fp + fn) and its mirror), PABAK (2 * po - 1) and
Gwet's AC1, which stay readable when one category dominates (kappa collapses there: the "kappa
paradox"), and flags `small_sample` under `MIN_RELIABLE_ITEMS` compared items.
"""

import itertools
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

DEFAULT_LABELER = "default"
# Below this many compared items a kappa is too noisy to read on its own.
MIN_RELIABLE_ITEMS = 30
ConsensusMode = Literal["unanimous", "majority"]
CONSENSUS_MODES: tuple[ConsensusMode, ...] = ("unanimous", "majority")


class LabelFormatError(ValueError):
    """A label row is malformed or contradicts another one."""


@dataclass(frozen=True, slots=True)
class LabelRow:
    id: str
    judge: str
    label: bool
    labeler: str = DEFAULT_LABELER
    answer_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class Confusion:
    """2x2 table, rows = first rating (reference), columns = second rating (judge)."""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    @classmethod
    def of(cls, first: Sequence[bool], second: Sequence[bool]) -> "Confusion":
        if len(first) != len(second):
            raise ValueError("the two ratings must have the same length")
        pairs = list(zip(first, second, strict=True))
        return cls(
            tp=sum(1 for a, b in pairs if a and b),
            fp=sum(1 for a, b in pairs if not a and b),
            fn=sum(1 for a, b in pairs if a and not b),
            tn=sum(1 for a, b in pairs if not a and not b),
        )

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def agreement(self) -> float | None:
        return (self.tp + self.tn) / self.n if self.n else None

    @property
    def kappa(self) -> float | None:
        """Cohen's kappa: (po - pe) / (1 - pe). None when undefined: no items, or both raters
        used a single category throughout (pe == 1, no variance to agree on)."""
        n = self.n
        if not n:
            return None
        observed = (self.tp + self.tn) / n
        first_true = (self.tp + self.fn) / n
        second_true = (self.tp + self.fp) / n
        expected = first_true * second_true + (1 - first_true) * (1 - second_true)
        if expected == 1.0:
            return None
        return (observed - expected) / (1 - expected)

    @property
    def gwet_ac1(self) -> float | None:
        """Gwet's AC1 for two raters, binary: (po - pe) / (1 - pe) with pe = 2 * pi * (1 - pi),
        pi = the mean share of `true` over both raters. Stable when one category dominates."""
        n = self.n
        if not n:
            return None
        observed = (self.tp + self.tn) / n
        share = ((self.tp + self.fn) + (self.tp + self.fp)) / (2 * n)
        expected = 2 * share * (1 - share)
        return (observed - expected) / (1 - expected)

    @property
    def pabak(self) -> float | None:
        """Prevalence-adjusted bias-adjusted kappa: 2 * po - 1."""
        agreement = self.agreement
        return None if agreement is None else 2 * agreement - 1

    @property
    def sensitivity(self) -> float | None:
        """tp / (tp + fn): the share of reference-true items the judge passes. None without any."""
        total = self.tp + self.fn
        return self.tp / total if total else None

    @property
    def specificity(self) -> float | None:
        """tn / (tn + fp): the share of reference-false items the judge fails. None without any."""
        total = self.tn + self.fp
        return self.tn / total if total else None

    @property
    def pass_rate(self) -> float | None:
        """(tp + fp) / n: the share of compared items the judge passes."""
        return (self.tp + self.fp) / self.n if self.n else None

    @property
    def positive_agreement(self) -> float | None:
        """2tp / (2tp + fp + fn): agreement on the `true` verdicts. None without any `true`."""
        total = 2 * self.tp + self.fp + self.fn
        return 2 * self.tp / total if total else None

    @property
    def negative_agreement(self) -> float | None:
        """2tn / (2tn + fp + fn): agreement on the `false` verdicts. None without any `false`."""
        total = 2 * self.tn + self.fp + self.fn
        return 2 * self.tn / total if total else None

    def to_json(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "agreement": self.agreement,
            "kappa": self.kappa,
            "gwet_ac1": self.gwet_ac1,
            "pabak": self.pabak,
            "positive_agreement": self.positive_agreement,
            "negative_agreement": self.negative_agreement,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
            "pass_rate": self.pass_rate,
            "small_sample": self.n < MIN_RELIABLE_ITEMS,
            "confusion": {"tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn},
        }


def fleiss_kappa(items: Sequence[Sequence[bool]]) -> float | None:
    """Fleiss' kappa of binary ratings: one inner sequence per item, one rating per rater (every
    item rated by the same number of raters, at least 2). None when undefined (no items, or all
    ratings in one category)."""
    if not items:
        return None
    raters = len(items[0])
    if raters < 2 or any(len(item) != raters for item in items):
        raise ValueError("Fleiss' kappa needs the same number (>= 2) of ratings per item")
    agreements: list[float] = []
    true_total = 0
    for item in items:
        true = sum(item)
        true_total += true
        false = raters - true
        agreements.append((true * true + false * false - raters) / (raters * (raters - 1)))
    mean_agreement = sum(agreements) / len(items)
    share = true_total / (len(items) * raters)
    expected = share * share + (1 - share) * (1 - share)
    if expected == 1.0:
        return None
    return (mean_agreement - expected) / (1 - expected)


def parse_label_rows(rows: Iterable[tuple[str, object]]) -> list[LabelRow]:
    """Validate `(where, parsed JSON)` pairs. A repeated (id, judge, labeler) is an error."""
    parsed: list[LabelRow] = []
    seen: dict[tuple[str, str, str], str] = {}
    for where, row in rows:
        if not isinstance(row, dict):
            raise LabelFormatError(f"{where}: a label row must be a JSON object")
        for key in ("id", "judge"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise LabelFormatError(f"{where}: {key!r} must be a non-empty string")
        if not isinstance(row.get("label"), bool):
            raise LabelFormatError(f"{where}: 'label' must be true or false")
        labeler = row.get("labeler", DEFAULT_LABELER)
        if not isinstance(labeler, str) or not labeler:
            raise LabelFormatError(f"{where}: 'labeler' must be a non-empty string")
        digest = row.get("answer_sha256")
        if digest is not None and (not isinstance(digest, str) or not digest):
            raise LabelFormatError(f"{where}: 'answer_sha256' must be a non-empty string")
        key3 = (row["id"], row["judge"], labeler)
        if key3 in seen:
            raise LabelFormatError(f"{where}: {key3} already labeled at {seen[key3]}")
        seen[key3] = where
        parsed.append(LabelRow(row["id"], row["judge"], row["label"], labeler, digest))
    return parsed


@dataclass(frozen=True, slots=True)
class BoundLabels:
    """Labels split by whether they still describe the judged answer."""

    kept: tuple[LabelRow, ...]
    stale: tuple[LabelRow, ...]  # hash differs from the judged record's
    unbound: int  # kept rows that carry no hash (cannot be checked)


def bind_labels(labels: Iterable[LabelRow], hashes: Mapping[str, str]) -> BoundLabels:
    """Drop the rows whose `answer_sha256` differs from the judged record's hash."""
    kept: list[LabelRow] = []
    stale: list[LabelRow] = []
    unbound = 0
    for row in labels:
        expected = hashes.get(row.id)
        if row.answer_sha256 is None:
            unbound += 1
            kept.append(row)
        elif expected is not None and row.answer_sha256 != expected:
            stale.append(row)
        else:
            kept.append(row)
    return BoundLabels(tuple(kept), tuple(stale), unbound)


def _by_item(rows: Iterable[LabelRow]) -> dict[str, dict[str, bool]]:
    items: dict[str, dict[str, bool]] = defaultdict(dict)
    for row in rows:
        items[row.id][row.labeler] = row.label
    return items


@dataclass(frozen=True, slots=True)
class Consensus:
    """The reference labels of ONE judge under one mode."""

    mode: ConsensusMode
    labelers: tuple[str, ...]
    labels: Mapping[str, bool]
    incomplete: tuple[str, ...]  # not enough labelers labeled it (unanimous: some did not)
    disagreed: tuple[str, ...]  # not unanimous / no strict majority


def consensus_labels(rows: Iterable[LabelRow], mode: ConsensusMode = "unanimous") -> Consensus:
    """Unanimous: every labeler of the judge labeled the item and they all agree. Majority: a
    label given by strictly more than half of the judge's labelers (an even-count tie, or a
    split of those who did label, goes to `disagreed` when all labeled, else `incomplete`)."""
    rows = list(rows)
    labelers = tuple(sorted({r.labeler for r in rows}))
    labels: dict[str, bool] = {}
    incomplete: list[str] = []
    disagreed: list[str] = []
    for item, by_labeler in sorted(_by_item(rows).items()):
        true = sum(by_labeler.values())
        false = len(by_labeler) - true
        if mode == "unanimous":
            decided = (true == len(labelers)) or (false == len(labelers))
        else:
            decided = 2 * true > len(labelers) or 2 * false > len(labelers)
        if decided:
            labels[item] = true > false
        elif len(by_labeler) < len(labelers):
            incomplete.append(item)
        else:
            disagreed.append(item)
    return Consensus(mode, labelers, labels, tuple(incomplete), tuple(disagreed))


def inter_labeler(rows: Iterable[LabelRow]) -> list[dict[str, Any]]:
    """Cohen's kappa for every pair of labelers on the items both labeled."""
    rows = list(rows)
    items = _by_item(rows)
    result: list[dict[str, Any]] = []
    for a, b in itertools.combinations(sorted({r.labeler for r in rows}), 2):
        common = sorted(i for i, by in items.items() if a in by and b in by)
        confusion = Confusion.of([items[i][a] for i in common], [items[i][b] for i in common])
        result.append({"labelers": [a, b], **confusion.to_json()})
    return result


def fleiss_for(rows: Iterable[LabelRow]) -> dict[str, Any] | None:
    """Fleiss' kappa over the items ALL labelers labeled; None with fewer than 3 labelers."""
    rows = list(rows)
    labelers = sorted({r.labeler for r in rows})
    if len(labelers) < 3:
        return None
    complete = [
        [by[labeler] for labeler in labelers]
        for _, by in sorted(_by_item(rows).items())
        if len(by) == len(labelers)
    ]
    return {
        "labelers": labelers,
        "n": len(complete),
        "kappa": fleiss_kappa(complete) if complete else None,
        "small_sample": len(complete) < MIN_RELIABLE_ITEMS,
    }


# id -> judge -> (status, verdict) of a judgments report
Verdicts = Mapping[str, Mapping[str, tuple[str, bool | None]]]


def _records(report: Mapping[str, Any]) -> list[Any]:
    records = report.get("records")
    if not isinstance(records, list):
        raise LabelFormatError("the judgments report has no 'records' list")
    return records


def verdicts_from_report(report: Mapping[str, Any]) -> Verdicts:
    """The `(status, verdict)` of every judge for every record of a `cli.judge` report."""
    verdicts: dict[str, dict[str, tuple[str, bool | None]]] = {}
    for record in _records(report):
        try:
            verdicts[record["id"]] = {
                judge: (str(j["status"]), j["verdict"]) for judge, j in record["judgments"].items()
            }
        except (KeyError, TypeError, AttributeError) as exc:
            raise LabelFormatError(
                f"malformed record in the judgments report ({type(exc).__name__})"
            ) from exc
    return verdicts


def hashes_from_report(report: Mapping[str, Any]) -> dict[str, str]:
    """`answer_sha256` of every record of a `cli.judge` report that has one."""
    return {
        r["id"]: r["answer_sha256"]
        for r in _records(report)
        if isinstance(r, dict) and isinstance(r.get("answer_sha256"), str)
    }


UNGROUPED = "(none)"


def _compare(
    consensus: Consensus,
    judge: str,
    verdicts: Verdicts,
    groups: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    reference: list[bool] = []
    verdict: list[bool] = []
    grouped: dict[str, tuple[list[bool], list[bool]]] = defaultdict(lambda: ([], []))
    no_judgment: list[str] = []
    not_decided: list[str] = []
    for item, label in sorted(consensus.labels.items()):
        entry = verdicts.get(item, {}).get(judge)
        if entry is None:
            no_judgment.append(item)
        elif entry[0] != "ok" or entry[1] is None:
            not_decided.append(item)
        else:
            reference.append(label)
            verdict.append(entry[1])
            if groups is not None:
                refs, verds = grouped[groups.get(item, UNGROUPED)]
                refs.append(label)
                verds.append(entry[1])
    by_group = (
        None
        if groups is None
        else {g: Confusion.of(r, v).to_json() for g, (r, v) in sorted(grouped.items())}
    )
    return {
        "consensus_items": len(consensus.labels),
        "incomplete_items": list(consensus.incomplete),
        "disagreed_items": list(consensus.disagreed),
        "judge_vs_consensus": Confusion.of(reference, verdict).to_json(),
        "by_group": by_group,
        "skipped": {"no_judgment": no_judgment, "not_decided": not_decided},
    }


def agreement_report(
    labels: Sequence[LabelRow],
    verdicts: Verdicts,
    hashes: Mapping[str, str] | None = None,
    mode: ConsensusMode = "unanimous",
    groups: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Per judge: labelers, inter-labeler agreement and the judge against the consensus (both
    modes under `by_mode`; the selected `mode` is also repeated at the top level of the entry).
    Under `excluded`: the rows dropped because the answer changed (`stale`) and the count of
    rows that carry no hash (`unbound`). With `groups` (record id -> group, e.g. the perturbation
    type from the dump's `meta`) every comparison also carries `by_group`: the confusion matrix,
    sensitivity, specificity and pass rate of each group (ids without a group go to `(none)`)."""
    bound = bind_labels(labels, hashes or {})
    by_judge: dict[str, list[LabelRow]] = defaultdict(list)
    for row in bound.kept:
        by_judge[row.judge].append(row)
    judges: dict[str, Any] = {}
    for judge, rows in sorted(by_judge.items()):
        by_mode = {
            m: _compare(consensus_labels(rows, m), judge, verdicts, groups) for m in CONSENSUS_MODES
        }
        labelers = sorted({r.labeler for r in rows})
        judges[judge] = {
            "labelers": labelers,
            "labeled_items": len({r.id for r in rows}),
            "inter_labeler": inter_labeler(rows) if len(labelers) > 1 else None,
            "fleiss": fleiss_for(rows),
            "consensus_mode": mode,
            "by_mode": by_mode,
            **by_mode[mode],
        }
    return {
        "judges": judges,
        "excluded": {
            "stale": [{"id": r.id, "judge": r.judge, "labeler": r.labeler} for r in bound.stale],
            "unbound": bound.unbound,
        },
    }


def consensus_rows(
    labels: Sequence[LabelRow],
    mode: ConsensusMode = "unanimous",
    hashes: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """The consensus label set as `{id, judge, label}` rows (sorted); with `hashes` (record id ->
    `answer_sha256`) every row also carries the hash of the answer it was decided on."""
    by_judge: dict[str, list[LabelRow]] = defaultdict(list)
    for row in labels:
        by_judge[row.judge].append(row)
    return [
        {
            "id": item,
            "judge": judge,
            "label": label,
            **({"answer_sha256": hashes[item]} if hashes and item in hashes else {}),
        }
        for judge, rows in sorted(by_judge.items())
        for item, label in sorted(consensus_labels(rows, mode).labels.items())
    ]
