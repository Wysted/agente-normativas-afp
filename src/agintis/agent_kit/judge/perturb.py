"""Controlled perturbations: meta-evaluation of the judges with labels known by construction.

A faithful original answer must PASS the faithfulness judge; the same answer with ONE injected
error (a `PerturbationSpec`: a span of the answer replaced once) must FAIL it. No human labels
are needed: the original is faithful by selection, the perturbed answer is unfaithful by
construction. `cli.perturb` builds the perturbed dump and the label rows from these pieces and
`cli.agreement` reports the judge's sensitivity (originals passing) and specificity (perturbed
answers failing), per perturbation type with `--by-meta type`.

```json
{"record_id": "in-cmp-02", "split": "dev", "type": "condition_inversion",
 "original": "si el afiliado cotiza", "replacement": "si el afiliado no cotiza",
 "cited": [1], "why": "inverts the condition of the benefit"}
```

`original` must occur EXACTLY ONCE in the source answer (otherwise the span is ambiguous and the
spec is skipped, see `SkipReason`). The perturbed record keeps the question, expected scope,
scope verdict and sources of its source and gets the id `<record_id>~p<k>`; its `answer_sha256`
is recomputed by `AnswerRecord` (the one canonical hash), so labels bind to the perturbed text.
The provenance goes in the free-form `meta` of the record.
"""

import re
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Literal, get_args

from agintis.agent_kit.judge.agreement import LabelRow
from agintis.agent_kit.judge.citations import CITATION, citation_numbers, split_sentences
from agintis.agent_kit.judge.records import AnswerRecord
from agintis.agent_kit.judge.refusal import refused_by_scope
from agintis.agent_kit.judge.results import JudgeName

PerturbationType = Literal[
    "negation",
    "condition_inversion",
    "condition_deletion",
    "quantifier",
    "number_date",
    "entity_swap",
    "unsupported_addition",
]
PERTURBATION_TYPES: tuple[str, ...] = get_args(PerturbationType)
SPLITS: tuple[str, ...] = ("dev", "test")
PERTURBATION_LABELER = "perturbation"
# Half a `[n]` marker: "[1" or "1]" (a span cut through a citation is a citation error too).
_PARTIAL_MARKER = re.compile(r"\[\s*\d|\d\s*\]")


class SkipReason(StrEnum):
    MALFORMED = "malformed spec"
    UNKNOWN_TYPE = "unknown type"
    UNKNOWN_RECORD = "unknown record_id"
    MISSING = "original not found in the answer"
    AMBIGUOUS = "original occurs more than once in the answer"
    NO_CHANGE = "original equals replacement"
    TOUCHES_MARKER = "original or replacement touches a [n] citation marker"
    RECORD_ERROR = "the source record is a failed run"
    NO_SOURCES = "the source record has no sources"
    RECORD_REFUSED = "the source record was refused by the scope check"
    NOT_IN_SENTENCE = "original does not lie inside one sentence"
    CITED_MISMATCH = "the sentence of original does not cite the spec's cited numbers"


class SpecFormatError(ValueError):
    """A spec row does not follow the schema; the message is the skip detail."""


class SplitLeakError(ValueError):
    """One record is perturbed in two different splits: dev and test would share its answer."""


@dataclass(frozen=True, slots=True)
class PerturbationSpec:
    record_id: str
    split: str
    type: str
    original: str
    replacement: str
    cited: tuple[int, ...] = ()
    why: str = ""


@dataclass(frozen=True, slots=True)
class MalformedSpec:
    """A spec row that did not parse. It still takes its place in the numbering of its record
    (`record_id` when the row carries a usable one), so fixing it never renumbers the others."""

    record_id: str | None
    detail: str


@dataclass(frozen=True, slots=True)
class SkippedSpec:
    where: str
    reason: SkipReason
    detail: str = ""

    def __str__(self) -> str:
        return f"{self.where}: {self.reason.value}" + (f" ({self.detail})" if self.detail else "")


def _string(row: Mapping[str, Any], key: str, *, allow_empty: bool = False) -> str:
    value = row.get(key)
    if not isinstance(value, str) or (not value and not allow_empty):
        raise SpecFormatError(
            f"{key!r} must be a {'string' if allow_empty else 'non-empty string'}"
        )
    return value


def spec_from_json(row: object) -> PerturbationSpec:
    """Parse one spec row (`SpecFormatError` names the problem). The type is NOT checked here:
    an unknown type is a skip reason of its own (`SkipReason.UNKNOWN_TYPE`)."""
    if not isinstance(row, dict):
        raise SpecFormatError("a spec must be a JSON object")
    split = _string(row, "split")
    if split not in SPLITS:
        raise SpecFormatError(f"'split' must be one of {SPLITS}")
    cited = row.get("cited", [])
    if not isinstance(cited, list) or not all(
        isinstance(n, int) and not isinstance(n, bool) and n > 0 for n in cited
    ):
        raise SpecFormatError("'cited' must be a list of positive integers")
    why = row.get("why", "")
    if not isinstance(why, str):
        raise SpecFormatError("'why' must be a string")
    return PerturbationSpec(
        record_id=_string(row, "record_id"),
        split=split,
        type=_string(row, "type"),
        original=_string(row, "original"),
        replacement=_string(row, "replacement", allow_empty=True),
        cited=tuple(cited),
        why=why,
    )


def perturbed_id(record_id: str, k: int) -> str:
    """The id of the k-th (1-based) perturbation of a record."""
    return f"{record_id}~p{k}"


def check_spec(
    spec: PerturbationSpec, record: AnswerRecord | None
) -> tuple[SkipReason, str] | None:
    """Why the spec cannot be applied to `record` (None = it can).

    Beyond the span itself, the record must be a usable faithful original (not a failed run, not a
    scope refusal, with sources), the span must not touch a `[n]` marker (that would be a
    citation error, not the injected one) and, when the spec names `cited` numbers, the span must
    lie inside ONE sentence (the splitter of `citations`) that cites all of them."""
    if spec.type not in PERTURBATION_TYPES:
        return SkipReason.UNKNOWN_TYPE, spec.type
    if record is None:
        return SkipReason.UNKNOWN_RECORD, spec.record_id
    if record.error is not None:
        return SkipReason.RECORD_ERROR, record.error
    if not record.sources:
        return SkipReason.NO_SOURCES, ""
    if refused_by_scope(record):
        return SkipReason.RECORD_REFUSED, ""
    if spec.original == spec.replacement:
        return SkipReason.NO_CHANGE, ""
    if any(
        m.search(t) for m in (CITATION, _PARTIAL_MARKER) for t in (spec.original, spec.replacement)
    ):
        return SkipReason.TOUCHES_MARKER, ""
    occurrences = record.answer.count(spec.original)
    if occurrences == 0:
        return SkipReason.MISSING, ""
    if occurrences > 1:
        return SkipReason.AMBIGUOUS, f"{occurrences} occurrences"
    if spec.cited:
        sentence = next((t for t in split_sentences(record.answer) if spec.original in t), None)
        if sentence is None:
            return SkipReason.NOT_IN_SENTENCE, ""
        if not set(spec.cited) <= set(citation_numbers(sentence)):
            return SkipReason.CITED_MISMATCH, f"cites {list(citation_numbers(sentence))}"
    return None


def apply_spec(record: AnswerRecord, spec: PerturbationSpec, k: int) -> AnswerRecord:
    """The perturbed copy of `record`: `original` replaced once, provenance in `meta`.

    Raises `ValueError` when `check_spec` would skip the spec (the caller validates first)."""
    problem = check_spec(spec, record)
    if problem is not None:
        raise ValueError(f"spec cannot be applied: {problem[0].value}")
    return replace(
        record,
        id=perturbed_id(record.id, k),
        answer=record.answer.replace(spec.original, spec.replacement, 1),
        meta={
            **record.meta,
            "source_id": record.id,
            "split": spec.split,
            "type": spec.type,
            "original": spec.original,
            "replacement": spec.replacement,
            "cited": list(spec.cited),
            "why": spec.why,
        },
    )


@dataclass(frozen=True, slots=True)
class PerturbedDump:
    originals: tuple[AnswerRecord, ...]
    perturbed: tuple[AnswerRecord, ...]
    skipped: tuple[SkippedSpec, ...]


def build_perturbed(
    records: Iterable[AnswerRecord],
    specs: Iterable[tuple[str, PerturbationSpec | MalformedSpec]],
    split: str | None = None,
) -> PerturbedDump:
    """Apply `(where, spec)` pairs to the source records.

    A record named by specs of two different splits is a `SplitLeakError` (checked over the
    whole input, before `split` filters): dev and test must not share a source answer.

    `k` counts the specs of a record over the WHOLE input in input order (malformed rows that
    carry a record id included, `split` filter NOT applied first), so neither fixing a skipped
    spec nor running one split changes the ids of the others. `split` then keeps only the specs
    of that split (a malformed row has none and is always reported). `originals` are the source
    records of the specs that were APPLIED (a skipped spec adds none), in source order and
    unchanged."""
    by_id = {r.id: r for r in records}
    counters: dict[str, int] = {}
    referenced: set[str] = set()
    split_of: dict[str, str] = {}
    perturbed: list[AnswerRecord] = []
    skipped: list[SkippedSpec] = []
    for where, spec in specs:
        if isinstance(spec, MalformedSpec):
            if spec.record_id is not None:
                counters[spec.record_id] = counters.get(spec.record_id, 0) + 1
            skipped.append(SkippedSpec(where, SkipReason.MALFORMED, spec.detail))
            continue
        if split_of.setdefault(spec.record_id, spec.split) != spec.split:
            raise SplitLeakError(
                f"{where}: record {spec.record_id!r} has specs in the splits "
                f"{split_of[spec.record_id]!r} and {spec.split!r}"
            )
        counters[spec.record_id] = k = counters.get(spec.record_id, 0) + 1
        if split is not None and spec.split != split:
            continue
        record = by_id.get(spec.record_id)
        problem = check_spec(spec, record)
        if problem is not None:
            skipped.append(SkippedSpec(where, *problem))
            continue
        assert record is not None
        referenced.add(record.id)
        perturbed.append(apply_spec(record, spec, k))
    originals = tuple(r for r in by_id.values() if r.id in referenced)
    return PerturbedDump(originals, tuple(perturbed), tuple(skipped))


def perturbation_labels(
    originals: Iterable[AnswerRecord],
    perturbed: Iterable[AnswerRecord],
    citation_truth: Collection[str] = (),
) -> list[LabelRow]:
    """The reference labels known by construction.

    originals: faithfulness true, and citation_support true ONLY for the ids in `citation_truth`
    (ids an independent labeling confirmed, with a matching hash: an original is faithful by
    selection, nothing says its citations are). Perturbed: faithfulness false, and
    citation_support false when the spec named cited sentences (`meta.cited`: the cited claim no
    longer holds). No relevance or refusal label: a perturbation says nothing about them."""
    rows = [
        LabelRow(r.id, JudgeName.FAITHFULNESS.value, True, PERTURBATION_LABELER, r.answer_sha256)
        for r in originals
    ]
    rows += [
        LabelRow(
            r.id, JudgeName.CITATION_SUPPORT.value, True, PERTURBATION_LABELER, r.answer_sha256
        )
        for r in originals
        if r.id in citation_truth
    ]
    for r in perturbed:
        rows.append(
            LabelRow(
                r.id, JudgeName.FAITHFULNESS.value, False, PERTURBATION_LABELER, r.answer_sha256
            )
        )
        if r.meta.get("cited"):
            rows.append(
                LabelRow(
                    r.id,
                    JudgeName.CITATION_SUPPORT.value,
                    False,
                    PERTURBATION_LABELER,
                    r.answer_sha256,
                )
            )
    return rows
