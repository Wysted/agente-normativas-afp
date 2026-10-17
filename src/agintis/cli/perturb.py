"""Build a controlled-perturbation answers dump to meta-evaluate the answer judges.

    uv run python -m agintis.cli.perturb --answers answers.jsonl --specs dev.jsonl test.jsonl \\
        --out perturbed.jsonl --labels-out labels.jsonl [--split dev|test] \\
        [--no-include-originals] [--citation-truth consensus.jsonl]

A known-faithful original answer must PASS the faithfulness judge and the same answer with ONE
injected error must FAIL it; both labels are known by construction (see
`agintis.agent_kit.judge.perturb`). `--specs` are JSON Lines written by an external generator
(ideally another model family than the judge's):

    {"record_id", "split": "dev|test", "type": "negation|condition_inversion|
     condition_deletion|quantifier|number_date|entity_swap|unsupported_addition", "original",
     "replacement", "cited": [n...], "why"}

`original` must occur EXACTLY ONCE in the `answer` of the `--answers` record `record_id`.
`--split` keeps only the specs of that split. `--out` receives, in source order, the originals of
every record the (filtered) specs reference, unchanged (`--no-include-originals` leaves them
out), then one record per spec with id `<record_id>~p<k>` (k = 1-based order of that record's
specs over the WHOLE input, malformed rows with a record id included and `--split` not applied,
so ids never depend on the filter): the answer with `original` -> `replacement` applied once,
the same question, expected scope and sources, `answer_sha256` recomputed, and the provenance
(`source_id`, `split`, `type`, `original`, `replacement`, `cited`, `why`) in `meta`.

`--labels-out` receives the labels (labeler `perturbation`) for `python -m agintis.cli.agreement`:
faithfulness true for the originals, false for the perturbed records, and citation_support false
for a perturbed record whose spec has `cited` sentences. Every row carries the record's
`answer_sha256`. `--citation-truth LABELS.jsonl` (label rows `{id, judge, label, answer_sha256}`,
e.g. a consensus written by `cli.agreement --consensus-out`) adds citation_support TRUE for the
originals that file confirms: every row of that id for `citation_support` must be true and carry
this original's hash (an original is faithful by selection, nothing says its citations are).
Labels are written before the dump. Run `cli.judge` over `--out`, then:

    uv run python -m agintis.cli.agreement --labels labels.jsonl --judgments judgments.json \\
        --by-meta type --answers perturbed.jsonl

for the sensitivity (originals passing) and specificity (perturbed answers failing) of each judge
and the confusion per perturbation type.

A record named by specs of two different splits is a usage error (dev and test would share an
answer, checked over all the specs before `--split` filters).

A spec whose `original` is missing or occurs more than once, with an unknown record_id or type,
with `original == replacement`, malformed, touching a `[n]` marker (or half of one), over a
record that is a failed run, has no sources or was refused by the scope check, or (with
`cited`) whose span does not lie in ONE sentence citing those numbers, is reported on stderr
and skipped; the valid ones are still written. Both files are written to `PATH.partial` (mode
0600: they hold retrieved text and model output) and renamed when complete. A summary
(counts per split and type, the skipped specs) goes to stderr.

Exit codes: 0 ok; 1 some specs were skipped (the output is complete for the valid ones); 2 usage
(unreadable or malformed answers, unreadable specs, no specs, output path); 3 abort.
"""

import argparse
import json
import logging
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path

from agintis.agent_kit.judge import (
    PERTURBATION_TYPES,
    SPLITS,
    AnswerRecord,
    JudgeName,
    LabelRow,
    MalformedSpec,
    PerturbationSpec,
    PerturbedDump,
    SkippedSpec,
    SpecFormatError,
    SplitLeakError,
    build_perturbed,
    perturbation_labels,
    spec_from_json,
)
from agintis.cli._common import (
    EXIT_ABORTED,
    EXIT_OK,
    EXIT_USAGE,
    UsageError,
    check_output_path,
    read_jsonl,
    read_lines,
    write_private_lines,
)
from agintis.cli.dump_answers import write_dump
from agintis.cli.judge import load_dump

logger = logging.getLogger("agintis.cli.perturb")

EXIT_SKIPPED = 1


def _record_id_of(row: object) -> str | None:
    value = row.get("record_id") if isinstance(row, dict) else None
    return value if isinstance(value, str) and value else None


def load_specs(paths: Iterable[Path]) -> list[tuple[str, PerturbationSpec | MalformedSpec]]:
    """`(where, spec)` pairs of the spec files in input order (blank lines skipped). A malformed
    row becomes a `MalformedSpec` (skipped with a reason, but it keeps its numbering slot)."""
    specs: list[tuple[str, PerturbationSpec | MalformedSpec]] = []
    for path in paths:
        for where, line in read_lines(path):
            row: object = None
            try:
                row = json.loads(line)
                specs.append((where, spec_from_json(row)))
            except (ValueError, SpecFormatError) as exc:
                specs.append((where, MalformedSpec(_record_id_of(row), str(exc))))
    if not specs:
        raise UsageError("the spec files contain no specs")
    return specs


def load_citation_truth(path: Path, originals: Iterable[AnswerRecord]) -> set[str]:
    """Ids of the `originals` that `path` confirms as citation_support true: every
    citation_support row of that id is true and carries the original's `answer_sha256`. Rows
    without a hash cannot be bound to an answer and confirm nothing."""
    votes: dict[str, list[tuple[bool, str | None]]] = {}
    for where, row in read_jsonl(path):
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise UsageError(f"{where}: a label row needs a string 'id'")
        if row.get("judge") != JudgeName.CITATION_SUPPORT.value:
            continue
        if not isinstance(row.get("label"), bool):
            raise UsageError(f"{where}: 'label' must be true or false")
        sha = row.get("answer_sha256")
        votes.setdefault(row["id"], []).append(
            (row["label"], sha if isinstance(sha, str) else None)
        )
    return {
        r.id
        for r in originals
        if votes.get(r.id) and all(label and sha == r.answer_sha256 for label, sha in votes[r.id])
    }


def write_labels(path: Path, rows: Sequence[LabelRow]) -> None:
    write_private_lines(
        path,
        (
            json.dumps(
                {
                    "id": r.id,
                    "judge": r.judge,
                    "label": r.label,
                    "labeler": r.labeler,
                    "answer_sha256": r.answer_sha256,
                }
            )
            for r in rows
        ),
    )


def _summary(dump: PerturbedDump, skipped: Sequence[SkippedSpec]) -> str:
    counts = Counter((r.meta["split"], r.meta["type"]) for r in dump.perturbed)
    lines = [
        f"perturbed {len(dump.perturbed)} answers from {len(dump.originals)} originals; "
        f"skipped {len(skipped)} specs"
    ]
    for split in SPLITS:
        total = sum(n for (s, _), n in counts.items() if s == split)
        if total:
            lines.append(f"  {split}: {total}")
            lines.extend(
                f"    {t:<22} {counts[(split, t)]}"
                for t in PERTURBATION_TYPES
                if counts[(split, t)]
            )
    lines.extend(f"  skipped {s}" for s in skipped)
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m agintis.cli.perturb",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--answers", type=Path, required=True, help="the source dump JSONL")
    parser.add_argument("--specs", type=Path, nargs="+", required=True, help="spec JSONL files")
    parser.add_argument("--split", choices=SPLITS, help="only the specs of this split")
    parser.add_argument("--out", type=Path, required=True, help="the perturbed dump JSONL")
    parser.add_argument(
        "--labels-out", type=Path, required=True, help="labels JSONL for cli.agreement"
    )
    parser.add_argument(
        "--citation-truth",
        type=Path,
        help="label rows confirming citation_support true for originals (needs answer_sha256)",
    )
    parser.add_argument(
        "--include-originals",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also write the originals the specs reference (default: on)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    try:
        check_output_path(args.out, "--out")
        check_output_path(args.labels_out, "--labels-out")
        if args.out.resolve() == args.labels_out.resolve():
            raise UsageError("--out and --labels-out must be different files")
        records = load_dump(args.answers)
        specs = load_specs(args.specs)
        try:
            dump = build_perturbed(records, specs, args.split)
        except SplitLeakError as exc:
            raise UsageError(str(exc)) from exc
        skipped = list(dump.skipped)
        if not dump.perturbed and not skipped:
            raise UsageError(f"no specs for split {args.split!r}")
        kept_originals = dump.originals if args.include_originals else ()
        truth = (
            set()
            if args.citation_truth is None
            else load_citation_truth(args.citation_truth, kept_originals)
        )
        # labels first: a dump without its labels is the half that cannot be audited
        write_labels(args.labels_out, perturbation_labels(kept_originals, dump.perturbed, truth))
        write_dump(args.out, [*kept_originals, *dump.perturbed])
        print(_summary(dump, skipped), file=sys.stderr)
        if args.citation_truth is not None:
            print(f"citation_support true for {len(truth)} originals", file=sys.stderr)
    except UsageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except Exception as exc:
        print(f"aborted: {type(exc).__name__}: {exc}", file=sys.stderr)
        logger.debug("abort details", exc_info=True)
        return EXIT_ABORTED
    return EXIT_SKIPPED if skipped else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
