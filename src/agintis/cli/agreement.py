"""Agreement of the judges with reference labels: Cohen's kappa and confusion matrices.

    uv run python -m agintis.cli.agreement --labels labels.jsonl --judgments judgments.json \\
        [--consensus unanimous|majority] [--out agreement.json] [--consensus-out consensus.jsonl] \\
        [--by-meta KEY --answers dump.jsonl]

`--labels`: JSON Lines `{"id", "judge", "label": true|false[, "labeler"][, "answer_sha256"]}` with
the reference verdict of a judge on a record (same ids and judge names as the `cli.judge` report
given as `--judgments`). Several labelers may label the same items through `labeler` (a row
without it belongs to the labeler `default`). `answer_sha256` is copied by the labeler from the
dump record: a row whose hash differs from the judged record's is EXCLUDED with a warning (the
answer or its sources changed after it was labeled) and listed under `excluded.stale`. Rows
without a hash, or a judgments report whose records carry none (an old `cli.judge`), cannot be
bound to an answer: that is an error unless `--allow-unbound` (then the rows are kept and counted
as `unbound`). Per judge it reports:

- the CONSENSUS labels in both modes (`by_mode`; `--consensus`, default unanimous, picks the one
  repeated at the top level of the entry and written by `--consensus-out`): `unanimous` = every
  labeler of that judge labeled the item and they all agree (one labeler = its own consensus),
  `majority` = strictly more than half of them agree (an even-count tie is left out); the items
  left out are listed (`incomplete_items`, `disagreed_items`);
- the inter-labeler Cohen's kappa for every pair of labelers and, with 3+ labelers, Fleiss' kappa
  (the quality of the reference);
- the judge against the consensus: n, agreement %, Cohen's kappa, Gwet's AC1, PABAK, positive and
  negative agreement and the confusion matrix
  (rows = reference, columns = judge: tp both true, fp reference false / judge true, fn reference
  true / judge false, tn both false). Items the judge did not decide (`not_applicable`, `error`)
  or that are not in the report are skipped and listed.

`--by-meta KEY` (needs `--answers`, the dump the judges ran on) adds `by_group` to every comparison:
the same numbers per value of `meta[KEY]` of the dump records (records without it go to `(none)`).
With the labels and the perturbed dump of `cli.perturb`, `--by-meta type` shows which error types
a judge misses: the confusion of each perturbation type (all reference-false, so `specificity` is
the share of those errors the judge catches) next to the originals under `(none)` (all
reference-true, `sensitivity` is the share it accepts). `sensitivity` = tp / (tp + fn),
`specificity` = tn / (tn + fp) and `pass_rate` = (tp + fp) / n are in every comparison.

Kappa is `null` when undefined (no items, or a rater used one category throughout). Reading it:
the usual benchmarks (Landis and Koch) call 0.61-0.80 substantial and above 0.80 almost perfect
agreement; with few items the number is noisy, read it next to n.

`--answers` is cross-checked against the judgments report (same record ids, same `answer_sha256`
where the report has one): a dump the judges did not run on is a usage error.

`--consensus-out` writes the consensus label set as `{id, judge, label, answer_sha256}` rows (the
hash of the judged answer, so the file can bind labels, e.g. `cli.perturb --citation-truth`).
`--consensus-out` is refused together with `--allow-unbound` (it would stamp a hash on labels
that never carried one).
Exit codes: 0 ok; 2 usage (unreadable or malformed labels / report, unwritable output); 3 abort.
"""

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from agintis.agent_kit.judge import (
    CONSENSUS_MODES,
    AnswerRecord,
    ConsensusMode,
    LabelFormatError,
    LabelRow,
    agreement_report,
    bind_labels,
    consensus_rows,
    hashes_from_report,
    parse_label_rows,
    verdicts_from_report,
)
from agintis.cli._common import (
    EXIT_ABORTED,
    EXIT_OK,
    EXIT_USAGE,
    UsageError,
    check_output_path,
    read_jsonl,
)
from agintis.cli.judge import load_dump

logger = logging.getLogger("agintis.cli.agreement")


def load_labels(path: Path) -> list[LabelRow]:
    try:
        labels = parse_label_rows(read_jsonl(path))
    except LabelFormatError as exc:
        raise UsageError(str(exc)) from exc
    if not labels:
        raise UsageError(f"{path}: no labels")
    return labels


def load_report(path: Path) -> dict[str, Any]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise UsageError(f"{path}: cannot read the judgments report ({exc})") from exc
    if not isinstance(report, dict):
        raise UsageError(f"{path}: the judgments report must be a JSON object")
    return report


def build_report(
    labels: Sequence[LabelRow],
    judgments: dict[str, Any],
    mode: ConsensusMode = "unanimous",
    allow_unbound: bool = False,
    groups: dict[str, str] | None = None,
    by_meta: str | None = None,
) -> dict[str, Any]:
    try:
        verdicts = verdicts_from_report(judgments)
        hashes = hashes_from_report(judgments)
    except LabelFormatError as exc:
        raise UsageError(str(exc)) from exc
    if not allow_unbound:
        unhashed = sorted(set(verdicts) - set(hashes))
        if unhashed:
            raise UsageError(
                f"{len(unhashed)} records of the judgments report carry no answer_sha256 (an "
                "older cli.judge?): labels cannot be bound to the judged answers; rerun the "
                "judge or pass --allow-unbound"
            )
        unbound = sum(1 for row in labels if row.answer_sha256 is None)
        if unbound:
            raise UsageError(
                f"{unbound} label rows carry no answer_sha256: copy it from the dump record, or "
                "pass --allow-unbound to accept labels that cannot be checked"
            )
    known = {name for entry in verdicts.values() for name in entry}
    unknown = sorted({row.judge for row in labels} - known)
    if unknown:
        raise UsageError(f"labels for judges the report does not have: {unknown}")
    report = agreement_report(labels, verdicts, hashes, mode, groups)
    return {
        "meta": {
            "labels": len(labels),
            "consensus_mode": mode,
            "by_meta": by_meta,
            "judge_model": judgments.get("meta", {}).get("judge_model"),
            "judge_version": judgments.get("meta", {}).get("judge_version"),
            "judge_prompts_sha256": judgments.get("meta", {}).get("judge_prompts_sha256"),
            "answers_sha256": judgments.get("meta", {}).get("answers_sha256"),
        },
        **report,
    }


def check_answers_match(records: Sequence[AnswerRecord], judgments: dict[str, Any]) -> None:
    """`UsageError` unless the dump is the one the report was made on (ids and hashes)."""
    try:
        reported = set(verdicts_from_report(judgments))
        hashes = hashes_from_report(judgments)
    except LabelFormatError as exc:
        raise UsageError(str(exc)) from exc
    dumped = {r.id: r.answer_sha256 for r in records}
    problems = []
    if missing := sorted(set(dumped) - reported):
        problems.append(f"{len(missing)} dump records not in the report (e.g. {missing[:3]})")
    if extra := sorted(reported - set(dumped)):
        problems.append(f"{len(extra)} report records not in the dump (e.g. {extra[:3]})")
    if changed := sorted(i for i, sha in dumped.items() if i in hashes and hashes[i] != sha):
        problems.append(f"{len(changed)} records whose answer_sha256 differs (e.g. {changed[:3]})")
    if problems:
        raise UsageError("--answers is not the dump the judgments ran on: " + "; ".join(problems))


def groups_of(records: Sequence[AnswerRecord], key: str) -> dict[str, str]:
    groups: dict[str, str] = {}
    for record in records:
        value = record.meta.get(key)
        if isinstance(value, str | int | float) and not isinstance(value, bool):
            groups[record.id] = str(value)
    return groups


def kept_labels(labels: Sequence[LabelRow], judgments: dict[str, Any]) -> list[LabelRow]:
    """The label rows that still describe the judged answers (stale hashes dropped)."""
    return list(bind_labels(labels, hashes_from_report(judgments)).kept)


def _fmt(value: float | None, percent: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1%}" if percent else f"{value:.3f}"


def _print(report: dict[str, Any]) -> None:
    excluded = report["excluded"]
    if excluded["stale"]:
        print(
            f"warning: {len(excluded['stale'])} label rows EXCLUDED: their answer_sha256 no longer "
            "matches the judged record (relabel them)",
            file=sys.stderr,
        )
    if excluded["unbound"]:
        print(
            f"warning: {excluded['unbound']} label rows carry no answer_sha256 (not checkable)",
            file=sys.stderr,
        )
    for judge, entry in report["judges"].items():
        cmp = entry["judge_vs_consensus"]
        c = cmp["confusion"]
        print(f"\n{judge} (consensus: {entry['consensus_mode']})")
        modes = ", ".join(f"{m} {v['consensus_items']}" for m, v in entry["by_mode"].items())
        print(
            f"  labelers: {', '.join(entry['labelers'])}; labeled items {entry['labeled_items']}; "
            f"consensus items {modes} (incomplete {len(entry['incomplete_items'])}, "
            f"disagreed {len(entry['disagreed_items'])})"
        )
        for pair in entry["inter_labeler"] or []:
            print(
                f"  inter-labeler {pair['labelers'][0]} vs {pair['labelers'][1]}: n={pair['n']} "
                f"agreement {_fmt(pair['agreement'], True)} kappa {_fmt(pair['kappa'])}"
            )
        if entry["fleiss"]:
            print(
                f"  Fleiss kappa over {len(entry['fleiss']['labelers'])} labelers: "
                f"n={entry['fleiss']['n']} kappa {_fmt(entry['fleiss']['kappa'])}"
            )
        print(
            f"  judge vs consensus: n={cmp['n']} agreement {_fmt(cmp['agreement'], True)} "
            f"kappa {_fmt(cmp['kappa'])} AC1 {_fmt(cmp['gwet_ac1'])} PABAK {_fmt(cmp['pabak'])} "
            f"pos {_fmt(cmp['positive_agreement'])} neg {_fmt(cmp['negative_agreement'])}"
        )
        if cmp["small_sample"]:
            print(f"  note: n={cmp['n']} is a small sample, read the numbers with care")
        print("                judge=true  judge=false")
        print(f"  ref=true      {c['tp']:>10}  {c['fn']:>11}")
        print(f"  ref=false     {c['fp']:>10}  {c['tn']:>11}")
        if entry["by_group"]:
            print(f"  by meta {report['meta']['by_meta']!r}:")
            for group, g in entry["by_group"].items():
                print(
                    f"    {group:<24} n={g['n']:>3} pass {_fmt(g['pass_rate'], True)} "
                    f"sens {_fmt(g['sensitivity'], True)} spec {_fmt(g['specificity'], True)} "
                    f"(tp {g['confusion']['tp']} fp {g['confusion']['fp']} "
                    f"fn {g['confusion']['fn']} tn {g['confusion']['tn']})"
                )
        skipped = entry["skipped"]
        if skipped["no_judgment"] or skipped["not_decided"]:
            print(
                f"  skipped: {len(skipped['no_judgment'])} not in the report, "
                f"{len(skipped['not_decided'])} not decided by the judge"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m agintis.cli.agreement",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--labels", type=Path, required=True, help="reference labels JSONL")
    parser.add_argument("--judgments", type=Path, required=True, help="the cli.judge JSON report")
    parser.add_argument(
        "--allow-unbound",
        action="store_true",
        help="accept label rows / a report without answer_sha256 (cannot be bound to the answers)",
    )
    parser.add_argument(
        "--consensus",
        choices=CONSENSUS_MODES,
        default="unanimous",
        help="consensus mode of the headline numbers and --consensus-out (default: %(default)s)",
    )
    parser.add_argument(
        "--by-meta", metavar="KEY", help="also break the numbers down by meta[KEY] of --answers"
    )
    parser.add_argument("--answers", type=Path, help="the dump the judges ran on (for --by-meta)")
    parser.add_argument("--out", type=Path, help="write the agreement JSON here")
    parser.add_argument("--consensus-out", type=Path, help="write the consensus labels JSONL here")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    try:
        check_output_path(args.out, "--out")
        check_output_path(args.consensus_out, "--consensus-out")
        if args.consensus_out is not None and args.allow_unbound:
            raise UsageError(
                "--consensus-out cannot be combined with --allow-unbound: the consensus file "
                "stamps the judged answers' hashes, which would vouch for labels that were never "
                "bound to them"
            )
        if (args.by_meta is None) != (args.answers is None):
            raise UsageError("--by-meta and --answers go together")
        labels = load_labels(args.labels)
        judgments = load_report(args.judgments)
        groups = None
        if args.by_meta is not None:
            dump = load_dump(args.answers)
            check_answers_match(dump, judgments)
            groups = groups_of(dump, args.by_meta)
        report = build_report(
            labels, judgments, args.consensus, args.allow_unbound, groups, args.by_meta
        )
        _print(report)
        if args.out is not None:
            args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        if args.consensus_out is not None:
            rows = consensus_rows(
                kept_labels(labels, judgments), args.consensus, hashes_from_report(judgments)
            )
            lines = [json.dumps(r, ensure_ascii=False) + "\n" for r in rows]
            args.consensus_out.write_text("".join(lines), encoding="utf-8")
    except UsageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except Exception as exc:
        print(f"aborted: {type(exc).__name__}: {exc}", file=sys.stderr)
        logger.debug("abort details", exc_info=True)
        return EXIT_ABORTED
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
