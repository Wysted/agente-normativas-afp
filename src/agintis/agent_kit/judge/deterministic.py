"""Judges that need no model."""

from agintis.agent_kit import source_text
from agintis.agent_kit.judge.records import AnswerRecord
from agintis.agent_kit.judge.refusal import is_refusal, refused_by_scope
from agintis.agent_kit.judge.results import JudgeName, JudgeResult


def _verdict(expected: str, refused: bool) -> bool:
    return refused if expected == "out" else not refused


class RefusalCorrectnessJudge:
    """Was refusing (or not) the right call for the question's expected scope?

    - expected `out` (the corpus cannot answer it): must be refused or answered "I don't know";
    - expected `in`: must NOT be a blanket refusal;
    - no expected scope in the dump: not applicable.

    "Refused" has up to three signals. Always: the agent's refuse path (scope `out` or a
    `refusal_reason`) and the documented text heuristic of `refusal.py`. When the panel ran the
    LLM judges (faithfulness / relevance) it also passes `model_refusal`, what the judge model
    said about `is_refusal`, and the refusal is the OR of the three: the heuristic is
    deliberately narrow (near-domain refusals such as "los documentos no mencionan ese plazo"
    slip through it) while the model reads any wording. The method is then `hybrid`; without
    the model signal (a `--judges` subset without them) it stays `deterministic` and uses only
    the first two. The verdict the HEURISTIC alone would give always stays in
    `details.heuristic_verdict`, and the aggregate reports it next to the hybrid one, so the
    contribution of the model is visible.
    """

    name = JudgeName.REFUSAL_CORRECTNESS
    method = "deterministic"  # "hybrid" when a model signal is passed in

    async def judge(self, record: AnswerRecord) -> JudgeResult:
        return await self.judge_with(record, None)

    async def judge_with(self, record: AnswerRecord, model_refusal: bool | None) -> JudgeResult:
        """`model_refusal`: the judge model's `is_refusal` as `panel.model_refusal_signal` picks it
        (answer_relevance's, else faithfulness'), None when no LLM judge ran or none reported."""
        if record.expected_scope is None:
            return JudgeResult.not_applicable(
                self.name, "the question has no expected scope", "no_expected_scope"
            )
        by_scope = refused_by_scope(record)
        by_text = is_refusal(record)  # scope OR heuristic
        refused = by_text or bool(model_refusal)
        method = "deterministic" if model_refusal is None else "hybrid"
        how = (
            "refuse path (scope 'out' or refusal reason)"
            if by_scope
            else "text heuristic"
            if by_text
            else "judge model"
        )
        details = {
            "expected_scope": record.expected_scope,
            "refused": refused,
            "scope": None if record.scope is None else record.scope.level,
            "heuristic_verdict": _verdict(record.expected_scope, by_text),
            "model_is_refusal": model_refusal,
        }
        verdict = _verdict(record.expected_scope, refused)
        if record.expected_scope == "out":
            rationale = (
                f"out-of-scope question refused ({how})"
                if refused
                else "out-of-scope question answered instead of refused"
            )
        else:
            rationale = (
                "in-scope question answered" if verdict else f"in-scope question refused ({how})"
            )
        return JudgeResult.decided(self.name, verdict, rationale, method, details)


class ScriptMixJudge:
    """Does the answer keep to the Latin script? A word with a letter of another script that the
    question and the sources do not contain (a model slipping devanagari / bengali / telugu into
    Spanish text) fails. Deterministic: see `source_text.foreign_script_tokens`."""

    name = JudgeName.SCRIPT_MIX
    method = "deterministic"

    async def judge(self, record: AnswerRecord) -> JudgeResult:
        if not record.answer.strip():
            return JudgeResult.not_applicable(self.name, "the answer is empty", "empty_answer")
        tokens = source_text.foreign_script_tokens(
            record.answer, f"{record.question}\n{record.rendered_sources}"
        )
        rationale = (
            f"foreign-script words: {', '.join(tokens)}" if tokens else "no foreign-script words"
        )
        return JudgeResult.decided(
            self.name,
            not tokens,
            rationale,
            self.method,
            {"tokens": tokens},
            code="foreign_script" if tokens else None,
        )
