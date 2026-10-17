"""The judge port: one binary quality judge over one answer record."""

from typing import Protocol, runtime_checkable

from agintis.agent_kit.judge.records import AnswerRecord
from agintis.agent_kit.judge.results import JudgeName, JudgeResult


class AnswerJudge(Protocol):
    """Decides ONE binary quality question about an answer.

    Implementations return `not_applicable` when the question does not apply to the record, a
    verdict with a short rationale otherwise, and may raise on provider failures (the caller,
    `judge_record`, turns any exception into an `error` result so one record never aborts a run).
    """

    @property
    def name(self) -> JudgeName: ...

    @property
    def method(self) -> str:
        """ "llm" or "deterministic": how this judge reaches a verdict."""
        ...

    async def judge(self, record: AnswerRecord) -> JudgeResult: ...


@runtime_checkable
class SignalAwareJudge(AnswerJudge, Protocol):
    """A judge that also takes what the judge model said about refusals (`refusal_correctness`).

    `model_refusal` is True / False when the LLM judges reported `is_refusal`, None when none did.
    """

    async def judge_with(self, record: AnswerRecord, model_refusal: bool | None) -> JudgeResult: ...
