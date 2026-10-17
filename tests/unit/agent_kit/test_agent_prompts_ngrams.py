"""The agents' own prompts (defaults and every case's YAML) must not leak calibration questions.

Only the versioned calibration files are read; the held-out `eval_questions.jsonl` is never
opened. Compared against the QUESTIONS, not the dumps (a dump echoes the agent's own messages).
"""

import json
from pathlib import Path

import pytest
import yaml

from agintis.agent_kit.nodes import prompts
from tests.unit.agent_kit.judge.test_guard_prompts import grams

CASE_ROOT = Path(__file__).resolve().parents[3] / "case"
HELD_OUT = "eval_questions.jsonl"


def test_no_agent_prompt_shares_a_5_gram_with_the_calibration_questions() -> None:
    files = [f for f in CASE_ROOT.glob("*/calibration/*.jsonl") if f.name != HELD_OUT]
    if not files:
        pytest.skip("no case calibration files in this checkout")
    questions = [
        json.loads(line)["question"]
        for f in files
        for line in f.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    texts = [
        v for k, v in vars(prompts).items() if k.startswith("DEFAULT_") and k.endswith("_PROMPT")
    ]
    for spec_file in CASE_ROOT.glob("*/agents/*.yaml"):
        spec = yaml.safe_load(spec_file.read_text(encoding="utf-8")) or {}
        texts += [v for v in (spec.get("prompts") or {}).values() if isinstance(v, str)]

    shared = grams("\n".join(texts)) & grams("\n".join(questions))

    assert not shared, f"agent prompts overlap calibration questions: {sorted(shared)[:5]}"
