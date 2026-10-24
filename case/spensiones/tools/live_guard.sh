#!/usr/bin/env bash
# Live guard for spensiones-rag: dump answers (and judge in_dev/ood/oficio_dev) with keys loaded from .env.
# usage: live_guard.sh <run name> [set ...]   (default sets: in_dev ood)
# Sets: any case/spensiones/calibration/<set>.jsonl (in_dev ood afiliacion_dev cotizaciones_dev compendio_new_dev nota_dev oficio_dev bank_dev).
# A set whose answers file already exists is not dumped again (delete it to redo it).
# LG_AGENTS=<dir> runs another agents directory (default case/spensiones/agents).
# LG_JUDGE=0 skips the judge (dump only).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."   # repo root
run=${1:?usage: live_guard.sh <run name> [set ...]}; shift
if [ $# -eq 0 ]; then sets=(in_dev ood); else sets=("$@"); fi
out=case/spensiones/judge/e2e/$run; logs=case/spensiones/judge/logs/$run   # gitignored (case/*/judge/)
mkdir -p "$out" "$logs"
export LLM_PROFILES_FILE=case/spensiones/llm_profiles.yaml OBSERVABILITY_TRACING=none \
  RERANKER_URL=http://localhost:8083 RERANKER_MAX_CANDIDATES=40
# An exported but empty key would win over .env (uv keeps existing variables): drop it.
for v in OPENAI_API_KEY DEEPSEEK_API_KEY; do [ -n "${!v:-}" ] || unset "$v"; done
# Named openai profiles read OPENAI_API_KEY; this .env names the same key LLM_API_KEY.
py() {
  uv run --env-file .env sh -c 'export OPENAI_API_KEY="${OPENAI_API_KEY:-${LLM_API_KEY:-}}"; exec python -m "$@"' py "$@"
}

for s in "${sets[@]}"; do
  answers=$out/${s}_answers.jsonl
  if [ ! -f "$answers" ]; then
    py agintis.cli.dump_answers --profiles case/spensiones/profiles --agents "${LG_AGENTS:-case/spensiones/agents}" \
      --agent spensiones-rag --tenant spensiones --questions case/spensiones/calibration/$s.jsonl \
      --out "$answers" --concurrency 4 2>&1 | tee "$logs/dump_$s.log"
    # A reranker fallback silently keeps the vector order: such a dump is not a valid measurement.
    if grep -q "Reranker unavailable" "$logs/dump_$s.log"; then
      mv "$answers" "$answers.degraded"
      echo "INVALID: $s fell back to vector order ($(grep -c 'Reranker unavailable' "$logs/dump_$s.log")x); moved to $answers.degraded" >&2
      exit 1
    fi
  fi
  [ "${LG_JUDGE:-1}" = 0 ] && continue
  case $s in in_dev | ood | oficio_dev)
    # judge exit 1 = some judgments errored; the report is still complete
    py agintis.cli.judge --profile judge --concurrency 4 --answers "$answers" \
      --report "$out/${s}_report.json" 2>&1 | tee "$logs/judge_$s.log" || true ;;
  esac
done
echo "done: $out"
