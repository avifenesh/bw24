#!/usr/bin/env bash
# 8-turn ACCUMULATIVE agent-loop acceptance protocol (MTP-heal battery, §item 3).
# Recreated in-repo from the ephemeral /tmp/w4a4-loop.sh structure (that path does not survive).
#
# Each turn appends a new user instruction to a GROWING transcript, runs one spec-decode call whose
# prompt is the whole transcript so far, appends the generated reply, and records that turn's MTP
# acceptance. The point is self-conditioning: acceptance under an accumulating, model-authored
# context (the real agent-loop regime), not on fixed prompts. One JSONL row per turn (prompt id
# "agentloop-tN", the same schema acceptance_battery.sh emits) so the delta table can compare the
# bf16 ceiling vs the NVFP4 hit turn-by-turn.
#
# Usage:
#   tools/agent_loop_acceptance.sh <model_path> <out.jsonl> <arm_label> [extra run-spec env ...]
# Env knobs (with battery-matched defaults):
#   K=3  NGEN=256  FULL_PREC=0  SEED_PROMPT=research/e2e/prompts/p2-code-medium.txt
#   RUNSPEC=./target/release/run-spec  TIMEOUT=900  PYBIN=python3
# Examples:
#   FULL_PREC=1 tools/agent_loop_acceptance.sh /data/ai-ml/hf-models/qwen35-9b-hf out.jsonl bf16-fullprec
#   tools/agent_loop_acceptance.sh /data/.../Qwen3.5-9B-NVFP4-MTP-GGUF.gguf out.jsonl nvfp4
set -euo pipefail

MODEL="${1:?usage: agent_loop_acceptance.sh <model> <out.jsonl> <arm> [extra env...]}"
OUT="${2:?missing out.jsonl}"
ARM="${3:?missing arm label}"
shift 3 || true
EXTRA_ENV=("$@")   # e.g. BW24_RP=0 BW24_MMQ=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

K="${K:-3}"
NGEN="${NGEN:-256}"
FULL_PREC="${FULL_PREC:-0}"
SEED_PROMPT="${SEED_PROMPT:-research/e2e/prompts/p2-code-medium.txt}"
RUNSPEC="${RUNSPEC:-./target/release/run-spec}"
TIMEOUT="${TIMEOUT:-900}"
PYBIN="${PYBIN:-python3}"
PARSE="$ROOT/tools/acceptance_parse.py"

TURNS=(
 "Review this code and list its three biggest problems."
 "Fix the first problem you listed. Show the full corrected function."
 "Now add proper error handling to it."
 "Write unit tests for the corrected function."
 "One test you wrote is flaky under concurrency. Find it and fix it."
 "Refactor the module to separate IO from logic."
 "Document the public API of the refactored module."
 "Summarize every change made across this session as a changelog."
)

FP_FLAG=""; [ "$FULL_PREC" = "1" ] && FP_FLAG="--full-prec"
W="$(mktemp -t agentloop.XXXXXX)"
trap 'rm -f "$W"' EXIT
cat "$SEED_PROMPT" > "$W"

echo "[agent-loop] arm=$ARM model=$MODEL K=$K ngen=$NGEN full_prec=$FULL_PREC turns=${#TURNS[@]}"
for i in "${!TURNS[@]}"; do
  turn=$((i + 1))
  printf '\n\n### USER TURN %d: %s\n### ASSISTANT:\n' "$turn" "${TURNS[$i]}" >> "$W"
  OUTLOG="$(env "${EXTRA_ENV[@]}" \
      ${FULL_PREC:+BW24_FULL_PREC=$FULL_PREC} \
      BW24_NGEN="$NGEN" BW24_SPEC_K="$K" BW24_SPEC_STATS=1 BW24_SPEC_HPOST=1 \
      BW24_PRINT_TEXT=1 BW24_PROMPT_FILE="$W" \
      timeout "$TIMEOUT" "$RUNSPEC" "$MODEL" 2>&1 || true)"
  # append the model's reply to the growing transcript (self-conditioning)
  awk '/^--- generated text ---$/{f=1;next} /^--- end ---$/{f=0} f' <<<"$OUTLOG" >> "$W"
  printf '%s\n' "$OUTLOG" | "$PYBIN" "$PARSE" --out "$OUT" --arm "$ARM" \
      --prompt "agentloop-t$turn" --k "$K" --run 1 --model "$MODEL" --ngen "$NGEN" $FP_FLAG \
      --extra "agentloop ctx_chars=$(wc -c <"$W")"
  ACC="$(printf '%s\n' "$OUTLOG" | grep -oE 'acceptance:[^%]*%' | head -1 || true)"
  echo "[agent-loop] $ARM turn$turn ${ACC:-<no acceptance>}  ctx_chars=$(wc -c <"$W")"
done
echo "[agent-loop] DONE arm=$ARM -> $OUT"
