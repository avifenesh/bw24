#!/usr/bin/env bash
# Full correctness battery for the W4A4 + residual-k32 arm, per CONTRIBUTING.md:
#   kernel-check ALL GREEN | run-gen prefill-vs-decode argmax MATCH | run-spec K=1..8 self-consistency
#
# Every cell runs with MEMRA_MMQ=1 MEMRA_RP=0 MEMRA_MMQ_RESIDUAL_K=32 — the arm a default flip would
# ship. The W4A8 default is already gated by the repo's own battery, so this window is about whether
# the W4A4 arm holds the same gates, not about re-proving the default.
#
# flock per invocation, released between, so the neighbour lane on the 5090 is not starved.
set -uo pipefail

LANE=/home/avifenesh/projects/wt-w4a4
LOG=$LANE/research/w4a4-rescue-20260803/logs/battery-k32.log
Q9=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
Q27=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
ARM="MEMRA_MMQ=1 MEMRA_RP=0 MEMRA_MMQ_RESIDUAL_K=32"

: > "$LOG"
run() {
  echo "########## $1 ##########" >> "$LOG"
  shift
  flock /tmp/gpu5090.lock env MEMRA_MMQ=1 MEMRA_RP=0 MEMRA_MMQ_RESIDUAL_K=32 "$@" >> "$LOG" 2>&1
  echo "(exit $?)" >> "$LOG"
}

echo "arm: $ARM" >> "$LOG"
run "kernel-check" "$LANE/target/release/kernel-check"
# run-gen: prefill argmax vs decode argmax, plus the batched-prime line on >=16-token prompts.
run "run-gen q9"  "$LANE/target/release/run-gen" "$Q9"
run "run-gen q27" "$LANE/target/release/run-gen" "$Q27"
# run-spec: every draft depth K=1..8 must stay token-identical to plain decode.
run "run-spec q9  K=1..8" env MEMRA_SPEC_K=all "$LANE/target/release/run-spec" "$Q9"
run "run-spec q27 K=1..8" env MEMRA_SPEC_K=all "$LANE/target/release/run-spec" "$Q27"

echo "raw -> $LOG"
