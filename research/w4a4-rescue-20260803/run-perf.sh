#!/usr/bin/env bash
# Prefill throughput: the default W4A8 arm vs the W4A4 MMQ arm, with and without the rank-k
# residual correction. Prices the exactness work — a correction that eats the speedup makes the
# whole W4A4 chase pointless, so this has to be measured before any default-flip argument.
#
# Protocol (research/benchmarks.md): arms INTERLEAVED inside one window, N=5 rounds, one engine on
# the GPU at a time (flock, taken per run and released between so the neighbour lane is not starved),
# medians reported with N stated. Cross-run and cross-day comparisons are clock-drift-invalid, which
# is exactly why the arms alternate rather than run in blocks.
#
# usage: run-perf.sh <label> [rounds]
set -uo pipefail

LANE=/home/avifenesh/projects/wt-w4a4
BIN=$LANE/target/release/run-gen
LOGDIR=$LANE/research/w4a4-rescue-20260803/logs
LABEL=${1:?usage: run-perf.sh <label> [rounds] [prompt-file]}
ROUNDS=${2:-5}
LOG=$LOGDIR/$LABEL-perf.log

MODEL=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
# Prompt is selectable: the correction's cost is per-token (y traffic) while the weight re-reads are
# per-CTA-pass, so the corrected arm's RATIO against W4A8 need not be constant in prompt length. A
# single-length window would not support a default-flip claim.
PROMPT=${3:-$LANE/research/e2e/prompts/p2-code-medium.txt}

mkdir -p "$LOGDIR"
: > "$LOG"

# The gate corpus prompt, verbatim — same text the exactness cells run, so the perf window and the
# exactness window describe the same workload.
PROMPT_TEXT=$(cat "$PROMPT")

# MEMRA_NGEN=1: this window prices PREFILL. Decode is m=1 and never reaches the prefill GEMM, so
# generating tokens would only add unrelated variance to the window.
run_arm() {
  local tag=$1; shift
  echo "=== ARM $tag round $R ===" >> "$LOG"
  flock /tmp/gpu5090.lock \
    env MEMRA_NGEN=1 MEMRA_PROMPT="$PROMPT_TEXT" "$@" "$BIN" "$MODEL" >> "$LOG" 2>&1
  echo "(exit $?)" >> "$LOG"
}

for R in $(seq 1 "$ROUNDS"); do
  # W4A8 = the shipped default (MEMRA_MMQ unset). MEMRA_RP=0 on EVERY arm: rp changes the weight
  # layout and would otherwise be a second variable between the default and the W4A4 arms.
  run_arm w4a8      MEMRA_RP=0
  run_arm w4a4-k0   MEMRA_RP=0 MEMRA_MMQ=1 MEMRA_MMQ_RESIDUAL_K=0
  run_arm w4a4-k16  MEMRA_RP=0 MEMRA_MMQ=1 MEMRA_MMQ_RESIDUAL_K=16
  # k=32 is the exactness-landing point (IDENTICAL on all five measurable gate cells), so it is the
  # arm any default-flip argument has to be priced on.
  run_arm w4a4-k32  MEMRA_RP=0 MEMRA_MMQ=1 MEMRA_MMQ_RESIDUAL_K=32
done

echo "raw -> $LOG"
