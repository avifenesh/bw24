#!/usr/bin/env bash
# W4A4 two-arm greedy-decode comparison gate over the reject corpus.
#
# Each cell runs BOTH activation contracts in one process against one set of loaded weights, so a
# divergence is attributable to the activation quantizer rather than to load order or clock drift.
# MEMRA_RP=0 is mandatory: an rp weight always takes the W4A8 tile, so with the repack on the W4A4
# arm would never engage and the gate would report a false PASS.
#
# usage: run-gate.sh <label> [residual_k]   (writes logs/<label>-gate.log, one JSONL line per cell)
#
# residual_k (default 0) is passed as MEMRA_MMQ_RESIDUAL_K. It touches ONLY the W4A4 arm — the
# reference arm never enters the W4A4 nvfp4 launcher — so the reference stream stays the same
# denominator across the whole k sweep.
set -uo pipefail

LANE=/home/avifenesh/projects/wt-w4a4
GATE=$LANE/target/release/w4a4-gate
LOGDIR=$LANE/research/w4a4-rescue-20260803/logs
LABEL=${1:?usage: run-gate.sh <label> [residual_k]}
RK=${2:-0}
LOG=$LOGDIR/$LABEL-gate.log

Q9=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
Q27=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
PROMPTS=$LANE/research/e2e/prompts

mkdir -p "$LOGDIR"
: > "$LOG"

# CELLS is overridable so the corpus can be WIDENED without editing the script (a k that holds on the
# three original cells has not been shown to hold generally — this path was rejected twice before, so
# the untested prompts are exactly where a premature default flip would break).
CELLS=${CELLS:-"p2-code-medium p3-agentic-long p4-16k"}

for cell in $CELLS; do
  for m in q9 q27; do
    case $m in
      q9)  MODEL=$Q9 ;;
      q27) MODEL=$Q27 ;;
    esac
    echo "########## $m / $cell ##########" | tee -a "$LOG"
    echo "MEMRA_MMQ_RESIDUAL_K=$RK" >> "$LOG"
    # The GPU is shared with another lane: take the lock per cell and release between cells so a
    # long corpus never starves the neighbour.
    flock /tmp/gpu5090.lock \
      env MEMRA_RP=0 MEMRA_MMQ_RESIDUAL_K="$RK" "$GATE" "$MODEL" "$PROMPTS/$cell.txt" 48 >> "$LOG" 2>&1
    echo "(exit $?)" | tee -a "$LOG"
  done
done

echo "=== verdicts ===" | tee -a "$LOG"
grep -h "^VERDICT" "$LOG" | tee -a /dev/stderr >/dev/null
grep -c "DIVERGENT" "$LOG" >/dev/null
