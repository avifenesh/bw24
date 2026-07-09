#!/usr/bin/env bash
# FINAL BOARD MEASUREMENTS for 9B ST modelopt checkpoint.
# Requires: GPU free. Run AFTER the sweep identifies the winning config.
# Produces: N=3 plain + spec + PP_ONLY rows for the board head-to-head.
set -euo pipefail

MODEL=/data/ai-ml/hf-models/qwen35-9b-nvfp4-st-modelopt
NGEN=256
N=3  # final config
OUT=/home/avifenesh/projects/bw24-9bspec/research/spec-9bst/final-9bst.jsonl
PROMPTS_DIR=/home/avifenesh/projects/bw24/research/e2e/prompts
P1=$PROMPTS_DIR/p1-code-short.txt
P2=$PROMPTS_DIR/p2-code-medium.txt
P3=$PROMPTS_DIR/p3-agentic-long.txt
PP=$PROMPTS_DIR/pp2048.txt

BIN_SPEC=/home/avifenesh/projects/bw24/target/release/run-spec
BIN_GEN=/home/avifenesh/projects/bw24/target/release/run-gen

# Winning config from sweep (EDIT after sweep):
BEST_K=${BEST_K:-3}
BEST_PMIN=${BEST_PMIN:-0.3}
TRIM=${TRIM:-/tmp/frspec-9bst-modelopt-32768.gguf}
USE_PMIN0=${USE_PMIN0:-0}  # 1 if base acceptance < 75%

# GPU guard
gpu_wait() {
  local attempt=0
  while [ $attempt -lt 40 ]; do
    local foreign=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null \
      | awk -F, '$2+0 > 2000' | grep -v "$$" || true)
    if [ -z "$foreign" ]; then return 0; fi
    echo "[gpu_wait] foreign GPU >2GB (attempt $((attempt+1))/40), waiting 180s..."
    sleep 180
    attempt=$((attempt+1))
  done
  echo "[gpu_wait] TIMEOUT"; exit 1
}

echo "=== FINAL BOARD: 9B ST modelopt ==="
echo "=== K=$BEST_K pmin=$BEST_PMIN trim=$TRIM PMIN0=$USE_PMIN0 ==="
echo "=== N=$N, NGEN=$NGEN ==="
date

gpu_wait

# --- 1. PLAIN DECODE (tg128 for board) ---
echo ""
echo "--- PLAIN DECODE tg128 ---"
for run in $(seq 1 $N); do
  for prompt_label in p1 p2 p3; do
    case $prompt_label in
      p1) pf=$P1 ;; p2) pf=$P2 ;; p3) pf=$P3 ;;
    esac
    echo "[plain] $prompt_label run $run"
    BW24_NGEN=128 BW24_PROMPT_FILE="$pf" BW24_GEN_ONLY=1 \
      "$BIN_SPEC" "$MODEL" 2>&1 | tee -a /tmp/9bst-plain-$prompt_label-$run.log
  done
done

# --- 2. SPEC at best config (N=3, p1/p2/p3) ---
echo ""
echo "--- SPEC K=$BEST_K pmin=$BEST_PMIN trim=$TRIM ---"
SPEC_ENV="BW24_NGEN=$NGEN BW24_SPEC_K=$BEST_K BW24_SPEC_PMIN=$BEST_PMIN BW24_SPEC_STATS=1"
if [ -n "$TRIM" ] && [ -f "$TRIM" ]; then
  SPEC_ENV="$SPEC_ENV BW24_FRSPEC_TRIM=$TRIM"
fi
if [ "$USE_PMIN0" = "1" ]; then
  SPEC_ENV="$SPEC_ENV BW24_SPEC_PMIN0=1"
fi

for run in $(seq 1 $N); do
  for prompt_label in p1 p2 p3; do
    case $prompt_label in
      p1) pf=$P1 ;; p2) pf=$P2 ;; p3) pf=$P3 ;;
    esac
    echo "[spec] $prompt_label run $run"
    eval "$SPEC_ENV BW24_PROMPT_FILE=$pf" "$BIN_SPEC" "$MODEL" 2>&1 | \
      python3 /home/avifenesh/projects/bw24/tools/acceptance_parse.py \
        --out "$OUT" \
        --arm "final_K${BEST_K}_pmin${BEST_PMIN}" \
        --prompt "$prompt_label" \
        --k "$BEST_K" \
        --run "$run" \
        --model "qwen35-9b-nvfp4-st-modelopt" \
        --ngen "$NGEN" \
        --extra "final-board"
  done
done

# --- 3. RUN-SPEC self-consistency PASS (K=1..8) ---
echo ""
echo "--- RUN-SPEC SELF-CONSISTENCY K=1..8 ---"
BW24_NGEN=64 BW24_PROMPT_FILE="$P2" "$BIN_SPEC" "$MODEL" 2>&1 | tee /tmp/9bst-runspec-gate.log
echo "--- gate result above ---"

# --- 4. PP_ONLY pp1845 (board row) ---
echo ""
echo "--- PP_ONLY pp1845 (N=$N) ---"
BW24_PP_ONLY=1 BW24_PP_REPS=$N BW24_PP_WARMUP=1 BW24_PROMPT_FILE="$P2" \
  "$BIN_GEN" "$MODEL" 2>&1 | tee /tmp/9bst-pp1845.log

echo ""
echo "=== FINAL BOARD COMPLETE ==="
date
