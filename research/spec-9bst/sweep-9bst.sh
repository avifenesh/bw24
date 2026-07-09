#!/usr/bin/env bash
# SPEC SWEEP for qwen35-9b-nvfp4-st-modelopt (AxionML modelopt NVFP4 safetensors).
# Measures K in {2,3,4} x pmin {0.15, 0.3} coarse grid, prompts p1/p2/p3, N=2 per cell.
# Output: JSONL rows appended to $OUT.
#
# GPU DISCIPLINE: checks for foreign >2GB processes before every model load.
# After coarse sweep, runs: trim arm at winner, PMIN0 arm per acceptance law.
set -euo pipefail

MODEL=/data/ai-ml/hf-models/qwen35-9b-nvfp4-st-modelopt
OUT=${1:-/home/avifenesh/projects/bw24-9bspec/research/spec-9bst/sweep-9bst.jsonl}
NGEN=${BW24_NGEN:-256}
N=${N:-2}  # reps per cell (sweep); final uses N=3

PROMPTS_DIR=/home/avifenesh/projects/bw24/research/e2e/prompts
P1=$PROMPTS_DIR/p1-code-short.txt
P2=$PROMPTS_DIR/p2-code-medium.txt
P3=$PROMPTS_DIR/p3-agentic-long.txt

BIN=/home/avifenesh/projects/bw24/target/release/run-spec

# --- GPU guard: wait for foreign process ---
gpu_wait() {
  local attempt=0
  while [ $attempt -lt 40 ]; do
    local foreign=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null \
      | awk -F, '$2+0 > 2000' | grep -v "$$" || true)
    if [ -z "$foreign" ]; then
      return 0
    fi
    echo "[gpu_wait] foreign GPU process >2GB detected (attempt $((attempt+1))/40), waiting 180s..."
    sleep 180
    attempt=$((attempt+1))
  done
  echo "[gpu_wait] TIMEOUT after 40 attempts (2h). Aborting."
  exit 1
}

# --- run one spec arm ---
run_arm() {
  local k=$1 pmin=$2 prompt_file=$3 prompt_label=$4 run_n=$5 extra_env="${6:-}"
  local env_str="BW24_NGEN=$NGEN BW24_SPEC_K=$k BW24_SPEC_PMIN=$pmin BW24_SPEC_STATS=1 BW24_PROMPT_FILE=$prompt_file"
  if [ -n "$extra_env" ]; then
    env_str="$env_str $extra_env"
  fi
  echo "[sweep] K=$k pmin=$pmin prompt=$prompt_label run=$run_n extra=$extra_env"
  local arm_label="K${k}_pmin${pmin}"
  if [ -n "$extra_env" ]; then
    arm_label="${arm_label}_$(echo $extra_env | tr ' =' '_')"
  fi

  eval $env_str "$BIN" "$MODEL" 2>&1 | python3 /home/avifenesh/projects/bw24/tools/acceptance_parse.py \
    --out "$OUT" \
    --arm "$arm_label" \
    --prompt "$prompt_label" \
    --k "$k" \
    --run "$run_n" \
    --model "qwen35-9b-nvfp4-st-modelopt" \
    --ngen "$NGEN" \
    --extra "sweep-9bst"
}

echo "=== SPEC SWEEP: 9B ST modelopt (K={2,3,4} x pmin={0.15,0.3}) ==="
echo "=== Model: $MODEL ==="
echo "=== NGEN=$NGEN, N=$N per cell ==="
echo "=== Output: $OUT ==="
date

gpu_wait

# COARSE SWEEP: K in {2,3,4} x pmin in {0.15, 0.3}
for k in 2 3 4; do
  for pmin in 0.15 0.3; do
    for prompt_label in p1 p2 p3; do
      case $prompt_label in
        p1) pf=$P1 ;;
        p2) pf=$P2 ;;
        p3) pf=$P3 ;;
      esac
      for run in $(seq 1 $N); do
        run_arm "$k" "$pmin" "$pf" "$prompt_label" "$run"
      done
    done
  done
done

echo ""
echo "=== COARSE SWEEP COMPLETE ==="
echo "=== Rows in $OUT ==="
date
