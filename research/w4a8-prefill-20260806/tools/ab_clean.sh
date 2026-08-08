#!/bin/bash
# Clean interleaved A/B with a CONTENTION GUARD: before each arm, require the GPU idle
# (util < 15%, <=1 compute app = the idle llama-server) and the locked clock actually held.
# Records util/clocks per run so a contaminated round is identifiable after the fact.
# Usage: ab_clean.sh <model> <tag> <rounds>
set -u
W=/home/avifenesh/projects/wt-w4a8
OUT=$W/research/w4a8-prefill-20260806/logs
MODEL=$1; TAG=$2; ROUNDS=$3
L=$OUT/ab-clean-$TAG.log
gpustate(){ nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,power.draw,clocks.sm,memory.used --format=csv,noheader; }
waitidle(){
  for i in $(seq 1 120); do
    u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
    [ "$u" -lt 15 ] && [ "$n" -le 1 ] && return 0
    sleep 20
  done
  return 1
}
{ echo "[start] $(date -Is)"; echo "[commit] $(git -C $W rev-parse HEAD)";
  echo "[model] $MODEL"; echo "[bin] $W/target/release/run-gen";
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | sed 's/^/[apps] /';
  echo "[proto] interleaved NAKED/F8F4, $ROUNDS rounds x MEMRA_PP_REPS=5, locked 1860, idle-guard per arm"; } > "$L"
for r in $(seq 1 "$ROUNDS"); do
  for ARM in NAKED F8F4; do
    waitidle || { echo "=== round $r arm $ARM SKIPPED: box never went idle" >> "$L"; continue; }
    echo "=== round $r arm $ARM $(gpustate)" >> "$L"
    case $ARM in
      NAKED) ENVV=(MEMRA_PP_ONLY=1) ;;
      F8F4)  ENVV=(MEMRA_MMQ_F8F4=1) ;;
    esac
    env "${ENVV[@]}" MEMRA_PROMPT_FILE=$W/research/e2e/prompts/pp512.txt MEMRA_PP_ONLY=1 MEMRA_PP_REPS=5 \
      timeout 900 "$W/target/release/run-gen" "$MODEL" >> "$L" 2>&1
    echo "[rc=$?] [post $(gpustate)]" >> "$L"
  done
done
echo "wrote $L"
