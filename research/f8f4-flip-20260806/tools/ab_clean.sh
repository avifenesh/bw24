#!/bin/bash
# Clean interleaved pp512 A/B with the w4a8 lane's PER-ARM IDLE GUARD (that lane's lesson: a
# locked-clock A/B needs a contention guard, not just a lock file). Adapted from
# research/w4a8-prefill-20260806/tools/ab_clean.sh for this worktree; arms renamed OFF/ON to
# match this lane's tables.
#
# Records util/clocks/temp per run so a contaminated round is identifiable after the fact. On
# this laptop 5090 the locked 1860 is a CEILING, not a floor — sustained prefill power-caps into
# the 1550-1750 range, so the per-run clock column is load-bearing for reading the deltas.
#
# Usage: ab_clean.sh <tag> <model> <rounds> [extra env words...]
set -u
W=/home/avifenesh/projects/wt-f8f4flip
OUT=$W/research/f8f4-flip-20260806/logs
TAG=$1; MODEL=$2; ROUNDS=$3; shift 3
EXTRA=("$@")
REPS=${MEMRA_PP_REPS:-5}
PROMPT=${PROMPT:-$W/research/e2e/prompts/pp512.txt}
L=$OUT/ab-$TAG.log
gpustate(){ nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,power.draw,clocks.sm,memory.used --format=csv,noheader; }
waitidle(){
  for i in $(seq 1 90); do
    u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
    [ "$u" -lt 15 ] && [ "$n" -le 1 ] && return 0
    sleep 20
  done
  return 1
}
{ echo "[start] $(date -Is)"; echo "[commit] $(git -C $W rev-parse HEAD)"
  echo "[model] $MODEL"; echo "[bin] $W/target/release/run-gen"; echo "[prompt] $PROMPT"
  echo "[extra] ${EXTRA[*]:-<none>}"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | sed 's/^/[apps] /'
  echo "[proto] interleaved OFF/ON, $ROUNDS rounds x MEMRA_PP_REPS=$REPS, locked 1860, idle-guard per arm"
} > "$L"
for r in $(seq 1 "$ROUNDS"); do
  for ARM in OFF ON; do
    waitidle || { echo "=== round $r arm $ARM SKIPPED: box never went idle" >> "$L"; continue; }
    echo "=== round $r arm $ARM $(gpustate)" >> "$L"
    case $ARM in
      OFF) AENV=() ;;
      ON)  AENV=(MEMRA_MMQ_F8F4=1) ;;
    esac
    env "${AENV[@]}" "${EXTRA[@]}" MEMRA_PROMPT_FILE="$PROMPT" MEMRA_PP_ONLY=1 MEMRA_PP_REPS=$REPS \
      timeout 1800 "$W/target/release/run-gen" "$MODEL" >> "$L" 2>&1
    echo "[rc=$?] [post $(gpustate)]" >> "$L"
  done
done
echo "wrote $L"
