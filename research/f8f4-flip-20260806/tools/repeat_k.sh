#!/bin/bash
# REPRODUCIBILITY of a quoted acceptance cell: N reps of one K, interleaved OFF/ON.
#
# Greedy spec acceptance should be run-to-run deterministic (fixed prompt, fixed K, greedy verify),
# so this is both a repeat measurement AND a determinism check: if a cell's acceptance moves
# between identical reps, the delta being quoted is not a measurement of the arm.
# The spec tok/s alongside it is NOT deterministic and gets a median over N.
#
# Usage: repeat_k.sh <tag> <model> <prompt> <K> <N> [extra env words...]
set -u
W=/home/avifenesh/projects/wt-f8f4flip
OUT=$W/research/f8f4-flip-20260806/logs
TAG=$1; MODEL=$2; PROMPT=$3; K=$4; N=$5; shift 5
EXTRA=("$@")
NGEN=${NGEN:-128}
L=$OUT/repeat-$TAG.log
{ echo "[start] $(date -Is)"; echo "[commit] $(git -C $W rev-parse HEAD)"
  echo "[model] $MODEL"; echo "[prompt] $PROMPT"; echo "[k] $K"; echo "[n] $N"; echo "[ngen] $NGEN"
  echo "[extra] ${EXTRA[*]:-<none>}"
  echo "[proto] $N reps, interleaved OFF/ON per rep, one run-spec per (rep,arm)"
} > "$L"
for rep in $(seq 1 "$N"); do
  for ARM in OFF ON; do
    case $ARM in
      OFF) AENV=() ;;
      ON)  AENV=(MEMRA_MMQ_F8F4=1) ;;
    esac
    echo "=== K=$K ARM=$ARM rep=$rep $(nvidia-smi --query-gpu=clocks.sm,temperature.gpu,utilization.gpu --format=csv,noheader)" >> "$L"
    env "${AENV[@]}" "${EXTRA[@]}" MEMRA_NGEN=$NGEN MEMRA_SPEC_K=$K MEMRA_SPEC_STATS=1 \
      MEMRA_PROMPT_FILE="$PROMPT" \
      timeout 2400 "$W/target/release/run-spec" "$MODEL" >> "$L" 2>&1
    echo "[rc=$?] K=$K ARM=$ARM rep=$rep done" >> "$L"
  done
done
echo "wrote $L"
