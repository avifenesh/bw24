#!/bin/bash
# GATE 2 — the decisive one: spec ACCEPTANCE A/B, run-spec K=1..8, OFF vs ON, INTERLEAVED per K.
#
# Why interleaved per K rather than arm-blocked: acceptance is a ratio (accepted/drafted) and is
# clock-insensitive, but the tok/s printed alongside it is not, and a co-resident lane drifting
# mid-battery would otherwise land entirely on one arm. Interleaving per K keeps any drift
# symmetric across arms.
#
# Each invocation also IS the K=1..8 self-consistency gate (run-spec asserts greedy identity to
# plain generate and exits non-zero on FAIL) — so this harness produces gate 2 of the correctness
# bar and the acceptance telemetry in the same runs.
#
# Usage: accept_ab.sh <tag> <model> <prompt-file> [extra env words...]
set -u
W=/home/avifenesh/projects/wt-f8f4flip
OUT=$W/research/f8f4-flip-20260806/logs
TAG=$1; MODEL=$2; PROMPT=$3; shift 3
EXTRA=("$@")
NGEN=${NGEN:-128}
KS=${KS:-"1 2 3 4 5 6 7 8"}
L=$OUT/accept-$TAG.log
{ echo "[start] $(date -Is)"; echo "[commit] $(git -C $W rev-parse HEAD)"
  echo "[model] $MODEL"; echo "[prompt] $PROMPT"; echo "[ngen] $NGEN"; echo "[ks] $KS"
  echo "[extra] ${EXTRA[*]:-<none>}"
  echo "[proto] interleaved per K: OFF then ON, one run-spec invocation per (arm,K)"
  nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader | sed 's/^/[gpu] /'
} > "$L"
for k in $KS; do
  for ARM in OFF ON; do
    case $ARM in
      OFF) AENV=() ;;
      ON)  AENV=(MEMRA_MMQ_F8F4=1) ;;
    esac
    echo "=== K=$k ARM=$ARM $(nvidia-smi --query-gpu=clocks.sm,temperature.gpu,utilization.gpu --format=csv,noheader)" >> "$L"
    env "${AENV[@]}" "${EXTRA[@]}" MEMRA_NGEN=$NGEN MEMRA_SPEC_K=$k MEMRA_SPEC_STATS=1 \
      MEMRA_PROMPT_FILE="$PROMPT" \
      timeout 2400 "$W/target/release/run-spec" "$MODEL" >> "$L" 2>&1
    echo "[rc=$?] K=$k ARM=$ARM done" >> "$L"
  done
done
echo "wrote $L"
