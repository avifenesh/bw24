#!/bin/bash
# Per-model run-gen argmax A/B: OFF (int8 W4A8 default) vs ON (MEMRA_MMQ_F8F4=1).
#
# Gate 1 of the flip bar. Two things are checked per arm:
#   (a) run-gen's OWN argmax MATCH lines (prefill-vs-decode and batched-prime-vs-tokenwise) —
#       an internal self-consistency gate that must hold WITHIN each arm.
#   (b) the generated `tokens:` line compared BETWEEN arms — the cross-arm greedy identity the
#       flip bar actually asks about (f8f4's e4m3 acts are a different numeric class than int8,
#       so this is NOT bit-exact by construction; it is the thing being measured).
#
# Usage: argmax_ab.sh <tag> <model> [extra env words...]
set -u
W=/home/avifenesh/projects/wt-f8f4flip
OUT=$W/research/f8f4-flip-20260806/logs
TAG=$1; MODEL=$2; shift 2
EXTRA=("$@")
NGEN=${NGEN:-24}
# PROMPT: a text file fed via MEMRA_PROMPT_FILE (safe across families, tokenized by the model's
# own tokenizer). IDS: a raw token-ids file fed as POSITIONAL ARGS instead — the shape
# tools/fast-gate/fast-gate.sh uses for the family-pinned gemma probes. Set exactly one; feeding
# the shared text probe to a gemma model is valid tokenization but is NOT the pinned probe, and
# the gemma depth/chat ids are what its goldens were minted under.
PROMPT=${PROMPT:-}
IDS=${IDS:-}
[ -z "$PROMPT" ] && [ -z "$IDS" ] && PROMPT=$W/tools/fast-gate/prompts/probe.txt
ARGS=()
PENV=()
if [ -n "$IDS" ]; then
  # shellcheck disable=SC2207
  ARGS=($(cat "$IDS"))
else
  PENV=(MEMRA_PROMPT_FILE="$PROMPT")
fi
L=$OUT/argmax-$TAG.log
{ echo "[start] $(date -Is)"; echo "[commit] $(git -C $W rev-parse HEAD)"
  echo "[model] $MODEL"; echo "[ngen] $NGEN"
  echo "[prompt] ${PROMPT:-<ids file: $IDS>}"
  echo "[extra] ${EXTRA[*]:-<none>}"
  nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader | sed 's/^/[gpu] /'
} > "$L"
for ARM in OFF ON; do
  case $ARM in
    OFF) AENV=() ;;
    ON)  AENV=(MEMRA_MMQ_F8F4=1) ;;
  esac
  echo "=== ARM $ARM $(nvidia-smi --query-gpu=clocks.sm,temperature.gpu,utilization.gpu --format=csv,noheader)" >> "$L"
  env "${AENV[@]}" "${EXTRA[@]}" "${PENV[@]}" MEMRA_NGEN=$NGEN \
    timeout 1800 "$W/target/release/run-gen" "$MODEL" "${ARGS[@]}" >> "$L" 2>&1
  echo "[rc=$?] ARM $ARM done" >> "$L"
done
echo "wrote $L"
