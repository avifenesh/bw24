#!/bin/bash
# Live-path correctness battery for the f8f4 MMA form swap (slice 5).
# Runs the CONTRIBUTING.md three gates in BOTH arms:
#   NAKED  -- regression control: the default int8 path must be byte-identical to pre-swap
#             (the swap only touches memra_mma_f8f4, so any NAKED move = collateral damage)
#   F8F4=1 -- in-config: the swapped tile is its own numeric config, so it gets its own
#             kernel-check / argmax / K=1..8 pass, per the SCOPE decision rule.
# Usage: battery.sh <model.gguf> <tag>
set -u
W=/home/avifenesh/projects/wt-w4a8
OUT=$W/research/w4a8-prefill-20260806/logs
MODEL=$1; TAG=$2
L=$OUT/battery-$TAG.log
: > "$L"
say(){ echo "$*" | tee -a "$L"; }
say "[start] $(date -Is)"
say "[commit] $(git -C $W rev-parse HEAD)"
say "[model] $MODEL"
say "[gpu] $(nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,clocks.sm,memory.used --format=csv,noheader)"
say "[apps] $(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | tr '\n' ';')"
for ARM in NAKED F8F4; do
  case $ARM in
    NAKED) E=(MEMRA_W4A8_BATTERY_ARM=naked) ;;
    F8F4)  E=(MEMRA_MMQ_F8F4=1) ;;
  esac
  say ""; say "################ ARM $ARM ################"

  say ""; say "===== [$ARM] GATE 1: kernel-check ====="
  env "${E[@]}" timeout 1800 "$W/target/release/kernel-check" >> "$L" 2>&1
  say "[rc=$?]"

  say ""; say "===== [$ARM] GATE 2: run-gen argmax (prefill vs decode + batched-prime) ====="
  env "${E[@]}" MEMRA_PROMPT_FILE=$W/research/e2e/prompts/pp512.txt \
    timeout 1800 "$W/target/release/run-gen" "$MODEL" >> "$L" 2>&1
  say "[rc=$?]"

  say ""; say "===== [$ARM] GATE 3: run-spec K=1..8 self-consistency ====="
  env "${E[@]}" timeout 3600 "$W/target/release/run-spec" "$MODEL" >> "$L" 2>&1
  say "[rc=$?]"
done
say ""; say "[end] $(date -Is)"
echo "wrote $L"
