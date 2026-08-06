#!/bin/bash
# pp512 interleaved harness. Usage: ppbench.sh <label> <model> <reps> <binary> [env...]
set -u
W=/home/avifenesh/projects/wt-prefill2
OUT=$W/research/prefill-ilp-20260806/logs
LABEL=$1; MODEL=$2; REPS=$3; BIN=$4; shift 4
gpustate(){ nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used --format=csv,noheader; }
L=$OUT/pp-$LABEL.log
{ echo "[start] $(date -Is)"; echo "[gpu-pre] $(gpustate)"; echo "[bin] $BIN"; echo "[model] $MODEL";
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | sed 's/^/[apps] /'; } > "$L"
env "$@" MEMRA_PROMPT_FILE=$W/research/e2e/prompts/pp512.txt MEMRA_PP_ONLY=1 MEMRA_PP_REPS=$REPS \
  timeout 1800 "$BIN" "$MODEL" >> "$L" 2>&1
rc=$?
{ echo "[gpu-post] $(gpustate)"; echo "rc=$rc"; } >> "$L"
grep -E "pp-only" "$L" | tail -$((REPS+1))
