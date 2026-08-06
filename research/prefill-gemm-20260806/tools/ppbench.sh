#!/bin/bash
# pp512 interleaved A/B harness. Usage: ppbench.sh <label> <model> <reps> [env...]
set -u
LABEL=$1; MODEL=$2; REPS=$3; shift 3
OUT=/home/avifenesh/projects/wt-prefill/research/prefill-gemm-20260806/logs
W=/home/avifenesh/projects/wt-prefill
gpustate() { nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used --format=csv,noheader; }
L=$OUT/pp-$LABEL.log
{ echo "[gpu-pre] $(gpustate)"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | sed 's/^/[apps] /'; } > "$L"
env "$@" MEMRA_PROMPT_FILE=$W/research/e2e/prompts/pp512.txt MEMRA_PP_ONLY=1 MEMRA_PP_REPS=$REPS \
  timeout 1800 $W/target/release/run-gen "$MODEL" >> "$L" 2>&1
rc=$?
{ echo "[gpu-post] $(gpustate)"; echo "rc=$rc"; } >> "$L"
grep -E "pp-only" "$L" | tail -$((REPS+1))
