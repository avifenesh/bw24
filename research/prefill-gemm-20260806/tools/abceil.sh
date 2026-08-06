#!/bin/bash
set -u
W=/home/avifenesh/projects/wt-prefill
OUT=$W/research/prefill-gemm-20260806/logs
MODEL=$1; TAG=$2; ROUNDS=$3
gpustate(){ nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm,memory.used --format=csv,noheader; }
L=$OUT/ab-foldceiling-$TAG.log
{ echo "[start] $(date -Is)"; echo "[gpu-pre] $(gpustate)";
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | sed 's/^/[apps] /';
  echo "[model] $MODEL"; echo "[proto] interleaved BASE/CEIL, $ROUNDS rounds, MEMRA_PP_REPS=5 each, clocks locked"; } > "$L"
for r in $(seq 1 $ROUNDS); do
  for ARM in BASE CEIL; do
    echo "=== round $r arm $ARM $(gpustate)" >> "$L"
    env MEMRA_PROMPT_FILE=$W/research/e2e/prompts/pp512.txt MEMRA_PP_ONLY=1 MEMRA_PP_REPS=5 \
      timeout 900 /tmp/run-gen-$ARM "$MODEL" >> "$L" 2>&1
    echo "[rc=$?]" >> "$L"
  done
done
echo "[gpu-post] $(gpustate)" >> "$L"
echo "wrote $L"
