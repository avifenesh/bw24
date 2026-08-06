#!/bin/bash
# Interleaved W4A8(default) vs W4A4(mxf4nvf4 door) pp512 A/B, locked clocks.
# Both arms are the SAME binary — the door is a runtime dispatch flag, not a build.
set -u
W=/home/avifenesh/projects/wt-prefill2
OUT=$W/research/prefill-ilp-20260806/logs
MODEL=$1; TAG=$2; ROUNDS=$3
gpustate(){ nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm,memory.used --format=csv,noheader; }
L=$OUT/ab-w4a4door-$TAG.log
{ echo "[start] $(date -Is)"; echo "[gpu-pre] $(gpustate)";
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | sed 's/^/[apps] /';
  echo "[model] $MODEL"; echo "[bin] $W/target/release/run-gen";
  echo "[proto] interleaved NAKED/W4A8rp0/W4A4RAW/W4A4K32, $ROUNDS rounds x MEMRA_PP_REPS=5, clocks locked 1860"; } > "$L"
for r in $(seq 1 $ROUNDS); do
  for ARM in NAKED W4A8 W4A4RAW W4A4K32; do
    echo "=== round $r arm $ARM $(gpustate)" >> "$L"
    case $ARM in
      NAKED)   ENVV=(MEMRA_PP_ONLY=1) ;;
      W4A8)    ENVV=(MEMRA_RP=0) ;;
      W4A4RAW) ENVV=(MEMRA_RP=0 MEMRA_MMQ=1 MEMRA_MMQ_RESIDUAL_K=0) ;;
      W4A4K32) ENVV=(MEMRA_RP=0 MEMRA_MMQ=1 MEMRA_MMQ_RESIDUAL_K=32) ;;
    esac
    env "${ENVV[@]}" MEMRA_PROMPT_FILE=$W/research/e2e/prompts/pp512.txt MEMRA_PP_ONLY=1 MEMRA_PP_REPS=5 \
      timeout 900 $W/target/release/run-gen "$MODEL" >> "$L" 2>&1
    echo "[rc=$?]" >> "$L"
  done
done
echo "[gpu-post] $(gpustate)" >> "$L"
echo "wrote $L"
