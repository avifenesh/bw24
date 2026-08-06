#!/bin/bash
# Interleaved NAKED (int8-act W4A8, m16n8k16.s8) vs F8F4 (e4m3-act W4A8, kind::f8f6f4 m16n8k32)
# pp512 A/B at locked clocks. Both arms are the SAME binary — the route is a runtime dispatch flag.
# Usage: ab_f8f4.sh <model.gguf> <tag> <rounds>
set -u
W=/home/avifenesh/projects/wt-w4a8
OUT=$W/research/w4a8-prefill-20260806/logs
MODEL=$1; TAG=$2; ROUNDS=$3
gpustate(){ nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used --format=csv,noheader; }
L=$OUT/ab-f8f4-$TAG.log
{ echo "[start] $(date -Is)"; echo "[gpu-pre] $(gpustate)";
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | sed 's/^/[apps] /';
  echo "[model] $MODEL"; echo "[bin] $W/target/release/run-gen";
  echo "[commit] $(git -C $W rev-parse HEAD)";
  echo "[proto] interleaved NAKED/F8F4, $ROUNDS rounds x MEMRA_PP_REPS=5, clocks locked 1860, flock /tmp/gpu5090.lock"; } > "$L"
for r in $(seq 1 "$ROUNDS"); do
  for ARM in NAKED F8F4; do
    echo "=== round $r arm $ARM $(gpustate)" >> "$L"
    case $ARM in
      NAKED) ENVV=(MEMRA_PP_ONLY=1) ;;
      F8F4)  ENVV=(MEMRA_MMQ_F8F4=1) ;;
    esac
    env "${ENVV[@]}" MEMRA_PROMPT_FILE=$W/research/e2e/prompts/pp512.txt MEMRA_PP_ONLY=1 MEMRA_PP_REPS=5 \
      timeout 900 "$W/target/release/run-gen" "$MODEL" >> "$L" 2>&1
    echo "[rc=$?]" >> "$L"
  done
done
echo "[gpu-post] $(gpustate)" >> "$L"
echo "wrote $L"
