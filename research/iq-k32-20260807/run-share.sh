#!/bin/bash
# iq-k32 lane, e2e-share gate: what fraction of GPU kernel time do the mmq_iq kernels hold?
# Capture shapes per model:
#   bal : prompt ~2048 tok, NGEN=128  (balanced shape — realistic e2e share)
#   pp  : prompt ~4096 tok, NGEN=16   (prefill-heavy — UPPER BOUND on the share; the kernels
#         are prefill-only so no workload can exceed this)
# nsys binaries go to /tmp/iqk32-nsys (NEVER committed); the kern_sum CSVs + console logs
# land in raw/ and are committed. Capture window = MEMRA_PROFILE_GEN=1 (prime + decode of
# the timed generate only — load and the argmax-gate loop excluded, the 2026-07-10 law).
#
# LESSON (first attempt, 2026-08-07 05:54Z): `nsys -c cudaProfilerApi` DEFAULTS to
# --capture-range-end=stop-shutdown — it TERMINATES the app right after cudaProfilerStop(),
# racing the final "generated N tokens" println (one run read rc=143, all runs lost the
# decode receipt line). The kernel sums looked complete but the run receipts were not.
# Fixed: explicit --capture-range-end=stop, app runs to completion, rc + tok/s lines real.
set -u
W=/home/avifenesh/projects/wt-iqexp
R=$W/research/iq-k32-20260807
RAW=$R/raw
NSYSD=/tmp/iqk32-nsys
NSYS=/usr/local/cuda-13.1/bin/nsys
mkdir -p "$RAW" "$NSYSD"

GEMMA=/data/ai-ml/hf-models/gemma4-26b-a4b-qat-gguf/gemma-4-26B_q4_0-it.gguf
KAT=/data/ai-ml/hf-models/kat-coder-v25-dev-gguf/Kwaipilot_KAT-Coder-V2.5-Dev-IQ4_XS.gguf
Q35=/data/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
P2048=$W/research/depth-decode-20260802/depth-2048-kat.txt
P4096=$W/research/depth-decode-20260802/depth-4096-kat.txt
P2048Q=$W/research/depth-decode-20260802/depth-2048-q35.txt

point() { # name model prompt ngen
  local name=$1 model=$2 prompt=$3 ngen=$4
  local rep=$NSYSD/$name log=$RAW/$name.log
  echo "=== $name (ngen=$ngen) $(date -u +%FT%TZ) ===" > "$log"
  { nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used --format=csv,noheader
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader; } >> "$log"
  MEMRA_PROFILE_GEN=1 MEMRA_NGEN=$ngen MEMRA_NMEASURE=0 MEMRA_PROMPT_FILE="$prompt" \
    flock /tmp/gpu5090.lock timeout 1800 \
    "$NSYS" profile -c cudaProfilerApi --capture-range-end=stop --trace=cuda -f true -o "$rep" \
    "$W/target/release/run-gen" "$model" >> "$log" 2>&1
  echo "rc=$?" >> "$log"
  "$NSYS" stats --report cuda_gpu_kern_sum --format csv --output "$NSYSD/$name" \
    "$rep.nsys-rep" > /dev/null 2>&1
  cp "$NSYSD/${name}_cuda_gpu_kern_sum.csv" "$RAW/$name-kern-sum.csv" 2>/dev/null \
    || echo "NO KERN SUM for $name" >> "$log"
  grep -a "rc=\|tok/s (\|MATCH\|MISMATCH" "$log" | tail -5
}

case "${1:-all}" in
  gemma) point gemma26b-bal "$GEMMA" "$P2048" 128
         point gemma26b-pp  "$GEMMA" "$P4096" 16 ;;
  kat)   point kat-bal "$KAT" "$P2048" 128
         point kat-pp  "$KAT" "$P4096" 16 ;;
  q35)   point q35-bal "$Q35" "$P2048Q" 32 ;;
  all)   point gemma26b-bal "$GEMMA" "$P2048" 128
         point gemma26b-pp  "$GEMMA" "$P4096" 16
         point kat-bal "$KAT" "$P2048" 128
         point kat-pp  "$KAT" "$P4096" 16
         point q35-bal "$Q35" "$P2048Q" 32 ;;
esac
echo DONE
