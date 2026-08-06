#!/bin/bash
# Matrix row 1's second cell: chunk-invariance gate in BOTH arms.
#
# The gate (tools/chunk-invariance-gate.sh) asserts that the same prompt primed at different
# MEMRA_PRIME_CHUNK values gives byte-identical greedy output. It is a PREFILL-arithmetic
# property, so it is exactly the kind of contract an f8f4 prefill route could break: f8f4
# changes the activation quantization inside the prime GEMM, and if that made the reduction
# order-sensitive again, chunked prefill would stop being byte-identical.
#
# Run naked (OFF) and with MEMRA_MMQ_F8F4=1 (ON), on both reachable NVFP4 models, plus the
# --canary teeth check in the ON arm (a gate that cannot fail under f8f4 proves nothing about
# f8f4). Both pinned prompts, chunks 2048,64,32.
#
# Usage: chunkinv_ab.sh
set -u
W=/home/avifenesh/projects/wt-f8f4flip
OUT=$W/research/f8f4-flip-20260806/logs
L=$OUT/chunkinv-ab.log
K27=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
Q9=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
{ echo "[start] $(date -Is)"; echo "[commit] $(git -C $W rev-parse HEAD)"
  echo "[gate] tools/chunk-invariance-gate.sh (probe target/release/concat-prime-probe)"
  nvidia-smi --query-gpu=clocks.sm,temperature.gpu,utilization.gpu --format=csv,noheader | sed 's/^/[gpu] /'
} > "$L"
run(){ # run <label> <env-words...> -- <gate args...>
  local label=$1; shift
  local envw=() ; while [ "$1" != "--" ]; do envw+=("$1"); shift; done; shift
  echo "=== $label  env=[${envw[*]:-none}] args=[$*]  $(nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader)" >> "$L"
  ( cd "$W" && env "${envw[@]}" timeout 3600 ./tools/chunk-invariance-gate.sh "$@" ) >> "$L" 2>&1
  echo "[rc=$?] $label done" >> "$L"
}
for M in "$Q9" "$K27"; do
  b=$(basename "$M" .gguf)
  run "chunkinv $b OFF" -- "$M"
  run "chunkinv $b ON"  MEMRA_MMQ_F8F4=1 -- "$M"
done
# teeth: the gate must still be able to FAIL while f8f4 is on
run "chunkinv canary q9 ON" MEMRA_MMQ_F8F4=1 -- "$Q9" --canary
echo "wrote $L"
grep -E "chunk-invariance-gate: (PASS|FAIL|CANARY)|^\[rc=|^=== " "$L"
