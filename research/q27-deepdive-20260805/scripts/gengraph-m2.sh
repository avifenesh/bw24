#!/bin/bash
# LEVER 2 CROSS-MODEL ARM: the MEMRA_GEN_GRAPH budget key is a SHIPPED CROSS-MODEL default, so
# moving it needs evidence on a second artifact, not just q27-Q8_0. Same interleaved protocol:
# arms alternate order per rep so thermal drift cancels in the pair mean.
set -u
W=/root/bw24; R=/root/receipts-dd
MDL=${3:-/root/models/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf}
STEM=$(basename "$MDL" .gguf)
export LD_LIBRARY_PATH=/usr/local/cuda-13.1/compat
export PATH=$HOME/.cargo/bin:/usr/local/cuda-13.1/bin:$PATH
PROMPT=$W/research/e2e/prompts/pp512.txt
NGEN=${1:?ngen}; REP=${2:?rep}
if [ $((REP % 2)) -eq 1 ]; then ORD="0 1"; else ORD="1 0"; fi
for a in $ORD; do
  TAG=eager; [ "$a" = 1 ] && TAG=graph
  L=$R/logs/gengraph-m2-$STEM-n$NGEN-$TAG-r$REP.log
  { nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm,clocks.mem --format=csv,noheader; echo "model=$STEM arm=$TAG MEMRA_GEN_GRAPH=$a ngen=$NGEN"; } > "$L" 2>&1
  MEMRA_GEN_GRAPH=$a MEMRA_PROMPT_FILE=$PROMPT MEMRA_NGEN=$NGEN \
    timeout 1800 "$W/target/release/run-gen" "$MDL" >> "$L" 2>&1
  rc=$?
  { nvidia-smi --query-gpu=temperature.gpu,clocks.mem --format=csv,noheader; } >> "$L" 2>&1
  echo "m2 n=$NGEN $TAG r$REP rc=$rc $(grep -oE "generated [0-9]+ tokens in [0-9.]+s = [0-9.]+ tok/s" "$L" | head -1) | $(grep -oE "(MATCH|MISMATCH)" "$L" | head -2 | tr "\n" " ") | $(grep -c "door CLOSED" "$L") closed"
done
