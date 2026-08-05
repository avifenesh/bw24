#!/bin/bash
# LEVER 2: MEMRA_GEN_GRAPH door. Default is budget-keyed >=256, so the 128-tok anchors ran EAGER.
# Two questions: (a) at the 512-budget official shape, does the door pay on q27-Q8_0?
# (b) does it pay at 128 (i.e. should the budget key drop)? Interleaved, alternating order.
set -u
W=/root/bw24; R=/root/receipts-dd
Q8=/root/models/Qwen3.6-27B-Q8_0.gguf
export LD_LIBRARY_PATH=/usr/local/cuda-13.1/compat
export PATH=$HOME/.cargo/bin:/usr/local/cuda-13.1/bin:$PATH
PROMPT=$W/research/e2e/prompts/pp512.txt
NGEN=${1:?ngen}; REP=${2:?rep}
if [ $((REP % 2)) -eq 1 ]; then ORD="0 1"; else ORD="1 0"; fi
for a in $ORD; do
  TAG=eager; [ "$a" = 1 ] && TAG=graph
  L=$R/logs/gengraph-n$NGEN-$TAG-r$REP.log
  { nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm,clocks.mem --format=csv,noheader; echo "arm=$TAG MEMRA_GEN_GRAPH=$a ngen=$NGEN"; } > "$L" 2>&1
  MEMRA_GEN_GRAPH=$a MEMRA_PROMPT_FILE=$PROMPT MEMRA_NGEN=$NGEN \
    timeout 1800 "$W/target/release/run-gen" "$Q8" >> "$L" 2>&1
  rc=$?
  echo "gengraph n=$NGEN $TAG r$REP rc=$rc $(grep -oE "generated [0-9]+ tokens in [0-9.]+s = [0-9.]+ tok/s" "$L" | head -1) | $(grep -oE "(MATCH|MISMATCH)" "$L" | head -2 | tr "\n" " ") | $(grep -c "door CLOSED" "$L") closed"
done
