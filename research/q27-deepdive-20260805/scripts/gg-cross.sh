#!/bin/bash
# LEVER 2 sizing: the gen-graph door's budget key is 256. It measured +5.46% at 512 and
# +3.80% at 128 (where it is OFF by default). Find the crossover: below what budget does the
# ~30ms capture stop amortizing? Interleaved arms, order alternated per rep.
set -u
W=/root/bw24; R=/root/receipts-dd
Q8=/root/models/Qwen3.6-27B-Q8_0.gguf
export LD_LIBRARY_PATH=/usr/local/cuda-13.1/compat
export PATH=$HOME/.cargo/bin:/usr/local/cuda-13.1/bin:$PATH
PROMPT=$W/research/e2e/prompts/pp512.txt
for NGEN in 16 32 64; do
  for REP in 1 2 3; do
    if [ $((REP % 2)) -eq 1 ]; then ORD="0 1"; else ORD="1 0"; fi
    for a in $ORD; do
      TAG=eager; [ "$a" = 1 ] && TAG=graph
      L=$R/logs/gengraph-n$NGEN-$TAG-r$REP.log
      { nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm --format=csv,noheader; echo "arm=$TAG ngen=$NGEN"; } > "$L" 2>&1
      MEMRA_GEN_GRAPH=$a MEMRA_PROMPT_FILE=$PROMPT MEMRA_NGEN=$NGEN \
        timeout 1800 "$W/target/release/run-gen" "$Q8" >> "$L" 2>&1
      echo "gengraph n=$NGEN $TAG r$REP rc=$? $(grep -oE 'generated [0-9]+ tokens in [0-9.]+s = [0-9.]+ tok/s' "$L" | head -1) | $(grep -oE '(MATCH|MISMATCH)' "$L" | head -2 | tr '\n' ' ')"
    done
  done
done
echo GG-CROSS-DONE
