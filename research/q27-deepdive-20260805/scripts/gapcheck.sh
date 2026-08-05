#!/bin/bash
# The "last 16%" question, closed: re-capture the decode timeline WITH the gen-graph door open
# and compare busy%/gap% against the eager capture (7.67% gaps over 129,919 gaps @ 1015
# launches/token). MEMRA_PROFILE_GEN=2 brackets the decode loop only.
set -u
W=/root/bw24; R=/root/receipts-dd
Q8=/root/models/Qwen3.6-27B-Q8_0.gguf
export LD_LIBRARY_PATH=/usr/local/cuda-13.1/compat
export PATH=$HOME/.cargo/bin:/usr/local/cuda-13.1/bin:$PATH
PROMPT=$W/research/e2e/prompts/pp512.txt
for a in 0 1; do
  TAG=eager; [ "$a" = 1 ] && TAG=graph
  OUT=$R/nsys/nsys-q8-decode-c1-$TAG; L=$OUT.log
  nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm --format=csv,noheader > "$L" 2>&1
  MEMRA_GEN_GRAPH=$a MEMRA_PROMPT_FILE=$PROMPT MEMRA_NGEN=128 MEMRA_PROFILE_GEN=2 \
    timeout 2400 nsys profile -o "$OUT" --force-overwrite=true -c cudaProfilerApi \
    --trace=cuda --cuda-memory-usage=false \
    "$W/target/release/run-gen" "$Q8" >> "$L" 2>&1
  echo "nsys $TAG rc=$?"
  nsys stats --report cuda_gpu_kern_sum --format csv -o "$OUT" "$OUT.nsys-rep" >> "$L" 2>&1
  nsys stats --report cuda_gpu_trace  --format csv -o "$OUT-trace" "$OUT.nsys-rep" >> "$L" 2>&1
done
echo GAPCHECK-DONE
