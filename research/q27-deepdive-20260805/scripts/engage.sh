#!/bin/bash
set -u
W=/root/bw24; R=/root/receipts-dd
Q8=/root/models/Qwen3.6-27B-Q8_0.gguf
export LD_LIBRARY_PATH=/usr/local/cuda-13.1/compat
export PATH=$HOME/.cargo/bin:/usr/local/cuda-13.1/bin:$PATH
PROMPT=$W/research/e2e/prompts/pp512.txt
for a in 0 1; do
  TAG=off; [ "$a" = 1 ] && TAG=on
  OUT=$R/nsys/engage-$TAG
  MEMRA_Q8_FFN_FUSE2=$a MEMRA_PROMPT_FILE=$PROMPT MEMRA_NGEN=16 MEMRA_PROFILE_GEN=2 \
    timeout 1800 nsys profile -o "$OUT" --force-overwrite=true -c cudaProfilerApi \
    --trace=cuda --cuda-memory-usage=false \
    "$W/target/release/run-gen" "$Q8" > "$R/logs/engage-$TAG.log" 2>&1
  nsys stats --report cuda_gpu_kern_sum --format csv -o "$OUT" "$OUT.nsys-rep" >> "$R/logs/engage-$TAG.log" 2>&1
done
for TAG in off on; do
  echo "== arm=$TAG"
  awk -F, "NR>1 {gsub(/\"/,\"\"); print \$3, \$NF}" $R/nsys/engage-$TAG_cuda_gpu_kern_sum.csv 2>/dev/null | head -1
done
