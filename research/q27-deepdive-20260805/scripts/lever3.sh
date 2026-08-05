#!/bin/bash
# LEVER 3: batched dense-FFN gate+up launch fusion (fused2_b8) at the SERVING tier.
# Gate first (decode-batch-gate = per-row bit-identity vs isolated m=1), then the A/B.
set -u
cd /root/bw24
R=/root/receipts-dd
Q8=/root/models/Qwen3.6-27B-Q8_0.gguf
export LD_LIBRARY_PATH=/usr/local/cuda-13.1/compat
export PATH=$HOME/.cargo/bin:/usr/local/cuda-13.1/bin:$PATH

# --- gate: batched decode bit-strength, both arms (the fused arm must not weaken it)
for a in 0 1; do
  TAG=off; [ "$a" = 1 ] && TAG=on
  MEMRA_Q8_FFN_FUSE2=$a timeout 3600 ./target/release/decode-batch-gate "$Q8" \
    > "$R/logs/gate-decode-batch-fuse-$TAG.log" 2>&1
  echo "gate decode-batch fuse=$TAG rc=$? : $(grep -cE 'FAIL' "$R/logs/gate-decode-batch-fuse-$TAG.log") FAILs | $(grep -oE 'ALL (GREEN|PASS)|PASS|GREEN' "$R/logs/gate-decode-batch-fuse-$TAG.log" | tail -1)"
done

# --- A/B: bench c=8 aggregate, arms interleaved, order alternated per pass
for pass in 1 2 3; do
  if [ $((pass % 2)) -eq 1 ]; then ORD="0 1"; else ORD="1 0"; fi
  for a in $ORD; do
    TAG=off; [ "$a" = 1 ] && TAG=on
    L=$R/logs/lever3-bench-$TAG-p$pass.log
    { nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm,clocks.mem --format=csv,noheader; echo "arm=$TAG"; } > "$L" 2>&1
    MEMRA_Q8_FFN_FUSE2=$a timeout 3600 ./target/release/decode-batch-bench "$Q8" \
      --steps 128 --reps 3 --batches 8 --ctx 512 >> "$L" 2>&1
    echo "lever3 bench $TAG p$pass rc=$? $(grep -E '^B=8' "$L")"
  done
done
echo LEVER3-DONE
