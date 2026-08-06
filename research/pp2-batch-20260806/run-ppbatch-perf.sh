#!/usr/bin/env bash
# pp2-batch STEP 5 — the CAPACITY receipt: what batched PP-2 costs vs door-shut single-device.
#
# NOT the goal of the lane (the goal is that batched serving is POSSIBLE at all on a >VRAM
# SKU), but the number the Step SKU's serving plan needs: does batched PP-2 cost the same
# ~0.4% as serial eager PP-2, or does the [B, n_embd] boundary transfer bite at m>1?
#
# Arms per rep (rep-major interleave — cross-run/cross-day comparisons are clock-drift
# invalid per the H100 laws, so all arms run back to back inside each rep):
#   A door SHUT, single device            — the denominator (full-speed, single-card capacity)
#   B door OPEN stages=2 SINGLEDEV        — the SEAM cost alone (2 streams, dtod boundary,
#                                           per-stage engines; no placement, no PCIe)
#   C door OPEN dev01 SHARDED (the split) — the real PP-2 serving config
#   D door OPEN dev10 SHARDED             — the other placement order
# Arm C vs A = the capacity price; C vs B = the transport price; C vs D = placement symmetry.
#
# All arms allocate caches through pp::new_cache (the bench was fixed in this lane), so a
# remote stage's own KV is NOT peer-read — otherwise the harness would charge the split for
# a harness bug.
set -uo pipefail
cd ~/memra
export PATH=$HOME/.cargo/bin:$PATH
OUT=~/receipts/pp2batch/perf
mkdir -p "$OUT"
Q9=/scratch-models/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
BIN=target/release
# B=16 exceeds the default width cap 8 and q9 (NVFP4) has NO exact-16 tier
# (`decode_batch_exact16_ok` admits only Q4_0/Q6_K/F8_E4M3/Q8_0+rp4), so without this door
# every arm would panic at B=16 with "> cap 8 with no exact tier — refused". Applied to ALL
# FOUR ARMS equally, so it cannot bias the comparison: it selects the same non-exact m>=16
# tier on both sides, and that tier was gated bit-identical across the split in this lane
# (serve receipts, ppbatch-q9-dev01-b16-cap16.log). It is a MEASUREMENT door, not a serving
# default — the capacity numbers at B=16 describe the door-open tier, not shipped behavior.
export MEMRA_DECODE_BATCH_CAP=16

nvidia-smi --query-gpu=index,memory.used,temperature.gpu,clocks.sm,power.draw \
  --format=csv > "$OUT/gpu-pre.csv"

for r in 1 2 3 4 5; do
  echo "--- rep $r arm A: door SHUT single-device ---"
  $BIN/decode-batch-bench "$Q9" --steps 64 --reps 1 --batches 1,4,8,16 \
    > "$OUT/r$r-A-doorshut.log" 2>&1
  echo "--- rep $r arm B: door OPEN stages=2 singledev (seam only) ---"
  MEMRA_PP_STAGES=2 $BIN/decode-batch-bench "$Q9" --steps 64 --reps 1 --batches 1,4,8,16 \
    > "$OUT/r$r-B-split-singledev.log" 2>&1
  echo "--- rep $r arm C: door OPEN dev01 sharded (THE serving config) ---"
  MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 \
    $BIN/decode-batch-bench "$Q9" --steps 64 --reps 1 --batches 1,4,8,16 \
    > "$OUT/r$r-C-split-dev01.log" 2>&1
  echo "--- rep $r arm D: door OPEN dev10 sharded (placement symmetry) ---"
  MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=1,0 \
    $BIN/decode-batch-bench "$Q9" --steps 64 --reps 1 --batches 1,4,8,16 \
    > "$OUT/r$r-D-split-dev10.log" 2>&1
done

nvidia-smi --query-gpu=index,memory.used,temperature.gpu,clocks.sm,power.draw \
  --format=csv > "$OUT/gpu-post.csv"

echo "==== raw per-arm per-rep lines (B=16 needs the exact-16 tier; a refusal shows here) ===="
for a in A-doorshut B-split-singledev C-split-dev01 D-split-dev10; do
  echo "-- arm $a"
  grep -h "^B=" $OUT/r*-$a.log
done
echo PERF_DONE
