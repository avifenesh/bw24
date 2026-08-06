#!/usr/bin/env bash
# pp2-batch B=1 RE-MEASURE — did the per-stage fast path recover the 15%?
#
# Same four arms, same rep-major interleave, B=1 ONLY (the widths that were already at parity
# do not need re-running, and keeping the arm set narrow keeps every comparison inside one
# lock hold on one binary). N=5.
#
#   A door SHUT single-device   — the denominator; unchanged code path, so it also serves as a
#                                 regression check that the edit did not touch the off-door tick
#   B door OPEN stages=2 singledev  — the seam; must now also be ~208, since the fix is the
#                                 same per-stage eager call regardless of placement
#   C door OPEN dev01           — THE serving config
#   D door OPEN dev10           — placement symmetry
#
# Pre-fix medians on this rig (5 reps, same script shape): A 208.5 / B 178.1 / C 177.5 / D 177.3.
set -uo pipefail
cd ~/memra
export PATH=$HOME/.cargo/bin:$PATH
OUT=~/receipts/pp2batch/perf-b1
mkdir -p "$OUT"
Q9=/scratch-models/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
BIN=target/release

nvidia-smi --query-gpu=index,memory.used,temperature.gpu,clocks.sm,power.draw \
  --format=csv > "$OUT/gpu-pre.csv"

for r in 1 2 3 4 5; do
  echo "--- rep $r ---"
  $BIN/decode-batch-bench "$Q9" --steps 64 --reps 1 --batches 1 \
    > "$OUT/r$r-A-doorshut.log" 2>&1
  MEMRA_PP_STAGES=2 $BIN/decode-batch-bench "$Q9" --steps 64 --reps 1 --batches 1 \
    > "$OUT/r$r-B-split-singledev.log" 2>&1
  MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 $BIN/decode-batch-bench "$Q9" --steps 64 --reps 1 --batches 1 \
    > "$OUT/r$r-C-split-dev01.log" 2>&1
  MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=1,0 $BIN/decode-batch-bench "$Q9" --steps 64 --reps 1 --batches 1 \
    > "$OUT/r$r-D-split-dev10.log" 2>&1
done

# Rollback-seam control: MEMRA_SERVE_B1FAST=0 over the split must reproduce the PRE-FIX 177-178
# class. Without it, "the fix worked" and "something else on the box got faster" are the same
# observation. Same lock hold, adjacent to the arms above.
for r in 1 2 3; do
  MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_SERVE_B1FAST=0 \
    $BIN/decode-batch-bench "$Q9" --steps 64 --reps 1 --batches 1 \
    > "$OUT/r$r-E-split-dev01-b1fastoff.log" 2>&1
done

nvidia-smi --query-gpu=index,memory.used,temperature.gpu,clocks.sm,power.draw \
  --format=csv > "$OUT/gpu-post.csv"

for a in A-doorshut B-split-singledev C-split-dev01 D-split-dev10 E-split-dev01-b1fastoff; do
  echo "-- arm $a"; grep -h "^B=1:" $OUT/r*-$a.log
done
echo B1_DONE
