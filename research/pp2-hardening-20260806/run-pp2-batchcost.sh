#!/usr/bin/env bash
# pp2-hardening — THE COST OF THE SILENT PEER-READ.
# decode_step_batch has NO pp guard: under an open cross-device door with sharded weights,
# it runs the WHOLE trunk on the primary engine's stream and peer-reads stage-1's weights
# over PCIe. Exactness passes (peer reads return identical bytes) so nothing fails loud.
# This measures what that costs — the number that decides whether the batch door needs a
# refusal (fail-closed) or a wiring (fail-forward).
# Interleaved A/B, rep-major, N=5. Same process per arm-invocation (door env is load-time).
set -uo pipefail
cd ~/memra
export PATH=$HOME/.cargo/bin:/usr/local/cuda-13.2/bin:$PATH
OUT=~/receipts/pp2/batchcost
mkdir -p "$OUT"
Q9=/scratch-models/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
BIN=target/release

nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm,power.draw --format=csv > "$OUT/gpu-pre.csv"

# rep-major interleave: for each rep, run all three arms back to back (never arm-major
# batches — cross-run clock drift invalidates that comparison per the H100 laws).
for r in 1 2 3 4 5; do
  echo "--- rep $r arm A: door SHUT (single-GPU baseline, unsharded load) ---"
  $BIN/decode-batch-bench "$Q9" --steps 64 --reps 1 --batches 1,4,8 \
    > "$OUT/r$r-A-doorshut.log" 2>&1
  echo "--- rep $r arm B: door OPEN stages=2 SINGLEDEV (split, no placement: no peer reads) ---"
  MEMRA_PP_STAGES=2 $BIN/decode-batch-bench "$Q9" --steps 64 --reps 1 --batches 1,4,8 \
    > "$OUT/r$r-B-door-singledev.log" 2>&1
  echo "--- rep $r arm C: door OPEN dev01 SHARDED (stage-1 weights on dev1 = peer reads) ---"
  MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 $BIN/decode-batch-bench "$Q9" --steps 64 --reps 1 --batches 1,4,8 \
    > "$OUT/r$r-C-door-dev01-sharded.log" 2>&1
  echo "--- rep $r arm D: door OPEN dev01 SHARD=0 (weights all on dev0 = no peer reads) ---"
  MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_PP_SHARD=0 $BIN/decode-batch-bench "$Q9" --steps 64 --reps 1 --batches 1,4,8 \
    > "$OUT/r$r-D-door-dev01-noshard.log" 2>&1
done

nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm,power.draw --format=csv > "$OUT/gpu-post.csv"
echo "==== raw aggregate lines per arm/rep ===="
for a in A-doorshut B-door-singledev C-door-dev01-sharded D-door-dev01-noshard; do
  echo "-- arm $a"
  grep -h "B=" $OUT/r*-$a.log | head -30
done
echo BATCHCOST_DONE
