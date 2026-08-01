#!/bin/bash
# inc3 (3a) pinpoint: attribute the B=16/32 bit-isolation break to a kernel class.
#  strict      — equalized-env plumbing pin (bit-identity law) on this rig.
#  b16-noattn  — z-batched fa/append OFF: if gate2 still fails, the matvec tier is the breaker.
#  b16-nogemm  — GEMM tier off, no mirror: m=16 falls to the dp4a grid.y=m tail.
#  b16-b16mmvq — GEMM off + q8rp mirror: m=16 rides qmatvec_q8_0_mmvq_b16_rp (the exact-16 candidate).
#  b32-b16mmvq — same env at B=32: m=32 has no b-tier kernel (documents the 32 wall).
set -u
W=/home/avifenesh/projects/wt-batched-tick-3
R=$W/research/batched-tick-inc3-20260801
M=/home/avifenesh/models/qwen3.5-9b-judge-q8_0.gguf
G=$W/target/release/decode-batch-gate
run() {
  local name=$1 batch=$2 steps=$3 mode=$4; shift 4
  local log=$R/dbg-$name-s$steps.log
  echo "=== $name --batch $batch --steps $steps --mode $mode env: $* $(date -u +%FT%TZ) ===" | tee "$log"
  flock /tmp/gpu5090.lock env "$@" "$G" "$M" --steps "$steps" --batch "$batch" --mode "$mode" >>"$log" 2>&1
  echo "exit=$? $(grep -E 'gate2 \(|ALL GREEN' "$log" | tail -1)"
}
run strict-b4 4 32 strict MEMRA_MMVQ=0 MEMRA_NO_FUSE_NORMQ=1
run strict-b4 4 160 strict MEMRA_MMVQ=0 MEMRA_NO_FUSE_NORMQ=1
run b16-noattn 16 32 config MEMRA_DECODE_BATCH_CAP=16 MEMRA_BATCH_FA=0 MEMRA_BATCH_APPEND=0
run b16-nogemm 16 32 config MEMRA_DECODE_BATCH_CAP=16 MEMRA_NO_GEMM=1
run b16-b16mmvq 16 32 config MEMRA_DECODE_BATCH_CAP=16 MEMRA_NO_GEMM=1 MEMRA_Q8RP=1
run b16-b16mmvq 16 160 config MEMRA_DECODE_BATCH_CAP=16 MEMRA_NO_GEMM=1 MEMRA_Q8RP=1
run b32-b16mmvq 32 32 config MEMRA_DECODE_BATCH_CAP=32 MEMRA_NO_GEMM=1 MEMRA_Q8RP=1
echo PINPOINT-3A-DONE
