#!/bin/bash
# inc3 (3a) FINAL gate matrix — one binary (post exact-16 policy + 3c defer + gate mem fix).
# fN- prefix = final. The exact-16 tier needs NO cap door: MEMRA_Q8RP=1 alone admits B=9..16
# (decode_batch_exact16_ok + verify_exact scope). b16-naked must REFUSE (assert) without
# the mirror. b32 stays the non-exact probe (env door + expected gate2 FAIL).
set -u
W=/home/avifenesh/projects/wt-batched-tick-3
R=$W/research/batched-tick-inc3-20260801
M=/home/avifenesh/models/qwen3.5-9b-judge-q8_0.gguf
G=$W/target/release/decode-batch-gate
run() {
  local name=$1 batch=$2 steps=$3 mode=$4; shift 4
  local log=$R/fN-$name-s$steps.log
  echo "=== FINAL $name --batch $batch --steps $steps --mode $mode env: ${*:-none} $(date -u +%FT%TZ) ===" | tee "$log"
  if [ $# -gt 0 ]; then
    flock /tmp/gpu5090.lock env "$@" "$G" "$M" --steps "$steps" --batch "$batch" --mode "$mode" >>"$log" 2>&1
  else
    flock /tmp/gpu5090.lock "$G" "$M" --steps "$steps" --batch "$batch" --mode "$mode" >>"$log" 2>&1
  fi
  echo "exit=$? | $(grep -E 'gate1 \(|gate2 \(|gate3 \(|ALL GREEN|refused|panicked' "$log" | tr '\n' ' | ')"
}
run b8 8 32 config
run b8 8 160 config
run strict-b4 4 160 strict MEMRA_MMVQ=0 MEMRA_NO_FUSE_NORMQ=1
run b12-q8rp 12 32 config MEMRA_Q8RP=1
run b16-q8rp 16 32 config MEMRA_Q8RP=1
run b16-q8rp 16 160 config MEMRA_Q8RP=1
run b16-refuse 16 32 config
run b32-door 32 32 config MEMRA_DECODE_BATCH_CAP=32 MEMRA_Q8RP=1
echo FINAL-GATES-DONE
