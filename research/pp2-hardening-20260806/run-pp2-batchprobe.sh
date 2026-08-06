#!/usr/bin/env bash
# pp2-hardening — WHAT DOES THE BATCH PATH DO UNDER AN OPEN PP DOOR?
# `warn_unwired_once` fires only for the two gemma4 eager sites (decode.rs:615,
# hybrid_forward.rs:6462). decode_step_batch has NO pp guard beyond the b1_fast
# exclusion (decode_batch.rs:361) — so with the door open + weights sharded to dev1,
# does it (a) fail loud, (b) silently peer-read the whole trunk from dev0, or
# (c) produce wrong logits? For a serving lane (c) is the dangerous answer.
set -uo pipefail
cd ~/memra
export PATH=$HOME/.cargo/bin:/usr/local/cuda-13.2/bin:$PATH
OUT=~/receipts/pp2/batchprobe
mkdir -p "$OUT"
Q9=/scratch-models/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
BIN=target/release

# baseline: door shut, the shipped batched gate (config mode) — must be clean
env MEMRA_KC_MODELS_DIR=/scratch-models $BIN/decode-batch-gate "$Q9" --steps 32 --batch 4 --mode config \
  > "$OUT/dbg-doorshut-b4.log" 2>&1
echo "=== door SHUT B=4 (baseline) ==="; grep -E "PASS|FAIL|gate1|gate2" "$OUT/dbg-doorshut-b4.log" | tail -8

# door open, SAME DEVICE (no placement): split exists but all weights on dev0
MEMRA_PP_STAGES=2 $BIN/decode-batch-gate "$Q9" --steps 32 --batch 4 --mode config \
  > "$OUT/dbg-door2-singledev-b4.log" 2>&1
echo "=== door OPEN stages=2 singledev B=4 ==="; grep -E "PASS|FAIL|gate1|gate2|unwired|panic|Error" "$OUT/dbg-door2-singledev-b4.log" | tail -10

# door open, CROSS-DEVICE + sharded: stage-1 weights live on dev1
MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 $BIN/decode-batch-gate "$Q9" --steps 32 --batch 4 --mode config \
  > "$OUT/dbg-door2-dev01-b4.log" 2>&1
echo "=== door OPEN stages=2 dev01 SHARDED B=4 ==="; grep -E "PASS|FAIL|gate1|gate2|unwired|panic|Error|CUDA" "$OUT/dbg-door2-dev01-b4.log" | tail -12

# and the serve-path B=1 fast arm under the door (the b1_fast exclusion at 361)
MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 $BIN/decode-batch-gate "$Q9" --steps 32 --batch 1 --mode config \
  > "$OUT/dbg-door2-dev01-b1.log" 2>&1
echo "=== door OPEN dev01 B=1 ==="; grep -E "PASS|FAIL|gate1|gate2|panic|Error|CUDA" "$OUT/dbg-door2-dev01-b1.log" | tail -10
echo BATCHPROBE_DONE
