#!/usr/bin/env bash
# pp2-hardening — gate the decode_step_batch fail-closed refusal (Item 4).
# The refusal must fire on EXACTLY one config class (open door + sharded + 2+ distinct
# devices) and on nothing else. Both halves are gated: it fires when it should (arm 3),
# and every other arm still PASSES its exactness battery (no collateral refusal).
set -uo pipefail
cd ~/memra
export PATH=$HOME/.cargo/bin:/usr/local/cuda-13.2/bin:$PATH
OUT=~/receipts/pp2/refusal
mkdir -p "$OUT"
Q9=/scratch-models/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
BIN=target/release
LOCK="flock /tmp/memra-gpu.lock"   # cohabitation: step37-p2 shares this box

run() { # name, then env+cmd
  local n="$1"; shift
  echo "######## $n"
  $LOCK "$@" > "$OUT/$n.log" 2>&1
  echo "exit=$? -> $OUT/$n.log"
  grep -E 'refused|PASS|FAIL|Error' "$OUT/$n.log" | head -5
}

# 1) door SHUT: must PASS (no refusal, and the B=1 fast path is back)
run 1-doorshut env $BIN/decode-batch-gate "$Q9" 4 32 --mode config
# 2) door OPEN, singledev (no placement): NOT cross-device -> must PASS unchanged
run 2-door-singledev env MEMRA_PP_STAGES=2 $BIN/decode-batch-gate "$Q9" 4 32 --mode config
# 3) door OPEN, dev01, SHARDED (the dangerous config): must REFUSE
run 3-door-dev01-sharded env MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 $BIN/decode-batch-gate "$Q9" 4 32 --mode config
# 4) same as 3 + SHARD=0 (weights home): documented escape -> must PASS
run 4-door-dev01-noshard env MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_PP_SHARD=0 $BIN/decode-batch-gate "$Q9" 4 32 --mode config
# 5) same as 3 + the measurement override -> must PASS (bench arm stays reachable)
run 5-door-dev01-override env MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_PP_ALLOW_UNSPLIT_BATCH=1 $BIN/decode-batch-gate "$Q9" 4 32 --mode config
# 6) door OPEN, repeated device 0,0 -> same-device, NOT remote -> must PASS
run 6-door-dev00 env MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,0 $BIN/decode-batch-gate "$Q9" 4 32 --mode config
# 7) regression: the EAGER pp arm must be untouched by this change
run 7-ppn-gate-dev01 env MEMRA_PP_DEVICES=0,1 $BIN/ppn-gate "$Q9" 2
# 8) regression: kernel-check still green
run 8-kernel-check env $BIN/kernel-check
echo REFUSAL_GATE_DONE
