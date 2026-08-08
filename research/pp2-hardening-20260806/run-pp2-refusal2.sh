#!/usr/bin/env bash
# pp2-hardening — the fail-closed guard now covers FOUR unsplit paths (batch, dc, graph,
# spec verify) through one shared helper. Gate both halves for each: refuses when weights
# are remote, and does NOT refuse (nor change any verdict) otherwise.
set -uo pipefail
cd ~/memra
export PATH=$HOME/.cargo/bin:/usr/local/cuda-13.2/bin:$PATH
OUT=~/receipts/pp2/refusal2
mkdir -p "$OUT"
Q9=/scratch-models/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
BIN=target/release
LOCK="flock /tmp/memra-gpu.lock"
run(){ local n="$1"; shift; echo "######## $n"; $LOCK "$@" > "$OUT/$n.log" 2>&1; echo "exit=$?"; grep -E 'refused|PASS|FAIL|MATCH|MISMATCH|Error' "$OUT/$n.log"|head -4; }

# ---- dc path ----
run dc-1-doorshut        env $BIN/decode-dc-gate "$Q9" 16
run dc-2-dev01-sharded   env MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 $BIN/decode-dc-gate "$Q9" 16
run dc-3-dev01-override  env MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_PP_ALLOW_UNSPLIT_BATCH=1 $BIN/decode-dc-gate "$Q9" 16
run dc-4-dev01-noshard   env MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_PP_SHARD=0 $BIN/decode-dc-gate "$Q9" 16
# ---- graph path (captures the dc chain) ----
run gr-1-doorshut        env $BIN/graph-decode-gate "$Q9" 16
run gr-2-dev01-sharded   env MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 $BIN/graph-decode-gate "$Q9" 16
# ---- spec verify path ----
run sp-1-doorshut        env $BIN/run-spec "$Q9" 16 8 4
run sp-2-dev01-sharded   env MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 $BIN/run-spec "$Q9" 16 8 4
run sp-3-dev01-override  env MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_PP_ALLOW_UNSPLIT_BATCH=1 $BIN/run-spec "$Q9" 16 8 4
# ---- regressions: the arms that DO work must be untouched ----
run rg-1-ppn-gate-dev01  env MEMRA_PP_DEVICES=0,1 $BIN/ppn-gate "$Q9" 2
run rg-2-rungen-doorshut env $BIN/run-gen "$Q9" 16 8
run rg-3-batch-doorshut  env $BIN/decode-batch-gate "$Q9" 4 32 --mode config
echo REFUSAL2_DONE
