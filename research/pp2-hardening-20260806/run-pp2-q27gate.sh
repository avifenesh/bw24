#!/usr/bin/env bash
# pp2-hardening — strong-form PP-2 exactness on the DAILY model (q27, 64 layers).
# Phase 1 gated q9 (the same vehicle M1/M2 were minted on, for cross-rig comparability);
# this closes the gap on the model that would actually serve over this pair.
set -uo pipefail
cd ~/memra
export PATH=$HOME/.cargo/bin:/usr/local/cuda-13.2/bin:$PATH
OUT=~/receipts/pp2/q27gate
mkdir -p "$OUT"
Q27=/scratch-models/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
BIN=target/release
LOCK="flock /tmp/memra-gpu.lock"   # cohabitation: step37-p2 shares this box
run(){ local n="$1"; shift; echo "######## $n"; $LOCK "$@" > "$OUT/$n.log" 2>&1; echo "exit=$?"; grep -E 'PASS|FAIL|MATCH|MISMATCH|Error' "$OUT/$n.log"|head -4; }
run ppn-gate-q27-dev01   env MEMRA_PP_DEVICES=0,1 $BIN/ppn-gate "$Q27" 2
run ppn-gate-q27-dev10   env MEMRA_PP_DEVICES=1,0 $BIN/ppn-gate "$Q27" 2
run ppn-gate-q27-singled env $BIN/ppn-gate "$Q27" 2
run rungen-q27-doorshut  env $BIN/run-gen "$Q27" 16 8
run rungen-q27-pp2       env MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 $BIN/run-gen "$Q27" 16 8
echo Q27GATE_DONE
