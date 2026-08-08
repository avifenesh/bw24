#!/usr/bin/env bash
# pp2-hardening Phase 1 gate battery — 2x RTX PRO 6000 (g7e.12xlarge), 2026-08-06.
# Receipts to ~/receipts/pp2/gates/. tee first, parse second. Params baked as literals.
set -uo pipefail
cd ~/memra
export PATH=$HOME/.cargo/bin:/usr/local/cuda-13.2/bin:$PATH
export MEMRA_NVCC=/usr/local/cuda-13.2/bin/nvcc
OUT=~/receipts/pp2/gates
mkdir -p "$OUT"
Q9=/scratch-models/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
BIN=target/release
FAILS=0

nvidia-smi --query-gpu=index,name,memory.used,temperature.gpu,clocks.sm,power.draw --format=csv > "$OUT/gpu-state-pre.txt" 2>&1

run() { local log="$OUT/$1"; shift
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  echo "=== $log: env[${envs[*]:-}] $*"
  if ! env "${envs[@]}" "$@" 2>&1 | tee "$log"; then echo "FAIL: $log"; FAILS=$((FAILS+1)); fi
}

# ---- transport smoke ----
run pp-transport-smoke.log -- $BIN/pp-transport-smoke

# ---- N=2 on the PRO pair: THE gate this lane exists for (never run on this silicon) ----
run ppn-q9-n2-singledev.log       -- $BIN/ppn-gate "$Q9" 2 16 32
run ppn-q9-n2-dev01.log           MEMRA_PP_DEVICES=0,1 -- $BIN/ppn-gate "$Q9" 2 16 32
run ppn-q9-n2-dev10.log           MEMRA_PP_DEVICES=1,0 -- $BIN/ppn-gate "$Q9" 2 16 32
run ppn-q9-n2-dev01-noshard.log   MEMRA_PP_DEVICES=0,1 MEMRA_PP_SHARD=0 -- $BIN/ppn-gate "$Q9" 2 16 32
run ppn-q9-n2-dev01-overlap.log   MEMRA_PP_DEVICES=0,1 MEMRA_PP_OVERLAP=1 -- $BIN/ppn-gate "$Q9" 2 16 32
run ppn-q9-n2-dev01-split5.log    MEMRA_PP_DEVICES=0,1 MEMRA_PP_SPLITS=5 -- $BIN/ppn-gate "$Q9" 2 16 32
run ppn-q9-n2-streams0.log        MEMRA_PP_STREAMS=0 -- $BIN/ppn-gate "$Q9" 2 16 32
# N=4 over 2 devices (2 stages per card — the shape a 4-cut split on a pair would take)
run ppn-q9-n4-dev0101.log         MEMRA_PP_DEVICES=0,1,0,1 -- $BIN/ppn-gate "$Q9" 4 16 32
run ppn-q9-n4-dev0011.log         MEMRA_PP_DEVICES=0,0,1,1 -- $BIN/ppn-gate "$Q9" 4 16 32

# ---- legacy pp2-gate (M1 binary semantics) ----
run pp2-q9-legacy-singledev.log   -- $BIN/pp2-gate "$Q9" 16 32
run pp2-q9-legacy-dev01.log       MEMRA_PP_DEVICES=0,1 -- $BIN/pp2-gate "$Q9" 16 32

# ---- door-shut regression: naked run-gen argmax (PP-2 must be exact vs 1-GPU) ----
run run-gen-q9-naked.log          MEMRA_NGEN=8 -- $BIN/run-gen "$Q9" 55

echo; echo "==== verdicts ===="
grep -H "ppn gate PASS\|ppn gate FAIL\|NOTE: pipelined" $OUT/ppn-*.log | sed "s|$OUT/||"
grep -H "pp2 gate PASS\|pp2 gate FAIL" $OUT/pp2-*.log | sed "s|$OUT/||"
grep -H "pp-transport-smoke PASS\|pp-transport-smoke FAIL" $OUT/pp-transport-smoke.log | sed "s|$OUT/||"
grep -H "MATCH\|MISMATCH" $OUT/run-gen-q9-naked.log | sed "s|$OUT/||" | tail -3
echo "script-detected failures: $FAILS"
exit $FAILS
