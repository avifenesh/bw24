#!/usr/bin/env bash
# M2 ppN gate battery — 8xH100 darklanes-bench box, receipts to ~/receipts/m2-pp8/.
# Every GPU run under the shared box lock. Raw logs land next to the verdicts
# (evidence discipline: tee first, parse second). Params baked as literals.
set -uo pipefail
cd ~/memra
OUT=~/receipts/m2-pp8
mkdir -p "$OUT"
Q9=/opt/dlami/nvme/models/Qwen3.5-9B-Q8_0.gguf
G12=/opt/dlami/nvme/models/gemma-4-12b-it-qat-q4_0.gguf
LOCK="flock /tmp/gpu-box.lock"
BIN=target/release
FAILS=0

nvidia-smi --query-gpu=index,name,memory.used,temperature.gpu --format=csv > "$OUT/gpu-state-pre-gates.txt" 2>&1

run() { # run <logname> <env...> -- <cmd...>
    local log="$OUT/$1"; shift
    local envs=()
    while [ "$1" != "--" ]; do envs+=("$1"); shift; done
    shift
    echo "=== $log: env[${envs[*]:-}] $*"
    if ! $LOCK env "${envs[@]}" "$@" 2>&1 | tee "$log"; then
        echo "FAIL: $log"; FAILS=$((FAILS+1))
    fi
}

# ---- transport smoke (peer-arm FFI + PpNRt event choreography, single device) ----
run pp-transport-smoke.log -- $BIN/pp-transport-smoke

# ---- q9 N=2: the M1 regression surface under the generalized runtime ----
run ppn-q9-n2-singledev.log            -- $BIN/ppn-gate "$Q9" 2 16 32
run ppn-q9-n2-dev01.log                MEMRA_PP_DEVICES=0,1 -- $BIN/ppn-gate "$Q9" 2 16 32
run ppn-q9-n2-dev01-noshard.log        MEMRA_PP_DEVICES=0,1 MEMRA_PP_SHARD=0 -- $BIN/ppn-gate "$Q9" 2 16 32
run ppn-q9-n2-dev01-overlap.log        MEMRA_PP_DEVICES=0,1 MEMRA_PP_OVERLAP=1 -- $BIN/ppn-gate "$Q9" 2 16 32
run ppn-q9-n2-streams0.log             MEMRA_PP_STREAMS=0 -- $BIN/ppn-gate "$Q9" 2 16 32
run ppn-q9-n2-dev01-split5.log         MEMRA_PP_DEVICES=0,1 MEMRA_PP_SPLITS=5 -- $BIN/ppn-gate "$Q9" 2 16 32

# ---- q9 N=4: the core M2 deliverable (single-device AND devices=0..3) ----
run ppn-q9-n4-singledev.log            -- $BIN/ppn-gate "$Q9" 4 16 32
run ppn-q9-n4-dev0123.log              MEMRA_PP_DEVICES=0,1,2,3 -- $BIN/ppn-gate "$Q9" 4 16 32
run ppn-q9-n4-dev0123-noshard.log      MEMRA_PP_DEVICES=0,1,2,3 MEMRA_PP_SHARD=0 -- $BIN/ppn-gate "$Q9" 4 16 32
run ppn-q9-n4-dev0123-asym.log         MEMRA_PP_DEVICES=0,1,2,3 MEMRA_PP_SPLITS=5,16,27 -- $BIN/ppn-gate "$Q9" 4 16 32
run ppn-q9-n4-streams0.log             MEMRA_PP_STREAMS=0 -- $BIN/ppn-gate "$Q9" 4 16 32

# ---- q9 N=8: full-box (single-device AND devices=0..7) ----
run ppn-q9-n8-singledev.log            -- $BIN/ppn-gate "$Q9" 8 16 32
run ppn-q9-n8-dev0to7.log              MEMRA_PP_DEVICES=0,1,2,3,4,5,6,7 -- $BIN/ppn-gate "$Q9" 8 16 32
run ppn-q9-n8-dev0to7-noshard.log      MEMRA_PP_DEVICES=0,1,2,3,4,5,6,7 MEMRA_PP_SHARD=0 -- $BIN/ppn-gate "$Q9" 8 16 32

# ---- g12 N=2 (gemma4 arm rides PpNRt; serial arm only) ----
run ppn-g12-n2-singledev.log           -- $BIN/ppn-gate "$G12" 2 16 32
run ppn-g12-n2-dev01.log               MEMRA_PP_DEVICES=0,1 -- $BIN/ppn-gate "$G12" 2 16 32

# ---- legacy pp2-gate regression (unchanged M1 gate binary semantics) ----
run pp2-q9-legacy-singledev.log        -- $BIN/pp2-gate "$Q9" 16 32
run pp2-q9-legacy-dev01.log            MEMRA_PP_DEVICES=0,1 -- $BIN/pp2-gate "$Q9" 16 32

# ---- no-door-open regression proof: kernel-check + naked run-gen argmax ----
run kernel-check.log -- $BIN/kernel-check
run run-gen-q9-naked.log MEMRA_NGEN=8 -- $BIN/run-gen "$Q9" 55
run run-gen-g12-naked.log MEMRA_NGEN=8 -- $BIN/run-gen "$G12" 55

echo
echo "==== verdicts ===="
grep -H "ppn gate PASS\|ppn gate FAIL" $OUT/ppn-*.log | sed "s|$OUT/||"
grep -H "pp2 gate PASS\|pp2 gate FAIL" $OUT/pp2-*.log | sed "s|$OUT/||"
grep -H "pp-transport-smoke PASS\|pp-transport-smoke FAIL" $OUT/pp-transport-smoke.log | sed "s|$OUT/||"
grep -Hc "FAIL" $OUT/kernel-check.log | sed "s|$OUT/||;s|$| (FAIL-line count; 0 = green)|"
grep -H "MATCH\|MISMATCH" $OUT/run-gen-q9-naked.log $OUT/run-gen-g12-naked.log | sed "s|$OUT/||" | tail -4
echo "script-detected failures: $FAILS"
exit $FAILS
