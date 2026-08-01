#!/usr/bin/env bash
# M1 increment 2 gate battery (local RTX 5090) — every GPU run under the shared rig lock.
# Raw logs land next to this script (evidence discipline: tee first, parse second).
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT="research/m1-inc2-20260801"
Q9=/home/avifenesh/models/qwen3.5-9b-judge-q8_0.gguf
G12=/home/avifenesh/models/gemma-4-12b-it-qat/gemma-4-12b-it-qat-q4_0.gguf
LOCK="flock /tmp/gpu5090.lock"
BIN=target/release
FAILS=0

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

# ---- transport smoke (peer-arm FFI + Pp2Rt event choreography, single device) ----
run pp-transport-smoke.log -- $BIN/pp-transport-smoke

# ---- pp2-gate: q9 splits 16(default)/5/31 x {inc2 baseline, overlap, devices=0,0} ----
for split in "" 5 31; do
    tag=${split:-default}
    run "pp2-q9-split${tag}-inc2.log"    -- $BIN/pp2-gate "$Q9" 16 32 $split
    run "pp2-q9-split${tag}-overlap.log" MEMRA_PP_OVERLAP=1 -- $BIN/pp2-gate "$Q9" 16 32 $split
    run "pp2-q9-split${tag}-dev00.log"   MEMRA_PP_DEVICES=0,0 -- $BIN/pp2-gate "$Q9" 16 32 $split
done
# rollback seam (increment-1 same-stream) — one split
run pp2-q9-splitdefault-streams0.log MEMRA_PP_STREAMS=0 -- $BIN/pp2-gate "$Q9" 16 32

# ---- pp2-gate: g12 splits 24(default)/7 x the same knob set ----
for split in "" 7; do
    tag=${split:-default}
    run "pp2-g12-split${tag}-inc2.log"    -- $BIN/pp2-gate "$G12" 16 32 $split
    run "pp2-g12-split${tag}-overlap.log" MEMRA_PP_OVERLAP=1 -- $BIN/pp2-gate "$G12" 16 32 $split
    run "pp2-g12-split${tag}-dev00.log"   MEMRA_PP_DEVICES=0,0 -- $BIN/pp2-gate "$G12" 16 32 $split
done
run pp2-g12-splitdefault-streams0.log MEMRA_PP_STREAMS=0 -- $BIN/pp2-gate "$G12" 16 32

# ---- no-door-open regression proof: kernel-check + naked run-gen argmax ----
run kernel-check.log -- $BIN/kernel-check
run run-gen-q9-naked.log MEMRA_NGEN=8 -- $BIN/run-gen "$Q9" 55
run run-gen-g12-naked.log MEMRA_NGEN=8 -- $BIN/run-gen "$G12" 55

echo
echo "==== verdicts ===="
grep -H "pp2 gate PASS\|pp2 gate FAIL" $OUT/pp2-*.log | sed "s|$OUT/||"
grep -H "pp-transport-smoke PASS\|pp-transport-smoke FAIL" $OUT/pp-transport-smoke.log | sed "s|$OUT/||"
grep -Hc "FAIL" $OUT/kernel-check.log | sed "s|$OUT/||;s|$| (FAIL-line count; 0 = green)|"
grep -H "MATCH\|MISMATCH" $OUT/run-gen-q9-naked.log $OUT/run-gen-g12-naked.log | sed "s|$OUT/||" | tail -4
echo "script-detected failures: $FAILS"
exit $FAILS
