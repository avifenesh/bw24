#!/usr/bin/env bash
# Verification of the recalibrated gate1-config fraction rule (FAIL iff >= 4 of 6 draws
# diverge before step 3):
#   (a) both models x all three seed bases of the characterization sweep — including the
#       two receipted pre-existing failure conditions (q9j MEMRA_GATE_SEED=0, the
#       battery-decode-batch-q9j-BASE.log signature; q35 MEMRA_GATE_SEED=12, the step-0
#       pair this lane surfaced) — must be ALL GREEN;
#   (b) strict-equalized bit-identity re-probe on the worst draws — the hard floor must
#       be untouched by the edit;
#   (c) MEMRA_GATE_CANARY=1 wrong-token plumbing canary on both models — MUST FAIL
#       (teeth check).
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT="research/gate1-recal-20260802"
LOCK="flock /tmp/gpu5090.lock"
BIN=target/release
Q35=/data/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
Q9J=/home/avifenesh/models/qwen3.5-9b-judge-q8_0.gguf

run() { # run <log> <expected-exit> <env...> -- <cmd...>
    local log="$OUT/$1" want="$2"; shift 2
    local envs=()
    while [ "$1" != "--" ]; do envs+=("$1"); shift; done
    shift
    $LOCK env "${envs[@]}" "$@" > "$log" 2>&1
    local got=$?
    if [ "$got" = "$want" ]; then echo "OK   exit=$got (want $want) $log";
    else echo "BAD  exit=$got (want $want) $log"; fi
}

# (a) recalibrated config-mode: both models x seed bases {0,6,12}
for base in 0 6 12; do
    run "verify-q9j-base$base.log" 0 MEMRA_GATE_SEED=$base -- $BIN/decode-batch-gate "$Q9J"
    run "verify-q35-base$base.log" 0 MEMRA_GATE_SEED=$base -- $BIN/decode-batch-gate "$Q35"
done

# (b) strict-equalized hard floor, worst draws, post-edit binary
run "verify-strict-q9j-seed0.log" 0 MEMRA_GATE_SEED=0 MEMRA_MMVQ=0 MEMRA_NO_FUSE_NORMQ=1 \
    -- $BIN/decode-batch-gate "$Q9J" --mode strict
run "verify-strict-q35-seed16.log" 0 MEMRA_GATE_SEED=16 MEMRA_MMVQ=0 MEMRA_NO_FUSE_NORMQ=1 \
    -- $BIN/decode-batch-gate "$Q35" --mode strict

# (c) canary: wrong token fed once -> every draw diverges early -> MUST FAIL (exit 1)
run "verify-canary-q9j.log" 1 MEMRA_GATE_CANARY=1 -- $BIN/decode-batch-gate "$Q9J"
run "verify-canary-q35.log" 1 MEMRA_GATE_CANARY=1 -- $BIN/decode-batch-gate "$Q35"

echo "---- verdicts ----"
grep -H "early draws\|gate1 (\|gate2 (\|gate3 (\|ALL GREEN\|FAILED" $OUT/verify-*.log | sed "s|$OUT/||"
