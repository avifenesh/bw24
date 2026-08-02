#!/usr/bin/env bash
# Strict-mode bit-identity probes under the EQUALIZED env (MEMRA_MMVQ=0
# MEMRA_NO_FUSE_NORMQ=1) on the exact draws where config-mode gate1 diverged earliest
# on this rig: q35 seeds 16/17 (step-0 argmax flips) and q9j seed 0 (step-1 flip).
# Bit-identity here PROVES the config-mode flips are the accepted FP-composition dice,
# not plumbing (#47) — same discriminator as the 2026-07-26 H100 proof.
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT="research/gate1-recal-20260802"
LOCK="flock /tmp/gpu5090.lock"
BIN=target/release
Q35=/data/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
Q9J=/home/avifenesh/models/qwen3.5-9b-judge-q8_0.gguf

probe() { # probe <tag> <seed> <model>
    local log="$OUT/strict-equalized-$1-seed$2.log"
    $LOCK env MEMRA_GATE_SEED=$2 MEMRA_MMVQ=0 MEMRA_NO_FUSE_NORMQ=1 \
        $BIN/decode-batch-gate "$3" --mode strict > "$log" 2>&1
    echo "exit=$? $log"
}

probe q35 16 "$Q35"
probe q35 17 "$Q35"
probe q9j 0  "$Q9J"

grep -H "gate1 (\|gate2 (\|gate3 (\|ALL GREEN\|FAIL" $OUT/strict-equalized-*.log | sed "s|$OUT/||"
