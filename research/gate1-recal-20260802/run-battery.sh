#!/usr/bin/env bash
# Ship battery for lane/gate1-recal: kernel-check untouched-green + full decode-batch
# battery (gates 1/2/3) on q9j + q35 under the recalibrated gate1 fraction rule.
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT="research/gate1-recal-20260802"
LOCK="flock /tmp/gpu5090.lock"
BIN=target/release
Q35=/data/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
Q9J=/home/avifenesh/models/qwen3.5-9b-judge-q8_0.gguf
FAILS=0

run() { # run <log> -- <cmd...>
    local log="$OUT/$1"; shift 2
    if ! $LOCK "$@" > "$log" 2>&1; then echo "FAIL: $log"; FAILS=$((FAILS+1));
    else echo "OK:   $log"; fi
}

run battery-kernel-check.log -- $BIN/kernel-check
run battery-decode-batch-q9j.log -- $BIN/decode-batch-gate "$Q9J"
run battery-decode-batch-q35.log -- $BIN/decode-batch-gate "$Q35"

echo "---- verdicts ----"
grep -c "FAIL" $OUT/battery-kernel-check.log | sed 's|^|kernel-check FAIL-lines: |'
grep -H "early draws\|gate1 (\|gate2 (\|gate3 (\|ALL GREEN" $OUT/battery-decode-batch-*.log | sed "s|$OUT/||"
echo "script-detected failures: $FAILS"
exit $FAILS
