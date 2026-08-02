#!/usr/bin/env bash
# Ship battery for lane/prime-gate-coverage: kernel-check, run-gen argmax (all tested
# models — now including the new #46 batched-prime line), decode-batch + prime-batch
# gates untouched, run-spec smoke. Raw logs land next to this script.
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT="research/prime-gate-coverage-20260802"
LOCK="flock /tmp/gpu5090.lock"
BIN=target/release
FAILS=0

declare -A MODELS=(
  [q35]=/data/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
  [q9j]=/home/avifenesh/models/qwen3.5-9b-judge-q8_0.gguf
  [o9b]=/data/ai-ml/hf-models/ornith-1.0-9b-gguf/ornith-1.0-9b-Q8_0.gguf
  [o35b]=/data/ai-ml/hf-models/ornith-1.0-35b-gguf/ornith-1.0-35b-Q4_K_M.gguf
  [kat]=/data/ai-ml/hf-models/kat-coder-v25-dev-gguf/Kwaipilot_KAT-Coder-V2.5-Dev-IQ4_XS.gguf
  [g12]=/home/avifenesh/models/gemma-4-12b-it-qat/gemma-4-12b-it-qat-q4_0.gguf
)

run() { # run <logname> <env...> -- <cmd...>
    local log="$OUT/$1"; shift
    local envs=()
    while [ "$1" != "--" ]; do envs+=("$1"); shift; done
    shift
    echo "=== $log: env[${envs[*]:-}] $*"
    if ! $LOCK env "${envs[@]}" "$@" > "$log" 2>&1; then
        echo "FAIL: $log"; FAILS=$((FAILS+1))
    fi
}

# 1. kernel-check
run battery-kernel-check.log -- $BIN/kernel-check

# 2. run-gen argmax gate, every tested model, board-shape prompt (pp512 text + NGEN=32)
for tag in q35 q9j o9b o35b kat g12; do
    m="${MODELS[$tag]}"
    [ -f "$m" ] || { echo "SKIP $tag (model absent)"; continue; }
    run "battery-run-gen-$tag.log" MEMRA_NGEN=32 MEMRA_PROMPT_FILE=research/e2e/prompts/pp512.txt \
        -- $BIN/run-gen "$m"
done

# 3. decode-batch gate untouched (probe model + dense control)
run battery-decode-batch-q35.log -- $BIN/decode-batch-gate "${MODELS[q35]}"
run battery-decode-batch-q9j.log -- $BIN/decode-batch-gate "${MODELS[q9j]}"

# 4. cross-request prime-batch gate untouched (fresh + carried)
run battery-prime-batch-q35.log -- $BIN/prime-batch-gate "${MODELS[q35]}" --carried
run battery-prime-batch-q9j.log -- $BIN/prime-batch-gate "${MODELS[q9j]}" --carried

# 5. run-spec K=1..8 self-consistency (probe model, own-trim drafter, board prompt)
run battery-run-spec-q35.log MEMRA_NGEN=64 \
    MEMRA_DRAFT=/data/ai-ml/hf-models/qwen36-35b-moe/draft-35b-owntrim-nvfp4head-q4blk.gguf \
    MEMRA_PROMPT_FILE=research/e2e/prompts/pp512.txt -- $BIN/run-spec "${MODELS[q35]}"

echo
echo "==== verdicts ===="
grep -c "FAIL" $OUT/battery-kernel-check.log | sed 's|^|kernel-check FAIL-lines: |'
grep -H "prefill argmax\|batched-prime argmax" $OUT/battery-run-gen-*.log | sed "s|$OUT/||"
grep -H "ALL GREEN\|FAIL" $OUT/battery-decode-batch-*.log $OUT/battery-prime-batch-*.log 2>/dev/null | sed "s|$OUT/||" | tail -8
grep -H "SELF-CONSISTENCY\|self-consistency" $OUT/battery-run-spec-q35.log | tail -3 | sed "s|$OUT/||"
echo "script-detected failures: $FAILS"
exit $FAILS
