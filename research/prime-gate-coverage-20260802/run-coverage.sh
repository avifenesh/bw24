#!/usr/bin/env bash
# prime-gate coverage sweep (gap #46): batched-prime vs tokenwise-prime first-token
# agreement across the locally-present supported/bring-up model set. Every GPU run under
# the shared rig lock; raw logs + per-prompt jsonl land next to this script (tee first,
# parse second).
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT="research/prime-gate-coverage-20260802"
LOCK="flock /tmp/gpu5090.lock"
BIN=target/release
PROMPTS="$OUT/prompts-mixed.txt"
CHATPROMPTS="research/concat-prime-exact-20260802/prompts16.txt"
FAILS=0

declare -A MODELS=(
  [q35]=/data/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
  [q9j]=/home/avifenesh/models/qwen3.5-9b-judge-q8_0.gguf
  [o9b]=/data/ai-ml/hf-models/ornith-1.0-9b-gguf/ornith-1.0-9b-Q8_0.gguf
  [o35b]=/data/ai-ml/hf-models/ornith-1.0-35b-gguf/ornith-1.0-35b-Q4_K_M.gguf
  [kat]=/data/ai-ml/hf-models/kat-coder-v25-dev-gguf/Kwaipilot_KAT-Coder-V2.5-Dev-IQ4_XS.gguf
  [g12]=/home/avifenesh/models/gemma-4-12b-it-qat/gemma-4-12b-it-qat-q4_0.gguf
)

for tag in q35 q9j o9b o35b kat g12; do
    m="${MODELS[$tag]}"
    [ -f "$m" ] || { echo "SKIP $tag (model absent)"; continue; }
    echo "=== $tag raw arm ==="
    $LOCK $BIN/prime-gate "$m" --prompts-file "$PROMPTS" --steps 16 \
        --jsonl "$OUT/coverage-$tag-raw.jsonl" 2>&1 | tee "$OUT/coverage-$tag-raw.log" \
        | grep -E "prompt |SUMMARY|GREEN|FAIL" || FAILS=$((FAILS+1))
    echo "=== $tag chat arm ==="
    $LOCK $BIN/prime-gate "$m" --prompts-file "$CHATPROMPTS" --chat --steps 16 \
        --jsonl "$OUT/coverage-$tag-chat.jsonl" 2>&1 | tee "$OUT/coverage-$tag-chat.log" \
        | grep -E "prompt |SUMMARY|GREEN|FAIL" || FAILS=$((FAILS+1))
done

echo "coverage sweep done; script-detected non-green runs: $FAILS"
