#!/usr/bin/env bash
# gate1-recal characterization: gate1-config divergence-step distribution on THIS rig
# (RTX 5090 Laptop), 18 draws per model (MEMRA_GATE_SEED in {0,6,12} x 6 internal draws),
# UNMODIFIED gate binary at the lane base (restructure/public-split). Full battery output
# is kept (gates 2/3 ride along at bit strength — they are the isolation contract).
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT="research/gate1-recal-20260802"
LOCK="flock /tmp/gpu5090.lock"
BIN=target/release

declare -A MODELS=(
  [q9j]=/home/avifenesh/models/qwen3.5-9b-judge-q8_0.gguf
  [q35]=/data/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
)

for tag in q9j q35; do
    m="${MODELS[$tag]}"
    for base in 0 6 12; do
        log="$OUT/sweep-$tag-base$base.log"
        echo "=== $log (MEMRA_GATE_SEED=$base) ==="
        $LOCK env MEMRA_GATE_SEED=$base $BIN/decode-batch-gate "$m" > "$log" 2>&1
        echo "exit=$? ($log)"
    done
done

echo "---- distribution ----"
grep -H "gate1 seed\|gate1 (" $OUT/sweep-*.log | sed "s|$OUT/||"
