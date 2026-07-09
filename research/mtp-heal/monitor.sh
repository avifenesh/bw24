#!/usr/bin/env bash
# Monitor MTP-heal battery progress

set -euo pipefail

cd "$(dirname "$0")"

echo "=== MTP-HEAL PROGRESS CHECK $(date) ==="
echo

echo "Row counts:"
wc -l out-bf16.jsonl out-nvfp4.jsonl 2>/dev/null || echo "  (files not yet created)"
echo

echo "ARM A (bf16) last 3 lines:"
tail -3 arm_a.log 2>/dev/null || echo "  (log not yet created)"
echo

echo "ARM B (nvfp4) last 3 lines:"
tail -3 arm_b.log 2>/dev/null || echo "  (log not yet created)"
echo

echo "Running processes:"
ps aux | grep -E '(run-spec|acceptance_battery)' | grep -v grep | wc -l
echo

echo "GPU status:"
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader
echo

# Check if both are complete
BF16_ROWS=$(wc -l < out-bf16.jsonl 2>/dev/null || echo 0)
NVFP4_ROWS=$(wc -l < out-nvfp4.jsonl 2>/dev/null || echo 0)

if [ "$BF16_ROWS" -ge 44 ] && [ "$NVFP4_ROWS" -ge 44 ]; then
    echo "✓ BOTH ARMS COMPLETE (bf16: $BF16_ROWS, nvfp4: $NVFP4_ROWS)"
    echo "Ready for delta analysis."
else
    echo "Status: bf16=$BF16_ROWS/44, nvfp4=$NVFP4_ROWS/44"
fi
