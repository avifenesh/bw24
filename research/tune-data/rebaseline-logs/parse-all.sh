#!/usr/bin/env bash
# Parse all rebaseline2 logs into structured data
set -euo pipefail

LOG_DIR=/home/avifenesh/projects/bw24/research/tune-data/rebaseline-logs

echo "=== REBASELINE2 DATA EXTRACTION ==="
echo ""

# Model A: 9B GGUF
echo "MODEL A (9B GGUF):"
echo "Plain @d512 bw24:" && grep "decode tg128 @ctx512" $LOG_DIR/A-9b-gguf-plain-d512.log | grep -oP '\d+\.\d+ tok/s'
echo "Plain @d512 llama:" && grep "tg128 @ d512" $LOG_DIR/A-9b-gguf-plain-d512.log | grep -oP '\d+\.\d+(?= ±)'
echo "Plain @d6257 bw24:" && grep "decode tg128 @ctx6257" $LOG_DIR/A-9b-gguf-plain-d6257.log | grep -oP '\d+\.\d+ tok/s'
echo "Plain @d6257 llama:" && grep "tg128 @ d6257" $LOG_DIR/A-9b-gguf-plain-d6257.log | grep -oP '\d+\.\d+(?= ±)'
echo "Spec p1 bw24:" && grep "generate_spec K=3" $LOG_DIR/A-9b-gguf-spec-p1.log | grep -oP '\d+\.\d+ tok/s'
echo "Spec p2 bw24:" && grep "generate_spec K=3" $LOG_DIR/A-9b-gguf-spec-p2.log | grep -oP '\d+\.\d+ tok/s'
echo "Spec p3 bw24:" && grep "generate_spec K=3" $LOG_DIR/A-9b-gguf-spec-p3-sampled.log | grep -oP '\d+\.\d+ tok/s'
echo "Spec llama: N/A (no 9B MTP draft)"
echo ""

# Model B: 9B ST
echo "MODEL B (9B ST):"
echo "Plain @d512 bw24:" && grep "decode tg128 @ctx512" $LOG_DIR/B-9b-st-plain-d512.log | grep -oP '\d+\.\d+ tok/s'
echo "Plain @d512 llama:" && grep "tg128 @ d512" $LOG_DIR/B-9b-st-plain-d512.log | grep -oP '\d+\.\d+(?= ±)'
echo "Plain @d6257 bw24:" && grep "decode tg128 @ctx6257" $LOG_DIR/B-9b-st-plain-d6257.log | grep -oP '\d+\.\d+ tok/s'
echo "Plain @d6257 llama:" && grep "tg128 @ d6257" $LOG_DIR/B-9b-st-plain-d6257.log | grep -oP '\d+\.\d+(?= ±)'
echo "Spec p1 bw24:" && grep "generate_spec K=2" $LOG_DIR/B-9b-st-spec-p1.log | grep -oP '\d+\.\d+ tok/s'
echo "Spec p2 bw24:" && grep "generate_spec K=2" $LOG_DIR/B-9b-st-spec-p2.log | grep -oP '\d+\.\d+ tok/s'
echo "Spec p3 bw24:" && grep "generate_spec K=2" $LOG_DIR/B-9b-st-spec-p3-sampled.log | grep -oP '\d+\.\d+ tok/s'
echo ""

# Model C: 27B GGUF
echo "MODEL C (27B GGUF):"
echo "Plain @d512 bw24:" && grep "decode tg128 @ctx512" $LOG_DIR/C-27b-gguf-plain.log | grep -oP '\d+\.\d+ tok/s'
echo "Plain @d512 llama:" && grep "tg128 @ d512" $LOG_DIR/C-27b-gguf-plain.log | grep -oP '\d+\.\d+(?= ±)'
echo "Plain @d6257 bw24:" && grep "decode tg128 @ctx6257" $LOG_DIR/C-27b-gguf-plain.log | grep -oP '\d+\.\d+ tok/s'
echo "Plain @d6257 llama:" && grep "tg128 @ d6257" $LOG_DIR/C-27b-gguf-plain.log | grep -oP '\d+\.\d+(?= ±)'
echo "Spec p1 bw24:" && grep "p1-code-short" -A1 $LOG_DIR/C-27b-gguf-spec.log | grep "generate_spec" | grep -oP '\d+\.\d+ tok/s'
echo "Spec p2 bw24:" && grep "p2-code-medium" -A1 $LOG_DIR/C-27b-gguf-spec.log | grep "generate_spec" | grep -oP '\d+\.\d+ tok/s'
echo "Spec p3 bw24:" && grep "p3 SAMPLED" -A1 $LOG_DIR/C-27b-gguf-spec.log | grep "generate_spec" | grep -oP '\d+\.\d+ tok/s'
# llama spec will be filled in when complete
echo ""
