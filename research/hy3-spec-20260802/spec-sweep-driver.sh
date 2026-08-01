#!/usr/bin/env bash
# Hy3 MTP spec K-sweep driver — lane/hy3-spec-sweep (gaps.md G2 closure), Mumbai H100.
# One invocation = ONE run-spec process: plain-generate oracle + K=1..8 spec battery,
# single model load (primed state reused within the process), board-d1736 RAW prompt
# (baseline.md protocol: no chat template), MEMRA_NGEN=128.
# Entire GPU-touching process held under flock /tmp/gpu-h100.lock (shared-box rule).
# Usage: spec-sweep-driver.sh <r1|r2|r3>
set -u
RUN=${1:?usage: spec-sweep-driver.sh <run-tag>}
TREE=/opt/dlami/nvme/hy3-spec-sweep/memra
ART=/opt/dlami/nvme/models/hy3-layer103p5-bw24-runtime
PROMPT=$TREE/research/gemma4-bringup/depth-prompt-1736.txt
LOG=/opt/dlami/nvme/hy3-spec-sweep/logs/sweep-$RUN.log

{
  echo "=== sweep $RUN start $(date -u +%FT%TZ) ==="
  echo "tree commit: $(cat $TREE/SOURCE-COMMIT.txt)"
  echo "gpu-pre: $(nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm,memory.used --format=csv,noheader)"
  echo "concurrent compute apps pre:"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
} >> "$LOG" 2>&1

flock /tmp/gpu-h100.lock env \
  MEMRA_PROMPT_FILE="$PROMPT" \
  MEMRA_NGEN=128 \
  "$TREE/target/release/run-spec" "$ART" >> "$LOG" 2>&1
rc=$?

{
  echo "=== sweep $RUN exit=$rc $(date -u +%FT%TZ) ==="
  echo "gpu-post: $(nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm,memory.used --format=csv,noheader)"
  echo "concurrent compute apps post:"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
} >> "$LOG" 2>&1
echo "SWEEP-$RUN-DONE rc=$rc"
