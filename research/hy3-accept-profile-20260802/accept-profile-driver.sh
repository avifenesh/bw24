#!/usr/bin/env bash
# Hy3 K=1 acceptance-profile driver — lane/hy3-accept-profile, Mumbai H100.
# One invocation = ONE run-spec process for ONE prompt: plain greedy oracle + spec K=1,
# MEMRA_CHAT=1 (single user turn, serving shape), MEMRA_NGEN=128, greedy (exactness gate on).
# Entire GPU-touching process held under flock /tmp/gpu-h100.lock (shared-box rule).
# Build REUSED from the K-sweep lane: /opt/dlami/nvme/hy3-spec-sweep/memra @ 2b9a6aa6
# (spec.rs / run_spec.rs identical to restructure/public-split tip c654329f).
# Usage: accept-profile-driver.sh <prompt-file> <tag>
set -u
PROMPT=${1:?usage: accept-profile-driver.sh <prompt-file> <tag>}
TAG=${2:?usage: accept-profile-driver.sh <prompt-file> <tag>}
TREE=/opt/dlami/nvme/hy3-spec-sweep/memra
ART=/opt/dlami/nvme/models/hy3-layer103p5-bw24-runtime
BASE=/opt/dlami/nvme/hy3-accept-profile
LOG=$BASE/logs/$TAG.log

{
  echo "=== $TAG start $(date -u +%FT%TZ) ==="
  echo "prompt file: $PROMPT ($(wc -c < "$PROMPT") chars)"
  echo "tree commit: $(cat $TREE/SOURCE-COMMIT.txt)"
  echo "gpu-pre: $(nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm,memory.used --format=csv,noheader)"
  echo "concurrent compute apps pre:"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
} >> "$LOG" 2>&1

flock /tmp/gpu-h100.lock env \
  MEMRA_PROMPT_FILE="$PROMPT" \
  MEMRA_CHAT=1 \
  MEMRA_NGEN=128 \
  MEMRA_SPEC_K=1 \
  "$TREE/target/release/run-spec" "$ART" >> "$LOG" 2>&1
rc=$?

{
  echo "=== $TAG exit=$rc $(date -u +%FT%TZ) ==="
  echo "gpu-post: $(nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm,memory.used --format=csv,noheader)"
  echo "concurrent compute apps post:"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
} >> "$LOG" 2>&1
echo "$TAG-DONE rc=$rc"
exit $rc
