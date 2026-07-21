#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
EXPECTED_COMMIT=${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the detached source commit}
PRACTICAL_ROOT=${PRACTICAL_ROOT:-/data/results/per-expert-quant/practical-smart100-v1}
PRACTICAL_READY=${PRACTICAL_READY:-/data/logs/practical-smart100-v1/final-complete}
OUT_ROOT=${OUT_ROOT:-/data/results/per-expert-quant/trusted-full-smart100-v1}
LOG_ROOT=${LOG_ROOT:-/data/logs/trusted-full-smart100-v1}

[[ $(git -C "$ROOT" rev-parse HEAD) == "$EXPECTED_COMMIT" ]] || {
  echo "smart trusted transition: source commit mismatch" >&2; exit 2;
}
PRACTICAL_ROOT="$PRACTICAL_ROOT" PRACTICAL_READY="$PRACTICAL_READY" \
  OUT_ROOT="$OUT_ROOT" LOG_ROOT="$LOG_ROOT" \
  SERVER_BIN=/data/build/bw24-portable-ada-fix-target/release/bw24-server \
  SPILL_DEPTH=8 TASK_ATTEMPTS=3 \
  "$ROOT/research/per-expert-quant/run_trusted_full_transition.sh"
