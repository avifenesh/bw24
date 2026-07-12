#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
EXPECTED_COMMIT=${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the detached source commit}
TRUSTED_ROOT=${TRUSTED_ROOT:-/data/results/per-expert-quant/trusted-full-smart100-v1}
TRUSTED_READY=${TRUSTED_READY:-/data/logs/trusted-full-smart100-v1/complete}
OUT_ROOT=${OUT_ROOT:-/data/results/per-expert-quant/full-agentic-smart100-v1}
LOG_ROOT=${LOG_ROOT:-/data/logs/full-agentic-smart100-v1}

[[ $(git -C "$ROOT" rev-parse HEAD) == "$EXPECTED_COMMIT" ]] || {
  echo "smart full-agentic transition: source commit mismatch" >&2; exit 2;
}
TRUSTED_ROOT="$TRUSTED_ROOT" TRUSTED_READY="$TRUSTED_READY" \
  OUT_ROOT="$OUT_ROOT" LOG_ROOT="$LOG_ROOT" \
  SERVER_BIN=/data/build/bw24-portable-ada-fix-target/release/bw24-server \
  HARBOR_BIN=/data/bin/harbor-0.18.0-0a01ad6/harbor \
  HARBOR_HOME=/data/cache/harbor-home SPILL_DEPTH=8 \
  "$ROOT/research/per-expert-quant/run_full_agentic_transition.sh"
