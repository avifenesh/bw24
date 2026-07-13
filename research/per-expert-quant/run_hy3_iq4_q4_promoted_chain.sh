#!/usr/bin/env bash
set -euo pipefail

# Continue only the frozen Pareto leaders from the combined six-format directional frontier through
# the matched practical panel, trusted capability suite, and full SWE/Terminal panels.

ROOT=${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
EXPECTED_COMMIT=${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the detached source commit}
DIRECTIONAL_ROOT=${DIRECTIONAL_ROOT:-/data/results/per-expert-quant/iq4-q4-directional-v1}
DIRECTIONAL_READY=${DIRECTIONAL_READY:-/data/logs/iq4-q4-directional-v1/complete}
PRACTICAL_ROOT=${PRACTICAL_ROOT:-/data/results/per-expert-quant/practical-iq4-q4-v1}
PRACTICAL_LOG=${PRACTICAL_LOG:-/data/logs/practical-iq4-q4-v1}
TRUSTED_ROOT=${TRUSTED_ROOT:-/data/results/per-expert-quant/trusted-full-iq4-q4-v1}
TRUSTED_LOG=${TRUSTED_LOG:-/data/logs/trusted-full-iq4-q4-v1}
AGENTIC_ROOT=${AGENTIC_ROOT:-/data/results/per-expert-quant/full-agentic-iq4-q4-v1}
AGENTIC_LOG=${AGENTIC_LOG:-/data/logs/full-agentic-iq4-q4-v1}
IQ4_ART_ROOT=${IQ4_ART_ROOT:-/scratch/bw24-artifacts-iq4-q4-99f3dc3}

die() { echo "IQ4/Q4 promoted chain: $*" >&2; exit 1; }
mkdir -p "$PRACTICAL_LOG" "$TRUSTED_LOG" "$AGENTIC_LOG"
exec 9>"$PRACTICAL_LOG/chain.lock"
flock -n 9 || die "another promoted chain owns the lock"
[[ $(git -C "$ROOT" rev-parse HEAD) == "$EXPECTED_COMMIT" ]] || die "source commit mismatch"
while [[ ! -f "$DIRECTIONAL_READY" || ! -s "$DIRECTIONAL_ROOT/_active-run-id" ]]; do sleep 30; done
directional_run=$(cat "$DIRECTIONAL_ROOT/_active-run-id")
promotion="$DIRECTIONAL_ROOT/iq4-q4-practical-input-$directional_run.json"
[[ -f "$promotion" ]] || die "missing directional promotion"

BW24_ROOT="$ROOT" PROMOTION="$promotion" PROMOTION_READY="$DIRECTIONAL_READY" \
  GATE_LOCK="$ROOT/research/per-expert-quant/smart100-practical-gates.lock.json" \
  PRACTICAL_LOCK="$ROOT/research/per-expert-quant/practical-evals.lock.json" \
  PRACTICAL_SELECTOR="$ROOT/research/per-expert-quant/select_practical_promotions.py" \
  OUT_ROOT="$PRACTICAL_ROOT" LOG_ROOT="$PRACTICAL_LOG" IQ4_ART_ROOT="$IQ4_ART_ROOT" \
  SERVER_BIN=/data/build/bw24-portable-ada-fix-target/release/bw24-server \
  HARBOR_BIN=/data/bin/harbor-0.18.0-0a01ad6/harbor \
  HARBOR_HOME=/data/cache/harbor-home SPILL_DEPTH=8 \
  "$ROOT/research/per-expert-quant/run_promoted_practical_transition.sh"

BW24_ROOT="$ROOT" PRACTICAL_ROOT="$PRACTICAL_ROOT" PRACTICAL_READY="$PRACTICAL_LOG/complete" \
  OUT_ROOT="$TRUSTED_ROOT" LOG_ROOT="$TRUSTED_LOG" IQ4_ART_ROOT="$IQ4_ART_ROOT" \
  SERVER_BIN=/data/build/bw24-portable-ada-fix-target/release/bw24-server \
  SPILL_DEPTH=8 TASK_ATTEMPTS=3 \
  "$ROOT/research/per-expert-quant/run_trusted_full_transition.sh"

BW24_ROOT="$ROOT" TRUSTED_ROOT="$TRUSTED_ROOT" TRUSTED_READY="$TRUSTED_LOG/complete" \
  OUT_ROOT="$AGENTIC_ROOT" LOG_ROOT="$AGENTIC_LOG" IQ4_ART_ROOT="$IQ4_ART_ROOT" \
  SERVER_BIN=/data/build/bw24-portable-ada-fix-target/release/bw24-server \
  HARBOR_BIN=/data/bin/harbor-0.18.0-0a01ad6/harbor \
  HARBOR_HOME=/data/cache/harbor-home SPILL_DEPTH=8 \
  "$ROOT/research/per-expert-quant/run_full_agentic_transition.sh"

date -u +%FT%TZ | tee "$AGENTIC_LOG/chain-complete"
