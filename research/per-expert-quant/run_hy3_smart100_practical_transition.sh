#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
EXPECTED_COMMIT=${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the detached source commit}
DIRECTIONAL_ROOT=${DIRECTIONAL_ROOT:-/data/results/per-expert-quant/smart100-directional-v1}
DIRECTIONAL_COMPLETE=${DIRECTIONAL_COMPLETE:-/data/logs/smart100-directional-v1/complete}
OUT_ROOT=${OUT_ROOT:-/data/results/per-expert-quant/practical-smart100-v1}
LOG_ROOT=${LOG_ROOT:-/data/logs/practical-smart100-v1}

die() { echo "smart100 practical transition: $*" >&2; exit 1; }
mkdir -p "$LOG_ROOT"
exec 9>"$LOG_ROOT/watcher.lock"
flock -n 9 || die "another smart practical watcher owns the lock"
echo "$(date -u +%FT%TZ) smart100 practical watcher started" | tee -a "$LOG_ROOT/watcher.log"
[[ $(git -C "$ROOT" rev-parse HEAD) == "$EXPECTED_COMMIT" ]] || die "source commit mismatch"
while [[ ! -f "$DIRECTIONAL_COMPLETE" ]]; do sleep 30; done
run_id=$(cat "$DIRECTIONAL_ROOT/_active-run-id")
promotion="$DIRECTIONAL_ROOT/smart100-practical-input-$run_id.json"
frontier="$DIRECTIONAL_ROOT/smart100-frontier-$run_id.json"
[[ -f "$promotion" && -f "$frontier" ]] || die "directional promotion evidence is incomplete"

PROMOTION="$promotion" PROMOTION_READY="$DIRECTIONAL_COMPLETE" \
  GATE_LOCK="$ROOT/research/per-expert-quant/smart100-practical-gates.lock.json" \
  PRACTICAL_LOCK="$ROOT/research/per-expert-quant/practical-evals.lock.json" \
  PRACTICAL_SELECTOR="$ROOT/research/per-expert-quant/select_practical_promotions.py" \
  OUT_ROOT="$OUT_ROOT" LOG_ROOT="$LOG_ROOT" \
  SERVER_BIN=/data/build/bw24-portable-ada-fix-target/release/bw24-server \
  HARBOR_BIN=/data/bin/harbor-0.18.0-0a01ad6/harbor \
  HARBOR_HOME=/data/cache/harbor-home SPILL_DEPTH=8 \
  "$ROOT/research/per-expert-quant/run_promoted_practical_transition.sh"

practical_run=$(cat "$OUT_ROOT/_active-run-id")
[[ -f "$OUT_ROOT/practical-promotion-$practical_run.json" ]] || die "missing practical promotion"
sha256sum "$promotion" "$frontier" "$OUT_ROOT/practical-promotion-$practical_run.json" \
  >"$LOG_ROOT/final-evidence.sha256"
date -u +%FT%TZ | tee "$LOG_ROOT/final-complete"
