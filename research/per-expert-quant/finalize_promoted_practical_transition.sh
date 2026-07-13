#!/usr/bin/env bash
set -euo pipefail

# Finalize an immutable promoted-practical generation run with the portable loopback comparator.
# This is also the recovery path for runs generated before unique per-GPU ports were normalized.

ROOT=${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
EXPECTED_COMMIT=${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the detached finalizer commit}
SOURCE_UNIT=${SOURCE_UNIT:-bw24-practical-c8689ec-r3.service}
OUT_ROOT=${OUT_ROOT:-/data/results/per-expert-quant/practical-iq3-iq4-q4-pareto-v1}
LOG_ROOT=${LOG_ROOT:-/data/logs/practical-iq3-iq4-q4-pareto-v1}
SUMMARIZER=${SUMMARIZER:-$ROOT/research/per-expert-quant/summarize_practical_results.py}
SELECTOR=${SELECTOR:-$ROOT/research/per-expert-quant/select_practical_promotions.py}
WAIT_INTERVAL_S=${WAIT_INTERVAL_S:-30}

die() { echo "promoted practical finalizer: $*" >&2; exit 2; }
[[ $(git -C "$ROOT" rev-parse HEAD) == "$EXPECTED_COMMIT" ]] || die "source commit mismatch"
[[ -z $(git -C "$ROOT" status --porcelain) ]] || die "source checkout is dirty"
[[ -f "$SUMMARIZER" && -f "$SELECTOR" ]] || die "missing finalizer tools"
mkdir -p "$LOG_ROOT"
exec 9>"$LOG_ROOT/portable-finalizer.lock"
flock -n 9 || die "another portable practical finalizer owns the lock"
exec >>"$LOG_ROOT/portable-finalizer.log" 2>&1
echo "$(date -u +%FT%TZ) portable practical finalizer waiting for generation unit"

while [[ ! -s "$OUT_ROOT/_active-run-id" ]]; do sleep "$WAIT_INTERVAL_S"; done
RUN_ID=$(<"$OUT_ROOT/_active-run-id")
RUN_CONFIG="$OUT_ROOT/run-configs/$RUN_ID.json"
[[ -f "$RUN_CONFIG" ]] || die "missing source practical run config"
while [[ $(systemctl --user is-active "$SOURCE_UNIT" 2>/dev/null || true) == active ]]; do
  sleep "$WAIT_INTERVAL_S"
done
if [[ -f "$LOG_ROOT/complete" && -f "$OUT_ROOT/practical-promotion-$RUN_ID.json" ]]; then
  echo "source practical transition already completed"
  exit 0
fi

mapfile -t resolved < <(python3 - "$RUN_CONFIG" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
if d.get("format") != "bw24-practical-transition-run-v1":
    raise SystemExit("wrong practical run config")
arms=d.get("arms")
if not isinstance(arms,list) or not 2 <= len(arms) <= 4 or len(set(arms)) != len(arms):
    raise SystemExit("invalid practical arms")
if arms[:2] != ["plain_quant","traffic_nvfp4_53_q2_139"]:
    raise SystemExit("practical references differ")
print(d["directional_promotion"]["path"])
print(d["gate_lock"]["path"])
print(d["practical_lock"]["path"])
print("\t".join(arms))
PY
)
(( ${#resolved[@]} == 4 )) || die "run config resolution failed"
PROMOTION=${resolved[0]}
GATE_LOCK=${resolved[1]}
PRACTICAL_LOCK=${resolved[2]}
IFS=$'\t' read -r -a ARMS <<<"${resolved[3]}"
for path in "$PROMOTION" "$GATE_LOCK" "$PRACTICAL_LOCK"; do
  [[ -f "$path" ]] || die "missing source protocol file $path"
done

COMPARE_ROOT="$OUT_ROOT/comparisons-portable/$RUN_ID"
mkdir -p "$COMPARE_ROOT"
for baseline in "${ARMS[@]:0:2}"; do
  for candidate in "${ARMS[@]}"; do
    [[ "$baseline" != "$candidate" ]] || continue
    for panel in swe terminal; do
      json_out="$COMPARE_ROOT/$baseline-vs-$candidate.$panel.json"
      markdown_out="$COMPARE_ROOT/$baseline-vs-$candidate.$panel.md"
      [[ ! -e "$json_out" && ! -e "$markdown_out" ]] \
        || die "refusing partial portable comparison $baseline/$candidate/$panel"
      python3 "$SUMMARIZER" \
        --baseline "$OUT_ROOT/$baseline/$panel/$RUN_ID" \
        --candidate "$OUT_ROOT/$candidate/$panel/$RUN_ID" \
        --panel "$panel" --lock "$PRACTICAL_LOCK" \
        --json-out "$json_out" --markdown-out "$markdown_out"
    done
  done
done

PRACTICAL_PROMOTION="$OUT_ROOT/practical-promotion-$RUN_ID.json"
[[ ! -e "$PRACTICAL_PROMOTION" ]] || die "refusing existing practical promotion"
python3 "$SELECTOR" --promotion "$PROMOTION" --gate-lock "$GATE_LOCK" \
  --comparison-root "$COMPARE_ROOT" --output "$PRACTICAL_PROMOTION"
find "$COMPARE_ROOT" -type f -print0 | sort -z | xargs -0 sha256sum \
  >"$LOG_ROOT/$RUN_ID-portable-comparisons.sha256"
find "$OUT_ROOT" -path "*/$RUN_ID/*" -type f -print0 | sort -z | xargs -0 sha256sum \
  >"$LOG_ROOT/$RUN_ID-generation-evidence.sha256"
sha256sum "$RUN_CONFIG" "$PROMOTION" "$GATE_LOCK" "$PRACTICAL_LOCK" \
  "$PRACTICAL_PROMOTION" "$LOG_ROOT/$RUN_ID-portable-comparisons.sha256" \
  "$LOG_ROOT/$RUN_ID-generation-evidence.sha256" \
  >"$LOG_ROOT/$RUN_ID-portable-finalizer-evidence.sha256"
sha256sum -c "$LOG_ROOT/$RUN_ID-portable-finalizer-evidence.sha256"
date -u +%FT%TZ | tee "$LOG_ROOT/complete"
