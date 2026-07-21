#!/usr/bin/env bash
set -euo pipefail

# Recover/finalize the overlapping layer-balanced practical run without regenerating answers.
# Raw Harbor receipts remain immutable. The portable summarizer applies the preregistered rule
# that AgentTimeoutError tasks score zero while retaining late verifier rewards as provenance.

ROOT=${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
EXPECTED_COMMIT=${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the detached finalizer commit}
SOURCE_UNIT=${SOURCE_UNIT:-bw24-practical-c8689ec-r3.service}
CANDIDATE_UNIT=${CANDIDATE_UNIT:-bw24-layer-balanced100-practical-0a1446f-r2.service}
SOURCE_ROOT=${SOURCE_ROOT:-/data/results/per-expert-quant/practical-iq3-iq4-q4-pareto-v1}
OUT_ROOT=${OUT_ROOT:-/data/results/per-expert-quant/practical-layer-balanced100-v1}
LOG_ROOT=${LOG_ROOT:-/data/logs/practical-layer-balanced100-v1}
SUMMARIZER=${SUMMARIZER:-$ROOT/research/per-expert-quant/summarize_practical_results.py}
SELECTOR=${SELECTOR:-$ROOT/research/per-expert-quant/select_practical_promotions.py}
WAIT_INTERVAL_S=${WAIT_INTERVAL_S:-30}

die() { echo "layer-balanced100 practical finalizer: $*" >&2; exit 2; }
[[ $(git -C "$ROOT" rev-parse HEAD) == "$EXPECTED_COMMIT" ]] || die "source commit mismatch"
[[ -z $(git -C "$ROOT" status --porcelain) ]] || die "source checkout is dirty"
[[ -f "$SUMMARIZER" && -f "$SELECTOR" ]] || die "missing finalizer tools"
mkdir -p "$LOG_ROOT"
exec 9>"$LOG_ROOT/timeout-normalized-finalizer.lock"
flock -n 9 || die "another timeout-normalized finalizer owns the lock"
exec >>"$LOG_ROOT/timeout-normalized-finalizer.log" 2>&1
echo "$(date -u +%FT%TZ) timeout-normalized layer-balanced100 finalizer waiting"

while [[ ! -s "$SOURCE_ROOT/_active-run-id" || ! -s "$OUT_ROOT/_active-run-id" ]]; do
  sleep "$WAIT_INTERVAL_S"
done
SOURCE_RUN_ID=$(<"$SOURCE_ROOT/_active-run-id")
RUN_ID=$(<"$OUT_ROOT/_active-run-id")
RUN_CONFIG="$OUT_ROOT/run-configs/$RUN_ID.json"
[[ -f "$RUN_CONFIG" ]] || die "missing candidate run config"

while [[ $(systemctl --user is-active "$SOURCE_UNIT" 2>/dev/null || true) == active \
  || $(systemctl --user is-active "$CANDIDATE_UNIT" 2>/dev/null || true) == active ]]; do
  sleep "$WAIT_INTERVAL_S"
done
PRACTICAL_PROMOTION="$OUT_ROOT/practical-promotion-$RUN_ID.json"
if [[ -f "$LOG_ROOT/complete" && -f "$PRACTICAL_PROMOTION" ]]; then
  echo "layer-balanced100 practical transition already completed"
  exit 0
fi

mapfile -t resolved < <(python3 - "$RUN_CONFIG" "$SOURCE_RUN_ID" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
if d.get("format") != "bw24-layer-balanced100-practical-reuse-run-v1":
    raise SystemExit("wrong candidate practical config")
if d.get("candidate") != "layer_balanced100" or d.get("candidate_selected") is not True:
    raise SystemExit("layer-balanced100 was not selected")
if d.get("source_practical",{}).get("run_id") != sys.argv[2]:
    raise SystemExit("source practical run differs")
print(d["directional_promotion"]["path"])
print(d["gate_lock"]["path"])
print(d["practical_lock"]["path"])
PY
)
(( ${#resolved[@]} == 3 )) || die "candidate config resolution failed"
PROMOTION=${resolved[0]}
GATE_LOCK=${resolved[1]}
PRACTICAL_LOCK=${resolved[2]}
for path in "$PROMOTION" "$GATE_LOCK" "$PRACTICAL_LOCK"; do
  [[ -f "$path" ]] || die "missing protocol file $path"
done

for arm in plain_quant traffic_nvfp4_53_q2_139; do
  for panel in swe terminal; do
    receipt="$SOURCE_ROOT/$arm/$panel/$SOURCE_RUN_ID/run-metadata.json"
    [[ -f "$receipt" ]] || die "missing source receipt $receipt"
  done
done
for panel in swe terminal; do
  receipt="$OUT_ROOT/layer_balanced100/$panel/$RUN_ID/run-metadata.json"
  [[ -f "$receipt" ]] || die "missing candidate receipt $receipt"
done

COMPARE_ROOT="$OUT_ROOT/comparisons-timeout-normalized/$RUN_ID"
mkdir -p "$COMPARE_ROOT"
for baseline in plain_quant traffic_nvfp4_53_q2_139; do
  for panel in swe terminal; do
    json_out="$COMPARE_ROOT/$baseline-vs-layer_balanced100.$panel.json"
    markdown_out="$COMPARE_ROOT/$baseline-vs-layer_balanced100.$panel.md"
    [[ ! -e "$json_out" && ! -e "$markdown_out" ]] \
      || die "refusing existing timeout-normalized comparison $baseline/$panel"
    python3 "$SUMMARIZER" \
      --baseline "$SOURCE_ROOT/$baseline/$panel/$SOURCE_RUN_ID" \
      --candidate "$OUT_ROOT/layer_balanced100/$panel/$RUN_ID" \
      --panel "$panel" --lock "$PRACTICAL_LOCK" \
      --json-out "$json_out" --markdown-out "$markdown_out"
  done
done

[[ ! -e "$PRACTICAL_PROMOTION" ]] || die "refusing existing practical promotion"
python3 "$SELECTOR" --promotion "$PROMOTION" --gate-lock "$GATE_LOCK" \
  --comparison-root "$COMPARE_ROOT" --output "$PRACTICAL_PROMOTION"
COMPARISON_INVENTORY="$LOG_ROOT/$RUN_ID-timeout-normalized-comparisons.sha256"
RUN_INVENTORY="$LOG_ROOT/$RUN_ID-timeout-normalized-runs.sha256"
find "$COMPARE_ROOT" -type f -print0 | sort -z | xargs -0 sha256sum >"$COMPARISON_INVENTORY"
find \
  "$SOURCE_ROOT/plain_quant/swe/$SOURCE_RUN_ID" \
  "$SOURCE_ROOT/plain_quant/terminal/$SOURCE_RUN_ID" \
  "$SOURCE_ROOT/traffic_nvfp4_53_q2_139/swe/$SOURCE_RUN_ID" \
  "$SOURCE_ROOT/traffic_nvfp4_53_q2_139/terminal/$SOURCE_RUN_ID" \
  "$OUT_ROOT/layer_balanced100/swe/$RUN_ID" \
  "$OUT_ROOT/layer_balanced100/terminal/$RUN_ID" \
  -type f -print0 | sort -z | xargs -0 sha256sum >"$RUN_INVENTORY"
FINAL_EVIDENCE="$LOG_ROOT/$RUN_ID-timeout-normalized-finalizer-evidence.sha256"
sha256sum "$RUN_CONFIG" "$PROMOTION" "$GATE_LOCK" "$PRACTICAL_LOCK" \
  "$SUMMARIZER" "$SELECTOR" "$COMPARISON_INVENTORY" "$RUN_INVENTORY" \
  "$PRACTICAL_PROMOTION" >"$FINAL_EVIDENCE"
sha256sum -c "$FINAL_EVIDENCE"
date -u +%FT%TZ | tee "$LOG_ROOT/complete"
