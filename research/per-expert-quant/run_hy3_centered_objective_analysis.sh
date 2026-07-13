#!/usr/bin/env bash
set -euo pipefail

# Re-solve the frozen seven-format private allocation with an algebraically equivalent centered
# objective.  This is analysis-only: it never reads public results, heals weights, or builds an
# artifact.  A materially different allocation can be promoted to a separate build only after its
# receipt is compared with the already-frozen uncentered candidate.

ROOT=${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
PY=${PY:-python3}
EXPECTED_COMMIT=${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the detached analysis commit}
SENSITIVITY=${SENSITIVITY:-/data/calibration/hy3-quant-iq3-iq4-q4-99f3dc3/seven-format-sensitivity.json}
RETENTION=${RETENTION:-/data/calibration/hy3-100gb-5f02c37/expert-retention-scores.json}
CONFIDENCE=${CONFIDENCE:-/data/calibration/hy3-100gb-5f02c37/confidence-expert-scores-13a4d92.json}
REFERENCE_PLAN=${REFERENCE_PLAN:-/data/plans/per-expert-quant-100gb-5f02c37/traffic-nvfp4-53-q2-139-exact100gb.json}
REFERENCE_RECEIPTS=${REFERENCE_RECEIPTS:-/data/heal/per-expert-quant-100gb-5f02c37/joint/receipts}
UNCENTERED_PLAN=${UNCENTERED_PLAN:-/data/plans/per-expert-quant-iq3-iq4-q4-99f3dc3/smart100_iq3_iq4_q4_empirical.json}
OUT_ROOT=${OUT_ROOT:-/data/analysis/per-expert-quant-iq3-iq4-q4-centered}
TARGET_BYTES=${TARGET_BYTES:-100000000000}
ARM=${ARM:-smart100_iq3_iq4_q4_centered}

die() { echo "centered objective analysis: $*" >&2; exit 1; }
[[ $(git -C "$ROOT" rev-parse HEAD) == "$EXPECTED_COMMIT" ]] || die "source commit mismatch"
mkdir -p "$OUT_ROOT"
exec 9>"$OUT_ROOT/analysis.lock"
flock -n 9 || die "another centered analysis owns the lock"
for path in "$RETENTION" "$CONFIDENCE" "$REFERENCE_PLAN"; do
  [[ -f "$path" ]] || die "missing input $path"
done
[[ -d "$REFERENCE_RECEIPTS" ]] || die "missing joint-heal receipts"
while [[ ! -f "$SENSITIVITY" || ! -f "$UNCENTERED_PLAN" ]]; do sleep 30; done

plan="$OUT_ROOT/$ARM.json"
comparison="$OUT_ROOT/allocation-comparison.json"
receipt="$OUT_ROOT/allocation-comparison.receipt.json"
for path in "$plan" "$comparison" "$receipt" "$OUT_ROOT/evidence.sha256" "$OUT_ROOT/complete"; do
  [[ ! -e "$path" ]] || die "refusing existing output $path"
done

"$PY" "$ROOT/tools/build_hy3_smart_budget_plan.py" --self-test
"$PY" "$ROOT/tools/summarize_hy3_smart_allocations.py" --self-test
"$PY" "$ROOT/tools/build_hy3_smart_budget_plan.py" \
  --retention-scores "$RETENTION" --quant-sensitivity "$SENSITIVITY" \
  --confidence-plan "$CONFIDENCE" --joint-receipts "$REFERENCE_RECEIPTS" \
  --reference-plan "$REFERENCE_PLAN" --target-logical-bytes "$TARGET_BYTES" \
  --min-survivors-per-layer 96 --retention-weight 0 --confidence-weight 0 --layer-weight 0 \
  --time-limit-seconds 1800 --mip-rel-gap 1e-6 --out "$plan"

"$PY" - "$plan" "$TARGET_BYTES" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
assert d["calibration"]["public_eval_data_used_for_selection"] is False
assert d["policy"]["result_logical_bytes"] <= int(sys.argv[2])
assert min(x["retained"] for x in d["layer_summary"].values()) >= 96
assert set(d["policy"]["candidate_qtypes"]) == {
    "Q8_0", "NVFP4", "IQ3_S", "IQ4_XS", "Q4_K", "Q3_K", "Q2_K"
}
c=d["policy"]["objective_centering"]
assert c["method"] == "per_expert_projection_minimum"
assert c["preserves_exact_argmin"] is True
assert c["constant_offset"] > 0 and c["centered_scale"] > 0
assert d["selection"]["estimated_absolute_output_damage"] >= c["constant_offset"]
PY

"$PY" "$ROOT/tools/summarize_hy3_smart_allocations.py" \
  "$REFERENCE_PLAN" "$UNCENTERED_PLAN" "$plan" \
  --analysis-commit "$EXPECTED_COMMIT" --out "$comparison" --receipt "$receipt"
sha256sum "$SENSITIVITY" "$REFERENCE_PLAN" "$UNCENTERED_PLAN" "$plan" \
  "$comparison" "$receipt" "$ROOT/tools/build_hy3_smart_budget_plan.py" \
  >"$OUT_ROOT/evidence.sha256"
sha256sum -c "$OUT_ROOT/evidence.sha256"
date -u +%FT%TZ | tee "$OUT_ROOT/complete"
