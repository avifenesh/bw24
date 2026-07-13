#!/usr/bin/env bash
set -euo pipefail

# Build the privately superior centered seven-format plan without repeating sensitivity.  The
# original lane JSON files are immutable measurement inputs; this transition hard-links them into
# a fresh build root, verifies the damage receipt, and delegates heal/repack validation to the same
# frozen extension pipeline used by the uncentered candidate.

ROOT=${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
EXPECTED_COMMIT=${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the detached build commit}
PY=${PY:-python3}
ORIGINAL_READY=${ORIGINAL_READY:-/data/logs/iq3-iq4-q4-extension-99f3dc3/complete}
ORIGINAL_ROOT=${ORIGINAL_ROOT:-/data/calibration/hy3-quant-iq3-iq4-q4-99f3dc3}
CENTERED_ANALYSIS=${CENTERED_ANALYSIS:-/data/analysis/per-expert-quant-iq3-iq4-q4-centered-a7200c0}
ARM=${ARM:-smart100_iq3_iq4_q4_centered}
OUT_ROOT=${OUT_ROOT:-/data/calibration/hy3-quant-iq3-iq4-q4-centered-0f98d7d}
PLAN_ROOT=${PLAN_ROOT:-/data/plans/per-expert-quant-iq3-iq4-q4-centered-0f98d7d}
HEAL_ROOT=${HEAL_ROOT:-/data/heal/per-expert-quant-iq3-iq4-q4-centered-0f98d7d}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-/data/artifacts/per-expert-quant-iq3-iq4-q4-centered-0f98d7d}
SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/bw24-artifacts-iq3-iq4-q4-centered-0f98d7d}
LOG_ROOT=${LOG_ROOT:-/data/logs/iq3-iq4-q4-centered-0f98d7d}
GGML_LIB=${GGML_LIB:-/data/build/llama-cpp-99f3dc3-cpu/bin/libggml-base.so.0.16.0}
GGML_LIB_SHA256=${GGML_LIB_SHA256:-a4e5f72ae34a6c4860738e16d0432e91e773df1a080ef1dab3a5f4ddcb22ea06}
GGML_SOURCE_COMMIT=${GGML_SOURCE_COMMIT:-99f3dc32296f825fec94f202da1e9fede1e78cf9}

die() { echo "centered candidate transition: $*" >&2; exit 1; }
[[ $(git -C "$ROOT" rev-parse HEAD) == "$EXPECTED_COMMIT" ]] || die "source commit mismatch"
mkdir -p "$LOG_ROOT"
exec 9>"$LOG_ROOT/centered-transition.lock"
flock -n 9 || die "another centered build transition owns the lock"
while [[ ! -f "$ORIGINAL_READY" || ! -f "$CENTERED_ANALYSIS/complete" ]]; do sleep 30; done

source_plan="$CENTERED_ANALYSIS/$ARM.json"
damage="$CENTERED_ANALYSIS/private-damage-comparison.json"
damage_receipt="$CENTERED_ANALYSIS/private-damage-comparison.receipt.json"
for path in "$source_plan" "$damage" "$damage_receipt" "$CENTERED_ANALYSIS/evidence.sha256" \
  "$ORIGINAL_ROOT/seven-format-sensitivity.json"; do
  [[ -f "$path" ]] || die "missing prerequisite $path"
done
sha256sum -c "$CENTERED_ANALYSIS/evidence.sha256" >"$LOG_ROOT/centered-evidence-check.log"
"$PY" - "$damage" "$damage_receipt" "$source_plan" \
  "$ORIGINAL_ROOT/seven-format-sensitivity.json" <<'PY'
import hashlib,json,pathlib,sys

def sha(path): return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
damage_path,receipt_path,plan_path,sensitivity_path=sys.argv[1:]
d=json.load(open(damage_path)); r=json.load(open(receipt_path))
assert d["format"] == "bw24-hy3-quant-plan-damage-v1"
assert d["public_eval_data_used"] is False
assert d["lowest_private_damage_plan"] == "centered"
assert d["plans"]["centered"]["sha256"] == sha(plan_path)
assert d["sensitivity"]["sha256"] == sha(sensitivity_path)
pair=d["pairwise"]["uncentered__centered"]
assert pair["right_minus_left_damage"] < 0
assert d["plans"]["centered"]["logical_bytes"] <= 100_000_000_000
assert r["format"] == "bw24-hy3-quant-plan-damage-receipt-v1"
assert r["public_eval_data_used"] is False
assert r["output"]["sha256"] == sha(damage_path)
assert r["sensitivity"]["sha256"] == sha(sensitivity_path)
assert any(x["name"] == "centered" and x["sha256"] == sha(plan_path) for x in r["plans"])
PY

for path in "$OUT_ROOT" "$PLAN_ROOT" "$HEAL_ROOT" "$ARTIFACT_ROOT" "$SCRATCH_ROOT"; do
  [[ ! -e "$path" ]] || die "refusing existing build path $path"
done
mkdir -p "$OUT_ROOT/lanes" "$PLAN_ROOT"
mapfile -t lanes < <(find "$ORIGINAL_ROOT/lanes" -maxdepth 1 -type f -name 'lane-*.json' | sort -V)
[[ ${#lanes[@]} == 24 ]] || die "expected 24 immutable sensitivity lanes"
for lane in "${lanes[@]}"; do
  ln "$lane" "$OUT_ROOT/lanes/$(basename "$lane")"
done
cp --reflink=auto "$source_plan" "$PLAN_ROOT/$ARM.json"
cmp "$source_plan" "$PLAN_ROOT/$ARM.json"

env ROOT="$ROOT" EXPECTED_COMMIT="$EXPECTED_COMMIT" PY="$PY" \
  GGML_LIB="$GGML_LIB" GGML_LIB_SHA256="$GGML_LIB_SHA256" \
  GGML_SOURCE_COMMIT="$GGML_SOURCE_COMMIT" BASE_BUILD_COMPLETE="$ORIGINAL_READY" \
  OUT_ROOT="$OUT_ROOT" PLAN_ROOT="$PLAN_ROOT" HEAL_ROOT="$HEAL_ROOT" \
  ARTIFACT_ROOT="$ARTIFACT_ROOT" SCRATCH_ROOT="$SCRATCH_ROOT" LOG_ROOT="$LOG_ROOT/build" \
  GPUS_CSV=1,3,4,5,6,7 LANES_PER_GPU=4 ARM="$ARM" \
  "$ROOT/research/per-expert-quant/run_hy3_iq4_q4_extension_transition.sh"

cmp "$ORIGINAL_ROOT/seven-format-sensitivity.json" "$OUT_ROOT/seven-format-sensitivity.json"
sha256sum "$CENTERED_ANALYSIS/evidence.sha256" "$source_plan" \
  "$OUT_ROOT/seven-format-sensitivity.json" "$PLAN_ROOT/$ARM.json" \
  "$ARTIFACT_ROOT/$ARM/manifest.json" "$LOG_ROOT/build/evidence.sha256" \
  >"$LOG_ROOT/evidence.sha256"
sha256sum -c "$LOG_ROOT/evidence.sha256"
date -u +%FT%TZ | tee "$LOG_ROOT/complete"
