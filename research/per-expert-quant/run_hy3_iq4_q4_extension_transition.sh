#!/usr/bin/env bash
set -euo pipefail

# Measure exact upstream IQ3_S/IQ4_XS/Q4_K bytes on the frozen private calibration corpus, merge
# those measurements with the immutable four-format map, and build one healed exact-100GB candidate.
# The default odd GPU lanes can overlap the existing even-GPU directional screen without sharing
# a GPU.  Public capability data is not read anywhere in this transition.

ROOT=${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
PY=${PY:-python3}
EXPECTED_COMMIT=${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the detached source commit}
GGML_LIB=${GGML_LIB:?set GGML_LIB to the pinned libggml-base shared library}
GGML_LIB_SHA256=${GGML_LIB_SHA256:?set GGML_LIB_SHA256 to the exact library SHA-256}
GGML_SOURCE_COMMIT=${GGML_SOURCE_COMMIT:?set GGML_SOURCE_COMMIT to the full llama.cpp commit}
BASE_BUILD_COMPLETE=${BASE_BUILD_COMPLETE:-/data/logs/smart100-build-2605fde/complete}
BASE_SENSITIVITY=${BASE_SENSITIVITY:-/data/calibration/hy3-quant-sensitivity-53de6ca/quant-sensitivity.json}
CALIBRATION=${CALIBRATION:-/data/calibration/hy3-100gb-5f02c37}
REQUESTS=${REQUESTS:-/data/calibration/hy3-confidence-v1/requests.jsonl}
SOURCE=${SOURCE:-/opt/dlami/nvme/models/hy3-source}
RETENTION=${RETENTION:-/data/calibration/hy3-100gb-5f02c37/expert-retention-scores.json}
CONFIDENCE=${CONFIDENCE:-/data/calibration/hy3-100gb-5f02c37/confidence-expert-scores-13a4d92.json}
REFERENCE_PLAN=${REFERENCE_PLAN:-/data/plans/per-expert-quant-100gb-5f02c37/traffic-nvfp4-53-q2-139-exact100gb.json}
REFERENCE_RECEIPTS=${REFERENCE_RECEIPTS:-/data/heal/per-expert-quant-100gb-5f02c37/joint/receipts}
OUT_ROOT=${OUT_ROOT:-/data/calibration/hy3-quant-iq3-iq4-q4-99f3dc3}
PLAN_ROOT=${PLAN_ROOT:-/data/plans/per-expert-quant-iq3-iq4-q4-99f3dc3}
HEAL_ROOT=${HEAL_ROOT:-/data/heal/per-expert-quant-iq3-iq4-q4-99f3dc3}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-/data/artifacts/per-expert-quant-iq3-iq4-q4-99f3dc3}
SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/bw24-artifacts-iq3-iq4-q4-99f3dc3}
LOG_ROOT=${LOG_ROOT:-/data/logs/iq3-iq4-q4-extension-99f3dc3}
TARGET_BYTES=${TARGET_BYTES:-100000000000}
GPUS_CSV=${GPUS_CSV:-1,3,5,7}
LANES_PER_GPU=${LANES_PER_GPU:-1}
ARM=${ARM:-smart100_iq3_iq4_q4_empirical}

SCORER="$ROOT/tools/build_hy3_quant_sensitivity.py"
MERGER="$ROOT/tools/merge_hy3_quant_sensitivity.py"
SUMMARIZER="$ROOT/tools/summarize_hy3_quant_effects.py"
ALLOCATION_SUMMARIZER="$ROOT/tools/summarize_hy3_smart_allocations.py"
PLAN_BUILDER="$ROOT/tools/build_hy3_smart_budget_plan.py"
HEALER="$ROOT/tools/heal_hy3_pruned_layer.py"
REPACKER="$ROOT/tools/prepare_mixed_expert_repack.py"
VALIDATOR="$ROOT/research/per-expert-quant/validate_artifact.py"

die() { echo "IQ4/Q4 extension transition: $*" >&2; exit 1; }
mkdir -p "$OUT_ROOT/lanes" "$OUT_ROOT/importance" "$PLAN_ROOT" "$HEAL_ROOT" \
  "$ARTIFACT_ROOT" "$SCRATCH_ROOT" "$LOG_ROOT"
exec 9>"$LOG_ROOT/transition.lock"
flock -n 9 || die "another transition owns $LOG_ROOT/transition.lock"
echo "$(date -u +%FT%TZ) IQ4/Q4 extension transition started" | tee -a "$LOG_ROOT/transition.log"

[[ $(git -C "$ROOT" rev-parse HEAD) == "$EXPECTED_COMMIT" ]] || die "source commit mismatch"
[[ "$GGML_SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "invalid ggml source commit"
[[ "$GGML_LIB_SHA256" =~ ^[0-9a-f]{64}$ ]] || die "invalid ggml library SHA-256"
[[ -f "$GGML_LIB" ]] || die "missing ggml library $GGML_LIB"
[[ $(sha256sum "$GGML_LIB" | cut -d' ' -f1) == "$GGML_LIB_SHA256" ]] \
  || die "ggml library SHA-256 mismatch"
for path in "$BASE_SENSITIVITY" "$CALIBRATION/moe-inputs.lock.json" \
  "$CALIBRATION/routes-weighted.trace" "$REQUESTS" "$SOURCE/config.json" \
  "$SOURCE/model.safetensors.index.json" "$RETENTION" "$CONFIDENCE" \
  "$REFERENCE_PLAN"; do
  [[ -f "$path" ]] || die "missing input $path"
done
[[ -d "$REFERENCE_RECEIPTS" ]] || die "missing reference heal receipts"
while [[ ! -f "$BASE_BUILD_COMPLETE" ]]; do sleep 30; done

[[ "$LANES_PER_GPU" =~ ^[0-9]+$ ]] || die "LANES_PER_GPU must be an integer"
((LANES_PER_GPU >= 1 && LANES_PER_GPU <= 12)) \
  || die "LANES_PER_GPU must be between 1 and 12"
IFS=, read -r -a physical_gpus <<<"$GPUS_CSV"
gpus=(); cpus=(); seen=,
for gpu in "${physical_gpus[@]}"; do
  [[ "$gpu" =~ ^[0-7]$ ]] || die "invalid GPU $gpu"
  [[ "$seen" != *",$gpu,"* ]] || die "duplicate GPU $gpu"
  seen+="$gpu,"
  for replica in $(seq 0 $((LANES_PER_GPU - 1))); do
    cpu_start=$((gpu * 12 + 12 * replica / LANES_PER_GPU))
    cpu_end=$((gpu * 12 + 12 * (replica + 1) / LANES_PER_GPU - 1))
    gpus+=("$gpu")
    cpus+=("$cpu_start-$cpu_end")
  done
done
lane_count=${#gpus[@]}
((lane_count >= 1 && lane_count <= 96)) || die "invalid lane count"
echo "$(date -u +%FT%TZ) sensitivity lanes=$lane_count physical_gpus=$GPUS_CSV lanes_per_gpu=$LANES_PER_GPU" \
  | tee -a "$LOG_ROOT/transition.log"

wait_lanes_idle() {
  while true; do
    local busy=0
    for gpu in "${physical_gpus[@]}"; do
      if nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits \
        | grep -Eq '^[0-9]+$'; then busy=1; fi
    done
    ((busy == 0)) && return
    echo "$(date -u +%FT%TZ) waiting for selected GPU lanes $GPUS_CSV" \
      | tee -a "$LOG_ROOT/transition.log"
    sleep 30
  done
}

"$PY" "$ROOT/tools/ggml_quant_bridge.py" --self-test \
  --ggml-lib "$GGML_LIB" --ggml-lib-sha256 "$GGML_LIB_SHA256" \
  --ggml-source-commit "$GGML_SOURCE_COMMIT" | tee "$LOG_ROOT/bridge-self-test.log"
"$PY" "$SCORER" --self-test | tee "$LOG_ROOT/sensitivity-self-test.log"
"$PY" "$MERGER" --self-test | tee "$LOG_ROOT/merge-self-test.log"
"$PY" "$SUMMARIZER" --self-test | tee "$LOG_ROOT/effects-self-test.log"
"$PY" "$ALLOCATION_SUMMARIZER" --self-test \
  | tee "$LOG_ROOT/allocation-comparison-self-test.log"
"$PY" "$PLAN_BUILDER" --self-test | tee "$LOG_ROOT/plan-self-test.log"
"$PY" "$HEALER" --self-test | tee "$LOG_ROOT/heal-self-test.log"
"$PY" "$REPACKER" test | tee "$LOG_ROOT/repack-self-test.log"
"$PY" "$ROOT/tools/merge_expert_overlay_fragments.py" --self-test \
  | tee "$LOG_ROOT/fragment-self-test.log"
"$PY" "$VALIDATOR" --self-test | tee "$LOG_ROOT/validator-self-test.log"
"$PY" "$ROOT/tools/audit_hy3_healed_routing.py" --self-test \
  | tee "$LOG_ROOT/routing-audit-self-test.log"

wait_lanes_idle
# Split all 79 layers deterministically over the selected lane count.
mapfile -t layer_ranges < <("$PY" - "$lane_count" <<'PY'
import sys
n=int(sys.argv[1])
for lane in range(n):
    start=1 + (79*lane)//n
    end=(79*(lane+1))//n
    print(f"{start}-{end}")
PY
)
pids=()
for lane in $(seq 0 $((lane_count - 1))); do
  out="$OUT_ROOT/lanes/lane-$lane.json"
  range=${layer_ranges[$lane]}
  if [[ -f "$out" ]]; then
    "$PY" - "$out" "$range" "$GGML_LIB_SHA256" "$GGML_SOURCE_COMMIT" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); a,b=map(int,sys.argv[2].split("-"))
assert d["model"]["moe_layers"] == list(range(a,b+1))
assert d["measurement"]["qtypes"] == ["IQ3_S","IQ4_XS","Q4_K"]
assert len(d["scores"]) == (b-a+1)*192
assert set(d["importance_sidecars"]) == {str(x) for x in range(a,b+1)}
for q in ("IQ3_S","IQ4_XS","Q4_K"):
    p=d["measurement"]["exact_quantizer_implementation"][q]
    assert p["library_sha256"] == sys.argv[3]
    assert p["llama_cpp_commit"] == sys.argv[4]
PY
    echo "reusing validated IQ4/Q4 lane $lane"
    continue
  fi
  CUDA_VISIBLE_DEVICES=${gpus[$lane]} taskset -c "${cpus[$lane]}" nice -n 19 \
    "$PY" "$SCORER" --trace-lock "$CALIBRATION/moe-inputs.lock.json" \
      --weight-trace "$CALIBRATION/routes-weighted.trace" --requests "$REQUESTS" \
      --source-dir "$SOURCE" --layers "$range" --device cuda:0 \
      --max-tokens-per-expert 16 --qtypes IQ3_S,IQ4_XS,Q4_K \
      --importance-dir "$OUT_ROOT/importance" --ggml-lib "$GGML_LIB" \
      --ggml-lib-sha256 "$GGML_LIB_SHA256" --ggml-source-commit "$GGML_SOURCE_COMMIT" \
      --out "$out" >"$LOG_ROOT/sensitivity-lane-$lane.log" 2>&1 &
  pids+=("$!")
done
failed=0; for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
((failed == 0)) || die "one or more IQ3/IQ4/Q4 sensitivity lanes failed"

"$PY" "$MERGER" "$OUT_ROOT"/lanes/lane-*.json --out "$OUT_ROOT/iq3-iq4-q4-sensitivity.json" \
  | tee "$LOG_ROOT/merge-lanes.log"
"$PY" "$MERGER" --merge-qtypes "$BASE_SENSITIVITY" \
  "$OUT_ROOT/iq3-iq4-q4-sensitivity.json" --out "$OUT_ROOT/seven-format-sensitivity.json" \
  | tee "$LOG_ROOT/merge-qtypes.log"
"$PY" "$SUMMARIZER" "$OUT_ROOT/seven-format-sensitivity.json" \
  --out "$OUT_ROOT/seven-format-effects-map.json" \
  --layer-csv "$OUT_ROOT/seven-format-layer-effects.csv" \
  --layer-projection-csv "$OUT_ROOT/seven-format-layer-projection-effects.csv" \
  | tee "$LOG_ROOT/effects-map.log"

plan="$PLAN_ROOT/$ARM.json"
if [[ ! -f "$plan" ]]; then
  taskset -c "$(IFS=,; echo "${cpus[*]}")" "$PY" "$PLAN_BUILDER" \
    --retention-scores "$RETENTION" --quant-sensitivity "$OUT_ROOT/seven-format-sensitivity.json" \
    --confidence-plan "$CONFIDENCE" --joint-receipts "$REFERENCE_RECEIPTS" \
    --reference-plan "$REFERENCE_PLAN" --target-logical-bytes "$TARGET_BYTES" \
    --min-survivors-per-layer 96 --retention-weight 0 --confidence-weight 0 --layer-weight 0 \
    --time-limit-seconds 900 --mip-rel-gap 1e-4 --out "$plan" | tee "$LOG_ROOT/plan.log"
fi
"$PY" - "$plan" "$TARGET_BYTES" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
assert d["calibration"]["public_eval_data_used_for_selection"] is False
assert d["policy"]["result_logical_bytes"] <= int(sys.argv[2])
assert set(d["policy"]["candidate_qtypes"]) == {"Q8_0","NVFP4","IQ3_S","IQ4_XS","Q4_K","Q3_K","Q2_K"}
assert min(x["retained"] for x in d["layer_summary"].values()) >= 96
PY
"$PY" "$ALLOCATION_SUMMARIZER" "$REFERENCE_PLAN" "$plan" \
  --out "$PLAN_ROOT/allocation-comparison.json" | tee "$LOG_ROOT/allocation-comparison.log"

wait_lanes_idle
mkdir -p "$HEAL_ROOT/$ARM/overlay" "$HEAL_ROOT/$ARM/receipts"
heal_lane() {
  local lane=$1
  for layer in $(seq 1 79); do
    (( (layer - 1) % lane_count == lane )) || continue
    shard="$HEAL_ROOT/$ARM/overlay/layer-$(printf '%03d' "$layer").safetensors"
    receipt="$HEAL_ROOT/$ARM/receipts/layer-$(printf '%03d' "$layer").receipt.json"
    if [[ -f "$shard" && -f "$receipt" ]]; then continue; fi
    [[ ! -e "$shard" && ! -e "$receipt" ]] || die "incomplete heal pair for layer $layer"
    CUDA_VISIBLE_DEVICES=${gpus[$lane]} taskset -c "${cpus[$lane]}" nice -n 19 \
      "$PY" "$HEALER" --mode joint --quantization-aware --rollback-non-improving --layer "$layer" \
        --plan "$plan" --scores "$RETENTION" --source-dir "$SOURCE" --device cuda:0 \
        --ggml-lib "$GGML_LIB" --ggml-lib-sha256 "$GGML_LIB_SHA256" \
        --ggml-source-commit "$GGML_SOURCE_COMMIT" \
        --out-shard "$shard" --receipt "$receipt"
  done
}
pids=(); for lane in $(seq 0 $((lane_count - 1))); do
  heal_lane "$lane" >"$LOG_ROOT/heal-lane-$lane.log" 2>&1 & pids+=("$!")
done
failed=0; for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
((failed == 0)) || die "one or more IQ4/Q4 heal lanes failed"

"$PY" "$ROOT/tools/merge_hy3_heal_shards.py" --receipt-dir "$HEAL_ROOT/$ARM/receipts" \
  --overlay-dir "$HEAL_ROOT/$ARM/overlay" --layers 1-79 \
  --lock "$HEAL_ROOT/$ARM/overlay.lock.json" | tee "$LOG_ROOT/heal-merge.log"
CUDA_VISIBLE_DEVICES=${gpus[0]} taskset -c "${cpus[0]}" nice -n 19 \
  "$PY" "$ROOT/tools/audit_hy3_healed_routing.py" --plan "$plan" \
    --trace-lock "$CALIBRATION/moe-inputs.lock.json" --overlay-dir "$HEAL_ROOT/$ARM/overlay" \
    --layers 1-79 --device cuda:0 --output "$LOG_ROOT/routing-audit.json" \
    | tee "$LOG_ROOT/routing-audit.log"
"$PY" - "$HEAL_ROOT/$ARM/receipts" "$LOG_ROOT/routing-audit.json" \
  "$LOG_ROOT/heal-quality.json" <<'PY'
import json,math,pathlib,sys
r=[json.loads((pathlib.Path(sys.argv[1])/f"layer-{x:03}.receipt.json").read_text()) for x in range(1,80)]
assert all(x["training"]["quantization_aware"] is True for x in r)
assert all(x["training"]["rollback_non_improving"] is True for x in r)
assert all(x["selection"]["policy"] == "private_holdout_terminal_mse_monotonic" for x in r)
assert all(x["selection"]["public_eval_data_used"] is False for x in r)
assert all(float(x["after"]["normalized_mse"]) <= float(x["before"]["normalized_mse"]) for x in r)
audit=json.loads(pathlib.Path(sys.argv[2]).read_text())
assert audit["format"] == "bw24-hy3-post-heal-routing-audit-v1"
assert audit["summary"]["all_layers_have_full_active_coverage"] is True
b=sum(float(x["before"]["normalized_mse"]) for x in r)/79
a=sum(float(x["after"]["normalized_mse"]) for x in r)/79
i=sum(float(x["after"]["normalized_mse"]) < float(x["before"]["normalized_mse"]) for x in r)
rolled_back=sum(bool(x["selection"]["rolled_back_to_unhealed_source"]) for x in r)
d={"format":"bw24-iq3-iq4-q4-post-requant-heal-gate-v1","layers":79,
   "mean_before_normalized_mse":b,"mean_after_requantization_normalized_mse":a,
   "improved_after_requantization_layers":i,
   "rolled_back_non_improving_layers":rolled_back,
   "holdout_dead_active_experts":sum(int(x["after"]["dead_active_experts"]) for x in r),
   "full_calibration_dead_active_experts":audit["summary"]["dead_active_experts"],
   "passed":a<b and i>=40,
   "public_eval_data_used":False}
pathlib.Path(sys.argv[3]).write_text(json.dumps(d,indent=2,sort_keys=True)+"\n")
print(json.dumps(d,sort_keys=True))
assert d["passed"]
PY

"$PY" "$ROOT/tools/export_hy3_router_overrides.py" --overlay-dir "$HEAL_ROOT/$ARM/overlay" \
  --layers 1-79 --blob "$HEAL_ROOT/$ARM/router-overrides.f32" \
  --receipt "$HEAL_ROOT/$ARM/router-overrides.json" | tee "$LOG_ROOT/router-export.log"
out="$ARTIFACT_ROOT/$ARM"; [[ ! -e "$out" ]] || die "refusing existing artifact $out"
mkdir -p "$out" "$LOG_ROOT/fragments"
pids=(); fragments=()
for lane in $(seq 0 $((lane_count - 1))); do
  selected=(); for layer in $(seq 1 79); do
    (( (layer - 1) % lane_count == lane )) && selected+=("$layer")
  done
  csv=$(IFS=,; echo "${selected[*]}")
  fragment="$LOG_ROOT/fragments/lane-$lane.manifest.json"; fragments+=("$fragment")
  taskset -c "${cpus[$lane]}" nice -n 19 "$PY" "$REPACKER" prepare \
    "$HEAL_ROOT/$ARM/overlay" "$out" --fallback-dir "$SOURCE" --plan "$plan" \
    --max-work-mb 64 --workers 1 --layers "$csv" --manifest-fragment "$fragment" \
    --ggml-lib "$GGML_LIB" --ggml-lib-sha256 "$GGML_LIB_SHA256" \
    --ggml-source-commit "$GGML_SOURCE_COMMIT" \
    >"$LOG_ROOT/repack-lane-$lane.log" 2>&1 & pids+=("$!")
done
failed=0; for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
((failed == 0)) || die "one or more IQ3/IQ4/Q4 repack lanes failed"
merge_args=(); for fragment in "${fragments[@]}"; do merge_args+=(--fragment "$fragment"); done
"$PY" "$ROOT/tools/merge_expert_overlay_fragments.py" "${merge_args[@]}" --plan "$plan" \
  --out-dir "$out" --tensor-overrides "$HEAL_ROOT/$ARM/router-overrides.json" \
  --output "$out/manifest.json" | tee "$LOG_ROOT/artifact-merge.log"
"$PY" "$VALIDATOR" "$out" --verify-sources >"$LOG_ROOT/validate-data.json"
scratch="$SCRATCH_ROOT/$ARM"; mkdir -p "$scratch"
rsync -a --delete "$out/" "$scratch/" | tee "$LOG_ROOT/rsync.log"
"$PY" "$VALIDATOR" "$scratch" --verify-sources >"$LOG_ROOT/validate-scratch.json"
(cd "$out" && find . -type f -print0 | sort -z | xargs -0 sha256sum) >"$LOG_ROOT/data.sha256"
(cd "$scratch" && find . -type f -print0 | sort -z | xargs -0 sha256sum) >"$LOG_ROOT/scratch.sha256"
cmp "$LOG_ROOT/data.sha256" "$LOG_ROOT/scratch.sha256"
sha256sum "$OUT_ROOT"/*.json "$OUT_ROOT"/*.csv "$PLAN_ROOT"/*.json \
  "$HEAL_ROOT/$ARM/overlay.lock.json" "$LOG_ROOT/heal-quality.json" \
  "$HEAL_ROOT/$ARM/router-overrides.json" "$out/manifest.json" >"$LOG_ROOT/evidence.sha256"
date -u +%FT%TZ | tee "$LOG_ROOT/complete"
