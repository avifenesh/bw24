#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
PY=${PY:-python3}
EXPECTED_COMMIT=${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the detached source commit}
SENSITIVITY_COMPLETE=${SENSITIVITY_COMPLETE:-/data/logs/hy3-quant-sensitivity-53de6ca/complete}
SENSITIVITY=${SENSITIVITY:-/data/calibration/hy3-quant-sensitivity-53de6ca/quant-sensitivity.json}
RETENTION=${RETENTION:-/data/calibration/hy3-100gb-5f02c37/expert-retention-scores.json}
CONFIDENCE=${CONFIDENCE:-/data/calibration/hy3-100gb-5f02c37/confidence-expert-scores-13a4d92.json}
REFERENCE_PLAN=${REFERENCE_PLAN:-/data/plans/per-expert-quant-100gb-5f02c37/traffic-nvfp4-53-q2-139-exact100gb.json}
REFERENCE_RECEIPTS=${REFERENCE_RECEIPTS:-/data/heal/per-expert-quant-100gb-5f02c37/joint/receipts}
SOURCE=${SOURCE:-/opt/dlami/nvme/models/hy3-source}
TRACE_LOCK=${TRACE_LOCK:-/data/calibration/hy3-100gb-5f02c37/moe-inputs.lock.json}
SCORES=${SCORES:-$RETENTION}
PLAN_ROOT=${PLAN_ROOT:-/data/plans/per-expert-quant-smart100-2605fde}
HEAL_ROOT=${HEAL_ROOT:-/data/heal/per-expert-quant-smart100-2605fde}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-/data/artifacts/per-expert-quant-smart100-2605fde}
SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/bw24-artifacts-smart100-2605fde}
LOG_ROOT=${LOG_ROOT:-/data/logs/smart100-build-2605fde}
TARGET_BYTES=${TARGET_BYTES:-100000000000}
BUILD_GPUS_CSV=${BUILD_GPUS_CSV:-}
ARMS_CSV=${ARMS_CSV:-smart100_empirical,smart100_balanced,smart100_rescue}
TARGET_BYTES_CSV=${TARGET_BYTES_CSV:-$TARGET_BYTES,$TARGET_BYTES,$TARGET_BYTES}
WEIGHTS_CSV=${WEIGHTS_CSV:-0:0:0,1:1:1,0.5:2:1}
MIP_REL_GAP=${MIP_REL_GAP:-1e-4}
MIN_IMPROVED_LAYERS=${MIN_IMPROVED_LAYERS:-40}
LAYER_CONSTRAINTS=${LAYER_CONSTRAINTS:-}
SELECTION_LOCK=${SELECTION_LOCK:-}
SELECTION_LOCK_VALIDATOR=${SELECTION_LOCK_VALIDATOR:-}
EXPECTED_PLAN_SHA256_CSV=${EXPECTED_PLAN_SHA256_CSV:-}
BASE_PLAN=${BASE_PLAN:-}
BASE_MODES_CSV=${BASE_MODES_CSV:-}

IFS=, read -r -a arms <<<"$ARMS_CSV"
IFS=, read -r -a targets <<<"$TARGET_BYTES_CSV"
IFS=, read -r -a weight_specs <<<"$WEIGHTS_CSV"
base_modes=()
if [[ -n "$BASE_MODES_CSV" ]]; then
  IFS=, read -r -a base_modes <<<"$BASE_MODES_CSV"
  [[ -n "$BASE_PLAN" && ${#base_modes[@]} -eq ${#arms[@]} ]] \
    || { echo "base plan and one base mode per arm are required together" >&2; exit 2; }
  for mode in "${base_modes[@]}"; do
    [[ "$mode" == restore-only || "$mode" == precision-only || "$mode" == hybrid ]] \
      || { echo "invalid base mode $mode" >&2; exit 2; }
  done
elif [[ -n "$BASE_PLAN" ]]; then
  echo "BASE_PLAN requires BASE_MODES_CSV" >&2; exit 2
fi
expected_plan_hashes=()
if [[ -n "$EXPECTED_PLAN_SHA256_CSV" ]]; then
  IFS=, read -r -a expected_plan_hashes <<<"$EXPECTED_PLAN_SHA256_CSV"
  [[ ${#expected_plan_hashes[@]} -eq ${#arms[@]} ]] \
    || { echo "expected plan hash count must match arms" >&2; exit 2; }
fi
[[ ${#arms[@]} -gt 0 && ${#arms[@]} -eq ${#targets[@]} \
  && ${#arms[@]} -eq ${#weight_specs[@]} ]] \
  || { echo "arms, target bytes, and weight counts must match" >&2; exit 2; }
[[ "$MIN_IMPROVED_LAYERS" =~ ^[1-9][0-9]*$ && "$MIN_IMPROVED_LAYERS" -le 79 ]] \
  || { echo "MIN_IMPROVED_LAYERS must be in 1..79" >&2; exit 2; }
weights=()
for spec in "${weight_specs[@]}"; do
  [[ "$spec" =~ ^[0-9.]+:[0-9.]+:[0-9.]+$ ]] \
    || { echo "invalid retention:confidence:layer weights $spec" >&2; exit 2; }
  weights+=("${spec//:/ }")
done
all_cpus=(0-11 12-23 24-35 36-47 48-59 60-71 72-83 84-95)
if [[ -n "$BUILD_GPUS_CSV" ]]; then
  IFS=, read -r -a gpus <<<"$BUILD_GPUS_CSV"
else
  gpus=(0 1 2 3 4 5 6 7)
fi
cpus=()
seen_gpus=,
for gpu in "${gpus[@]}"; do
  [[ "$gpu" =~ ^[0-7]$ ]] || { echo "invalid build GPU $gpu" >&2; exit 2; }
  [[ "$seen_gpus" != *",$gpu,"* ]] || { echo "duplicate build GPU $gpu" >&2; exit 2; }
  seen_gpus+="$gpu,"
  cpus+=("${all_cpus[$gpu]}")
done
[[ ${#gpus[@]} -eq ${#cpus[@]} && ${#gpus[@]} -gt 0 ]] || exit 2
lane_count=${#gpus[@]}
control_cpus=$(IFS=,; echo "${cpus[*]}")
layers=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79)

die() { echo "smart100 build transition: $*" >&2; exit 1; }
mkdir -p "$PLAN_ROOT" "$HEAL_ROOT" "$ARTIFACT_ROOT" "$SCRATCH_ROOT" "$LOG_ROOT"
exec 9>"$LOG_ROOT/transition.lock"
flock -n 9 || die "another transition owns $LOG_ROOT/transition.lock"
echo "$(date -u +%FT%TZ) smart100 build transition started" | tee -a "$LOG_ROOT/transition.log"
[[ $(git -C "$ROOT" rev-parse HEAD) == "$EXPECTED_COMMIT" ]] || die "source commit mismatch"

while [[ -n "$SENSITIVITY_COMPLETE" && ! -f "$SENSITIVITY_COMPLETE" ]]; do sleep 30; done
for path in "$SENSITIVITY" "$RETENTION" "$REFERENCE_PLAN" "$TRACE_LOCK" \
  "$SOURCE/config.json" "$SOURCE/model.safetensors.index.json"; do
  [[ -f "$path" ]] || die "missing input $path"
done
if [[ -n "$CONFIDENCE" ]]; then
  [[ -f "$CONFIDENCE" ]] || die "missing confidence plan $CONFIDENCE"
fi
if [[ -n "$REFERENCE_RECEIPTS" ]]; then
  [[ -d "$REFERENCE_RECEIPTS" ]] || die "missing reference receipts $REFERENCE_RECEIPTS"
fi
if [[ -n "$BASE_PLAN" ]]; then
  [[ -f "$BASE_PLAN" ]] || die "missing base plan $BASE_PLAN"
fi
if [[ -n "$LAYER_CONSTRAINTS" ]]; then
  [[ -f "$LAYER_CONSTRAINTS" ]] || die "missing layer constraints $LAYER_CONSTRAINTS"
fi
if [[ -n "$SELECTION_LOCK" || -n "$SELECTION_LOCK_VALIDATOR" ]]; then
  [[ -n "$SELECTION_LOCK" && -n "$SELECTION_LOCK_VALIDATOR" ]] \
    || die "selection lock and validator must be provided together"
  [[ -f "$SELECTION_LOCK" && -f "$SELECTION_LOCK_VALIDATOR" ]] \
    || die "missing selection lock or validator"
  "$PY" "$SELECTION_LOCK_VALIDATOR" --lock "$SELECTION_LOCK" --verify-inputs \
    | tee "$LOG_ROOT/selection-lock-validation.log"
fi
if [[ -n "$BUILD_GPUS_CSV" ]]; then
  echo "$(date -u +%FT%TZ) isolated build lanes GPUs=$BUILD_GPUS_CSV CPUs=$control_cpus" \
    | tee -a "$LOG_ROOT/transition.log"
  while true; do
    busy=0
    for gpu in "${gpus[@]}"; do
      if nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits \
        | grep -Eq '^[0-9]+$'; then busy=1; fi
    done
    ((busy == 0)) && break
    echo "$(date -u +%FT%TZ) waiting for isolated build GPUs to become idle" \
      | tee -a "$LOG_ROOT/transition.log"
    sleep 30
  done
else
  while pgrep -x bw24-server >/dev/null \
    || pgrep -af '/harbor run ' >/dev/null \
    || [[ -n $(docker ps -q) ]]; do
    echo "$(date -u +%FT%TZ) waiting for active model/eval work to release GPUs" \
      | tee -a "$LOG_ROOT/transition.log"
    sleep 30
  done
fi

"$PY" "$ROOT/tools/build_hy3_smart_budget_plan.py" --self-test | tee "$LOG_ROOT/plan-self-test.log"
"$PY" "$ROOT/tools/summarize_hy3_smart_allocations.py" --self-test \
  | tee "$LOG_ROOT/allocation-comparison-self-test.log"
"$PY" "$ROOT/tools/summarize_hy3_plan_agreement.py" --self-test \
  | tee "$LOG_ROOT/plan-agreement-self-test.log"
"$PY" "$ROOT/tools/heal_hy3_pruned_layer.py" --self-test | tee "$LOG_ROOT/heal-self-test.log"
"$PY" "$ROOT/tools/merge_hy3_heal_shards.py" --self-test | tee "$LOG_ROOT/heal-merge-self-test.log"
"$PY" "$ROOT/tools/export_hy3_router_overrides.py" --self-test | tee "$LOG_ROOT/export-self-test.log"
"$PY" "$ROOT/tools/prepare_mixed_expert_repack.py" test | tee "$LOG_ROOT/repack-self-test.log"
"$PY" "$ROOT/tools/merge_expert_overlay_fragments.py" --self-test | tee "$LOG_ROOT/fragment-self-test.log"
"$PY" "$ROOT/tools/audit_hy3_healed_routing.py" --self-test \
  | tee "$LOG_ROOT/routing-audit-self-test.log"

for index in "${!arms[@]}"; do
  arm=${arms[$index]}; read -r retention_weight confidence_weight layer_weight <<<"${weights[$index]}"
  target_bytes=${targets[$index]}
  [[ "$target_bytes" =~ ^[1-9][0-9]+$ ]] || die "invalid target bytes for $arm: $target_bytes"
  plan="$PLAN_ROOT/$arm.json"
  if [[ -f "$plan" ]]; then
    echo "$(date -u +%FT%TZ) reusing existing frozen plan $plan" | tee -a "$LOG_ROOT/transition.log"
    continue
  fi
  constraint_args=()
  [[ -z "$LAYER_CONSTRAINTS" ]] || constraint_args=(--layer-constraints "$LAYER_CONSTRAINTS")
  base_args=()
  if [[ ${#base_modes[@]} -gt 0 ]]; then
    base_args=(--base-plan "$BASE_PLAN" --base-mode "${base_modes[$index]}")
  fi
  confidence_args=()
  [[ -z "$CONFIDENCE" ]] || confidence_args=(--confidence-plan "$CONFIDENCE")
  receipt_args=()
  [[ -z "$REFERENCE_RECEIPTS" ]] || receipt_args=(--joint-receipts "$REFERENCE_RECEIPTS")
  taskset -c "$control_cpus" "$PY" "$ROOT/tools/build_hy3_smart_budget_plan.py" \
    --retention-scores "$RETENTION" --quant-sensitivity "$SENSITIVITY" \
    --reference-plan "$REFERENCE_PLAN" --target-logical-bytes "$target_bytes" \
    --min-survivors-per-layer 96 --retention-weight "$retention_weight" \
    --confidence-weight "$confidence_weight" --layer-weight "$layer_weight" \
    --time-limit-seconds 900 --mip-rel-gap "$MIP_REL_GAP" "${constraint_args[@]}" \
    "${base_args[@]}" "${confidence_args[@]}" "${receipt_args[@]}" \
    --out "$plan" | tee "$LOG_ROOT/plan-$arm.log"
done

plan_paths=()
for index in "${!arms[@]}"; do
  arm=${arms[$index]}; plan="$PLAN_ROOT/$arm.json"; plan_paths+=("$plan")
  if [[ ${#expected_plan_hashes[@]} -gt 0 ]]; then
    expected=${expected_plan_hashes[$index]}
    [[ "$expected" =~ ^[0-9a-f]{64}$ ]] || die "invalid expected plan hash for $arm"
    [[ $(sha256sum "$plan" | cut -d' ' -f1) == "$expected" ]] \
      || die "frozen plan hash mismatch for $arm"
  fi
done
"$PY" "$ROOT/tools/summarize_hy3_smart_allocations.py" \
  "${plan_paths[@]}" --out "$PLAN_ROOT/allocation-comparison.json" \
  --require-distinct | tee "$LOG_ROOT/allocation-comparison.log"
"$PY" "$ROOT/tools/summarize_hy3_plan_agreement.py" \
  "${plan_paths[@]}" --out "$PLAN_ROOT/plan-agreement.json" \
  --retention-scores "$RETENTION" \
  --layer-csv "$PLAN_ROOT/plan-agreement-layers.csv" | tee "$LOG_ROOT/plan-agreement.log"

arm_specs=()
for index in "${!arms[@]}"; do
  arm_specs+=("${arms[$index]}=${targets[$index]}=${weight_specs[$index]}")
done
"$PY" - "$PLAN_ROOT" "${arm_specs[@]}" <<'PY'
import json, pathlib, sys
root, *arm_specs = sys.argv[1:]
for arm_spec in arm_specs:
    arm, target, weights = arm_spec.split("=", 2)
    retention, confidence, layer = map(float, weights.split(":"))
    d=json.loads((pathlib.Path(root)/f"{arm}.json").read_text())
    assert d["calibration"]["public_eval_data_used_for_selection"] is False
    assert d["policy"]["target_logical_bytes"] == int(target)
    assert d["policy"]["result_logical_bytes"] <= int(target)
    assert d["policy"]["importance_weights"] == {
        "retention": retention, "confidence": confidence, "layer": layer,
    }
    assert d["selection"]["retained_experts"] + d["selection"]["pruned_experts"] == 79*192
    assert min(int(x["retained"]) for x in d["layer_summary"].values()) >= 96
PY
if [[ ${#base_modes[@]} -gt 0 ]]; then
  "$PY" - "$PLAN_ROOT" "$BASE_PLAN" "$BASE_MODES_CSV" "${arms[@]}" <<'PY'
import hashlib,json,pathlib,sys
root,base_path,modes,*arms=sys.argv[1:]
base_hash=hashlib.sha256(pathlib.Path(base_path).read_bytes()).hexdigest()
modes=modes.split(",")
for arm,mode in zip(arms,modes,strict=True):
    d=json.loads((pathlib.Path(root)/f"{arm}.json").read_text())
    receipt=d["policy"]["base_preservation"]
    assert receipt["mode"] == mode
    assert receipt["base_plan"]["sha256"] == base_hash
    assert receipt["retained_experts_may_be_pruned"] is False
PY
fi

# Run every candidate/layer repair through fixed GPU/CPU lanes. Candidate tasks are interleaved so
# every GPU stays occupied without running multiple repairs on one device.
heal_lane() {
  local lane=$1 ordinal=0
  for arm in "${arms[@]}"; do
    mkdir -p "$HEAL_ROOT/$arm/overlay" "$HEAL_ROOT/$arm/receipts"
    for layer in "${layers[@]}"; do
      if ((ordinal % lane_count == lane)); then
        shard="$HEAL_ROOT/$arm/overlay/layer-$(printf '%03d' "$layer").safetensors"
        receipt="$HEAL_ROOT/$arm/receipts/layer-$(printf '%03d' "$layer").receipt.json"
        if [[ -f "$shard" && -f "$receipt" ]]; then
          echo "$arm lane=$lane layer=$layer already complete"
          ordinal=$((ordinal + 1))
          continue
        fi
        [[ ! -e "$shard" && ! -e "$receipt" ]] \
          || die "$arm layer $layer has an incomplete shard/receipt pair"
        CUDA_VISIBLE_DEVICES=${gpus[$lane]} taskset -c "${cpus[$lane]}" nice -n 19 \
          "$PY" "$ROOT/tools/heal_hy3_pruned_layer.py" \
            --mode joint --quantization-aware --rollback-non-improving --layer "$layer" \
            --plan "$PLAN_ROOT/$arm.json" --scores "$SCORES" --source-dir "$SOURCE" \
            --device cuda:0 --out-shard "$shard" --receipt "$receipt"
        echo "$arm lane=$lane layer=$layer complete"
      fi
      ordinal=$((ordinal + 1))
    done
  done
}
pids=()
for lane in $(seq 0 $((lane_count - 1))); do
  heal_lane "$lane" >"$LOG_ROOT/heal-lane-$lane.log" 2>&1 & pids+=("$!")
done
failed=0; for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
((failed == 0)) || die "one or more quantization-aware heal lanes failed"

# Holdout MSE stays isolated for the quality gate, but an expert is considered dead only when it is
# unused across every frozen private calibration token.  A 10% holdout alone is too sparse to audit
# up to 192 active experts reliably.
for arm in "${arms[@]}"; do
  CUDA_VISIBLE_DEVICES=${gpus[0]} taskset -c "${cpus[0]}" nice -n 19 \
    "$PY" "$ROOT/tools/audit_hy3_healed_routing.py" \
      --plan "$PLAN_ROOT/$arm.json" --trace-lock "$TRACE_LOCK" \
      --overlay-dir "$HEAL_ROOT/$arm/overlay" --layers 1-79 --device cuda:0 \
      --output "$LOG_ROOT/routing-audit-$arm.json" \
      | tee "$LOG_ROOT/routing-audit-$arm.log"
done

eligible_arms=()
rejected_arms=()
for arm in "${arms[@]}"; do
  "$PY" - "$HEAL_ROOT/$arm/receipts" "$LOG_ROOT/routing-audit-$arm.json" \
    "$LOG_ROOT/heal-quality-$arm.json" "$MIN_IMPROVED_LAYERS" <<'PY'
import json,math,pathlib,sys
minimum_improved=int(sys.argv[4])
receipts=[]
for layer in range(1,80):
    p=pathlib.Path(sys.argv[1])/f"layer-{layer:03}.receipt.json"
    d=json.loads(p.read_text()); assert d["training"]["quantization_aware"] is True
    assert d["training"]["rollback_non_improving"] is True
    assert d["selection"]["policy"] == "private_holdout_terminal_mse_monotonic"
    assert d["selection"]["public_eval_data_used"] is False
    for section in ("before","trained_after_pre_requantization","trained_after_requantization",
                    "after_pre_requantization","after"):
        assert all(math.isfinite(float(v)) for v in d[section].values())
    assert float(d["after"]["normalized_mse"]) <= float(d["before"]["normalized_mse"])
    receipts.append(d)
audit=json.loads(pathlib.Path(sys.argv[2]).read_text())
assert audit["format"] == "bw24-hy3-post-heal-routing-audit-v1"
assert audit["summary"]["layers"] == 79
assert audit["summary"]["all_layers_have_full_active_coverage"] is True
before=sum(float(d["before"]["normalized_mse"]) for d in receipts)/len(receipts)
pre=sum(float(d["after_pre_requantization"]["normalized_mse"]) for d in receipts)/len(receipts)
after=sum(float(d["after"]["normalized_mse"]) for d in receipts)/len(receipts)
improved=sum(float(d["after"]["normalized_mse"]) < float(d["before"]["normalized_mse"]) for d in receipts)
rolled_back=sum(bool(d["selection"]["rolled_back_to_unhealed_source"]) for d in receipts)
result={"format":"bw24-smart100-post-requant-heal-gate-v1","layers":len(receipts),
        "mean_before_normalized_mse":before,"mean_after_pre_requantization_normalized_mse":pre,
        "mean_after_requantization_normalized_mse":after,"improved_after_requantization_layers":improved,
        "minimum_improved_layers":minimum_improved,
        "rolled_back_non_improving_layers":rolled_back,
        "holdout_dead_active_experts":sum(int(d["after"]["dead_active_experts"]) for d in receipts),
        "full_calibration_dead_active_experts":audit["summary"]["dead_active_experts"],
        "passed":after < before and improved >= minimum_improved
            and audit["summary"]["dead_active_experts"] == 0,
        "public_eval_data_used":False}
pathlib.Path(sys.argv[3]).write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
print(json.dumps(result,sort_keys=True))
PY
  if [[ $("$PY" -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["passed"]).lower())' \
      "$LOG_ROOT/heal-quality-$arm.json") != true ]]; then
    rejected_arms+=("$arm")
    echo "$(date -u +%FT%TZ) rejecting $arm after post-requantization heal gate" \
      | tee -a "$LOG_ROOT/transition.log"
    continue
  fi
  eligible_arms+=("$arm")
  "$PY" "$ROOT/tools/merge_hy3_heal_shards.py" \
    --receipt-dir "$HEAL_ROOT/$arm/receipts" --overlay-dir "$HEAL_ROOT/$arm/overlay" \
    --layers 1-79 --lock "$HEAL_ROOT/$arm/overlay.lock.json" | tee "$LOG_ROOT/merge-$arm.log"
  "$PY" "$ROOT/tools/export_hy3_router_overrides.py" \
    --overlay-dir "$HEAL_ROOT/$arm/overlay" --layers 1-79 \
    --blob "$HEAL_ROOT/$arm/router-overrides.f32" \
    --receipt "$HEAL_ROOT/$arm/router-overrides.json" | tee "$LOG_ROOT/export-$arm.log"

  out="$ARTIFACT_ROOT/$arm"; [[ ! -e "$out" ]] || die "refusing existing artifact $out"
  mkdir -p "$out" "$LOG_ROOT/fragments/$arm"
  pids=(); fragments=()
  for lane in $(seq 0 $((lane_count - 1))); do
    selected=()
    for layer in "${layers[@]}"; do
      (( (layer - 1) % lane_count == lane )) && selected+=("$layer")
    done
    csv=$(IFS=,; echo "${selected[*]}")
    fragment="$LOG_ROOT/fragments/$arm/lane-$lane.manifest.json"; fragments+=("$fragment")
    taskset -c "${cpus[$lane]}" nice -n 19 "$PY" "$ROOT/tools/prepare_mixed_expert_repack.py" prepare \
      "$HEAL_ROOT/$arm/overlay" "$out" --fallback-dir "$SOURCE" \
      --plan "$PLAN_ROOT/$arm.json" --max-work-mb 64 --workers 1 \
      --layers "$csv" --manifest-fragment "$fragment" \
      >"$LOG_ROOT/build-$arm-lane-$lane.log" 2>&1 & pids+=("$!")
  done
  failed=0; for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  ((failed == 0)) || die "artifact fragment build failed for $arm"
  merge_args=(); for fragment in "${fragments[@]}"; do merge_args+=(--fragment "$fragment"); done
  "$PY" "$ROOT/tools/merge_expert_overlay_fragments.py" "${merge_args[@]}" \
    --plan "$PLAN_ROOT/$arm.json" --out-dir "$out" \
    --tensor-overrides "$HEAL_ROOT/$arm/router-overrides.json" \
    --output "$out/manifest.json" | tee "$LOG_ROOT/artifact-merge-$arm.log"
  "$PY" "$ROOT/research/per-expert-quant/validate_artifact.py" "$out" --verify-sources \
    >"$LOG_ROOT/validate-$arm-data.json"
  scratch="$SCRATCH_ROOT/$arm"; mkdir -p "$scratch"
  rsync -a --delete "$out/" "$scratch/" | tee "$LOG_ROOT/rsync-$arm.log"
  "$PY" "$ROOT/research/per-expert-quant/validate_artifact.py" "$scratch" --verify-sources \
    >"$LOG_ROOT/validate-$arm-scratch.json"
  (cd "$out" && find . -type f -print0 | sort -z | xargs -0 sha256sum) >"$LOG_ROOT/$arm.data.sha256"
  (cd "$scratch" && find . -type f -print0 | sort -z | xargs -0 sha256sum) >"$LOG_ROOT/$arm.scratch.sha256"
  cmp "$LOG_ROOT/$arm.data.sha256" "$LOG_ROOT/$arm.scratch.sha256"
done

(( ${#eligible_arms[@]} > 0 )) || die "all smart100 candidates failed the post-requantization heal gate"
"$PY" - "$LOG_ROOT/eligible-arms.json" \
  "$(IFS=,; echo "${eligible_arms[*]}")" "$(IFS=,; echo "${rejected_arms[*]}")" <<'PY'
import json,pathlib,sys
out,eligible,rejected=sys.argv[1:]
split=lambda value: [item for item in value.split(",") if item]
pathlib.Path(out).write_text(json.dumps({
    "format":"bw24-smart100-heal-eligibility-v1",
    "eligible_arms":split(eligible),
    "rejected_arms":split(rejected),
},indent=2,sort_keys=True)+"\n")
PY
evidence=("$PLAN_ROOT"/*.json "$PLAN_ROOT/plan-agreement-layers.csv" \
  "$LOG_ROOT"/heal-quality-*.json "$LOG_ROOT"/routing-audit-*.json "$LOG_ROOT/eligible-arms.json")
[[ -z "$LAYER_CONSTRAINTS" ]] || evidence+=("$LAYER_CONSTRAINTS")
[[ -z "$SELECTION_LOCK" ]] || evidence+=("$SELECTION_LOCK")
for arm in "${eligible_arms[@]}"; do
  evidence+=("$HEAL_ROOT/$arm/overlay.lock.json" "$HEAL_ROOT/$arm/router-overrides.json" \
    "$ARTIFACT_ROOT/$arm/manifest.json")
done
sha256sum "${evidence[@]}" >"$LOG_ROOT/evidence.sha256"
date -u +%FT%TZ | tee "$LOG_ROOT/complete"
