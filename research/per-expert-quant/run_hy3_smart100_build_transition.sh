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
SCORES=${SCORES:-$RETENTION}
PLAN_ROOT=${PLAN_ROOT:-/data/plans/per-expert-quant-smart100-2605fde}
HEAL_ROOT=${HEAL_ROOT:-/data/heal/per-expert-quant-smart100-2605fde}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-/data/artifacts/per-expert-quant-smart100-2605fde}
SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/bw24-artifacts-smart100-2605fde}
LOG_ROOT=${LOG_ROOT:-/data/logs/smart100-build-2605fde}
TARGET_BYTES=${TARGET_BYTES:-100000000000}

arms=(smart100_empirical smart100_balanced smart100_rescue)
weights=("0 0 0" "1 1 1" "0.5 2 1")
cpus=(0-11 12-23 24-35 36-47 48-59 60-71 72-83 84-95)
layers=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79)

die() { echo "smart100 build transition: $*" >&2; exit 1; }
mkdir -p "$PLAN_ROOT" "$HEAL_ROOT" "$ARTIFACT_ROOT" "$SCRATCH_ROOT" "$LOG_ROOT"
exec 9>"$LOG_ROOT/transition.lock"
flock -n 9 || die "another transition owns $LOG_ROOT/transition.lock"
echo "$(date -u +%FT%TZ) smart100 build transition started" | tee -a "$LOG_ROOT/transition.log"
[[ $(git -C "$ROOT" rev-parse HEAD) == "$EXPECTED_COMMIT" ]] || die "source commit mismatch"

while [[ ! -f "$SENSITIVITY_COMPLETE" ]]; do sleep 30; done
for path in "$SENSITIVITY" "$RETENTION" "$CONFIDENCE" "$REFERENCE_PLAN" \
  "$SOURCE/config.json" "$SOURCE/model.safetensors.index.json"; do
  [[ -f "$path" ]] || die "missing input $path"
done
while pgrep -x bw24-server >/dev/null \
  || pgrep -af '/harbor run ' >/dev/null \
  || [[ -n $(docker ps -q) ]]; do
  echo "$(date -u +%FT%TZ) waiting for active model/eval work to release GPUs" \
    | tee -a "$LOG_ROOT/transition.log"
  sleep 30
done

"$PY" "$ROOT/tools/build_hy3_smart_budget_plan.py" --self-test | tee "$LOG_ROOT/plan-self-test.log"
"$PY" "$ROOT/tools/heal_hy3_pruned_layer.py" --self-test | tee "$LOG_ROOT/heal-self-test.log"
"$PY" "$ROOT/tools/merge_hy3_heal_shards.py" --self-test | tee "$LOG_ROOT/heal-merge-self-test.log"
"$PY" "$ROOT/tools/export_hy3_router_overrides.py" --self-test | tee "$LOG_ROOT/export-self-test.log"
"$PY" "$ROOT/tools/prepare_mixed_expert_repack.py" test | tee "$LOG_ROOT/repack-self-test.log"
"$PY" "$ROOT/tools/merge_expert_overlay_fragments.py" --self-test | tee "$LOG_ROOT/fragment-self-test.log"

for index in "${!arms[@]}"; do
  arm=${arms[$index]}; read -r retention_weight confidence_weight layer_weight <<<"${weights[$index]}"
  plan="$PLAN_ROOT/$arm.json"
  [[ ! -e "$plan" ]] || die "refusing existing plan $plan"
  "$PY" "$ROOT/tools/build_hy3_smart_budget_plan.py" \
    --retention-scores "$RETENTION" --quant-sensitivity "$SENSITIVITY" \
    --confidence-plan "$CONFIDENCE" --joint-receipts "$REFERENCE_RECEIPTS" \
    --reference-plan "$REFERENCE_PLAN" --target-logical-bytes "$TARGET_BYTES" \
    --min-survivors-per-layer 96 --retention-weight "$retention_weight" \
    --confidence-weight "$confidence_weight" --layer-weight "$layer_weight" \
    --time-limit-seconds 900 --mip-rel-gap 1e-4 --out "$plan" | tee "$LOG_ROOT/plan-$arm.log"
done

"$PY" - "$PLAN_ROOT" "$TARGET_BYTES" "${arms[@]}" <<'PY'
import json, pathlib, sys
root, target, *arms = sys.argv[1:]
for arm in arms:
    d=json.loads((pathlib.Path(root)/f"{arm}.json").read_text())
    assert d["calibration"]["public_eval_data_used_for_selection"] is False
    assert d["policy"]["result_logical_bytes"] <= int(target)
    assert d["selection"]["retained_experts"] + d["selection"]["pruned_experts"] == 79*192
    assert min(int(x["retained"]) for x in d["layer_summary"].values()) >= 96
PY

# Run all 237 layer repairs through eight fixed GPU/CPU lanes.  Candidate tasks are interleaved so
# every GPU stays occupied without running multiple repairs on one device.
heal_lane() {
  local lane=$1 ordinal=0
  for arm in "${arms[@]}"; do
    mkdir -p "$HEAL_ROOT/$arm/overlay" "$HEAL_ROOT/$arm/receipts"
    for layer in "${layers[@]}"; do
      if ((ordinal % 8 == lane)); then
        CUDA_VISIBLE_DEVICES=$lane taskset -c "${cpus[$lane]}" nice -n 19 \
          "$PY" "$ROOT/tools/heal_hy3_pruned_layer.py" \
            --mode joint --quantization-aware --layer "$layer" \
            --plan "$PLAN_ROOT/$arm.json" --scores "$SCORES" --source-dir "$SOURCE" \
            --device cuda:0 --out-shard "$HEAL_ROOT/$arm/overlay/layer-$(printf '%03d' "$layer").safetensors" \
            --receipt "$HEAL_ROOT/$arm/receipts/layer-$(printf '%03d' "$layer").receipt.json"
        echo "$arm lane=$lane layer=$layer complete"
      fi
      ordinal=$((ordinal + 1))
    done
  done
}
pids=()
for lane in $(seq 0 7); do heal_lane "$lane" >"$LOG_ROOT/heal-lane-$lane.log" 2>&1 & pids+=("$!"); done
failed=0; for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
((failed == 0)) || die "one or more quantization-aware heal lanes failed"

for arm in "${arms[@]}"; do
  "$PY" "$ROOT/tools/merge_hy3_heal_shards.py" \
    --receipt-dir "$HEAL_ROOT/$arm/receipts" --overlay-dir "$HEAL_ROOT/$arm/overlay" \
    --layers 1-79 --lock "$HEAL_ROOT/$arm/overlay.lock.json" | tee "$LOG_ROOT/merge-$arm.log"
  "$PY" - "$HEAL_ROOT/$arm/receipts" "$LOG_ROOT/heal-quality-$arm.json" <<'PY'
import json,math,pathlib,sys
receipts=[]
for layer in range(1,80):
    p=pathlib.Path(sys.argv[1])/f"layer-{layer:03}.receipt.json"
    d=json.loads(p.read_text()); assert d["training"]["quantization_aware"] is True
    for section in ("before","after_pre_requantization","after"):
        assert all(math.isfinite(float(v)) for v in d[section].values())
    assert int(d["after"]["dead_active_experts"]) == 0
    receipts.append(d)
before=sum(float(d["before"]["normalized_mse"]) for d in receipts)/len(receipts)
pre=sum(float(d["after_pre_requantization"]["normalized_mse"]) for d in receipts)/len(receipts)
after=sum(float(d["after"]["normalized_mse"]) for d in receipts)/len(receipts)
improved=sum(float(d["after"]["normalized_mse"]) < float(d["before"]["normalized_mse"]) for d in receipts)
result={"format":"bw24-smart100-post-requant-heal-gate-v1","layers":len(receipts),
        "mean_before_normalized_mse":before,"mean_after_pre_requantization_normalized_mse":pre,
        "mean_after_requantization_normalized_mse":after,"improved_after_requantization_layers":improved,
        "passed":after < before and improved >= 40,"public_eval_data_used":False}
pathlib.Path(sys.argv[2]).write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
print(json.dumps(result,sort_keys=True))
assert result["passed"], result
PY
  "$PY" "$ROOT/tools/export_hy3_router_overrides.py" \
    --overlay-dir "$HEAL_ROOT/$arm/overlay" --layers 1-79 \
    --blob "$HEAL_ROOT/$arm/router-overrides.f32" \
    --receipt "$HEAL_ROOT/$arm/router-overrides.json" | tee "$LOG_ROOT/export-$arm.log"

  out="$ARTIFACT_ROOT/$arm"; [[ ! -e "$out" ]] || die "refusing existing artifact $out"
  mkdir -p "$out" "$LOG_ROOT/fragments/$arm"
  pids=(); fragments=()
  for lane in $(seq 0 7); do
    selected=()
    for layer in "${layers[@]}"; do (( (layer - 1) % 8 == lane )) && selected+=("$layer"); done
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

sha256sum "$PLAN_ROOT"/*.json "$HEAL_ROOT"/*/overlay.lock.json "$LOG_ROOT"/heal-quality-*.json \
  "$HEAL_ROOT"/*/router-overrides.json "$ARTIFACT_ROOT"/*/manifest.json \
  >"$LOG_ROOT/evidence.sha256"
date -u +%FT%TZ | tee "$LOG_ROOT/complete"
