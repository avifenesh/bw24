#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/data/src/bw24-hy3-110gb}
PY=${PY:-/data/venvs/hy3-110gb/bin/python}
ROOT=${ROOT:-/data/experiments/hy3-110gb}
SOURCE=${SOURCE:-/opt/dlami/nvme/models/hy3-source}
SOURCE_RECEIPT=${SOURCE_RECEIPT:-$ROOT/receipts/source-download.json}
QUANTIZER_RECEIPT=${QUANTIZER_RECEIPT:-$ROOT/receipts/llama-cpp-quantizer.json}
ARMS_CSV=${ARMS_CSV:-layer100-matched,layer110-delta-restore}
PLANS_CSV=${PLANS_CSV:-$ROOT/plans/layer-balanced100.frozen.json,$ROOT/plans/layer110-delta-restore.json}
LANES_PER_ARM=${LANES_PER_ARM:-8}
WORKERS_PER_LANE=${WORKERS_PER_LANE:-10}

LOG_ROOT=$ROOT/logs/matched-repack
ARTIFACT_ROOT=$ROOT/artifacts/matched-repack
BOUND_ROOT=$ROOT/plans/repack-bound
IMPORTANCE_ROOT=$ROOT/calibration/repack-importance
mkdir -p "$LOG_ROOT" "$ARTIFACT_ROOT" "$BOUND_ROOT" "$IMPORTANCE_ROOT"
exec 9>"$LOG_ROOT/transition.lock"
flock -n 9 || { echo "another matched repack owns the transition lock" >&2; exit 1; }
exec > >(tee -a "$LOG_ROOT/transition.log") 2>&1

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] waiting for pinned source and quantizer"
while [[ ! -f "$SOURCE_RECEIPT" || ! -f "$QUANTIZER_RECEIPT" ]]; do sleep 20; done
jq -e '.complete == true and .shard_count == 99' "$SOURCE_RECEIPT" >/dev/null
ggml_lib=$(jq -r .library "$QUANTIZER_RECEIPT")
ggml_sha=$(jq -r .library_sha256 "$QUANTIZER_RECEIPT")
ggml_commit=$(jq -r .llama_cpp_commit "$QUANTIZER_RECEIPT")
[[ -f "$ggml_lib" && $(sha256sum "$ggml_lib" | awk '{print $1}') == "$ggml_sha" ]]
export LD_LIBRARY_PATH="$(dirname "$ggml_lib"):${LD_LIBRARY_PATH:-}"

IFS=, read -r -a arms <<<"$ARMS_CSV"
IFS=, read -r -a plans <<<"$PLANS_CSV"
[[ ${#arms[@]} -eq ${#plans[@]} && ${#arms[@]} -eq 2 ]]
[[ "$LANES_PER_ARM" == 8 ]]

"$PY" "$REPO/tools/build_hy3_repack_importance.py" --self-test
"$PY" "$REPO/tools/prepare_mixed_expert_repack.py" test
"$PY" "$REPO/tools/merge_expert_overlay_fragments.py" --self-test
"$PY" "$REPO/research/per-expert-quant/validate_artifact.py" --self-test

for index in "${!arms[@]}"; do
  arm=${arms[$index]}
  plan=${plans[$index]}
  [[ -f "$plan" ]]
  "$PY" "$REPO/tools/build_hy3_repack_importance.py" \
    --plan "$plan" \
    --sidecar-dir "$IMPORTANCE_ROOT/sidecars" \
    --out-map "$IMPORTANCE_ROOT/$arm.json" \
    --out-plan "$BOUND_ROOT/$arm.json" \
    --ggml-lib "$ggml_lib" --ggml-lib-sha256 "$ggml_sha" \
    --ggml-source-commit "$ggml_commit"
done

nproc_total=$(nproc)
lane_total=$((${#arms[@]} * LANES_PER_ARM))
cpus_per_lane=$((nproc_total / lane_total))
((cpus_per_lane >= 4))
pids=()
for arm_index in "${!arms[@]}"; do
  arm=${arms[$arm_index]}
  out="$ARTIFACT_ROOT/$arm"
  fragment_root="$LOG_ROOT/fragments/$arm"
  mkdir -p "$out" "$fragment_root"
  for lane in $(seq 0 $((LANES_PER_ARM - 1))); do
    layers=$(
      "$PY" - "$lane" "$LANES_PER_ARM" <<'PY'
import sys
lane, lanes = map(int, sys.argv[1:])
print(",".join(str(layer) for layer in range(1, 80) if (layer - 1) % lanes == lane))
PY
    )
    global_lane=$((arm_index * LANES_PER_ARM + lane))
    cpu_start=$((global_lane * cpus_per_lane))
    cpu_end=$((cpu_start + cpus_per_lane - 1))
    fragment="$fragment_root/lane-$lane.json"
    taskset -c "$cpu_start-$cpu_end" nice -n 10 \
      "$PY" "$REPO/tools/prepare_mixed_expert_repack.py" prepare \
        "$SOURCE" "$out" --fallback-dir "$SOURCE" \
        --plan "$BOUND_ROOT/$arm.json" --layers "$layers" \
        --manifest-fragment "$fragment" --workers "$WORKERS_PER_LANE" \
        --max-work-mb 512 --resume \
        --ggml-lib "$ggml_lib" --ggml-lib-sha256 "$ggml_sha" \
        --ggml-source-commit "$ggml_commit" \
        >"$LOG_ROOT/$arm-lane-$lane.log" 2>&1 &
    pids+=("$!")
  done
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
((failed == 0)) || { echo "one or more matched repack lanes failed" >&2; exit 1; }

for arm in "${arms[@]}"; do
  fragment_args=()
  for lane in $(seq 0 $((LANES_PER_ARM - 1))); do
    fragment_args+=(--fragment "$LOG_ROOT/fragments/$arm/lane-$lane.json")
  done
  "$PY" "$REPO/tools/merge_expert_overlay_fragments.py" \
    "${fragment_args[@]}" --plan "$BOUND_ROOT/$arm.json" \
    --out-dir "$ARTIFACT_ROOT/$arm" --output "$ARTIFACT_ROOT/$arm/manifest.json"
  "$PY" "$REPO/research/per-expert-quant/validate_artifact.py" \
    "$ARTIFACT_ROOT/$arm" --verify-sources
done

"$PY" - "$ROOT/receipts/matched-repack.json" "$ARTIFACT_ROOT" "${arms[@]}" <<'PY'
import hashlib
import json
import pathlib
import sys
from datetime import datetime, timezone

output, artifact_root, *arms = sys.argv[1:]
root = pathlib.Path(artifact_root)
rows = []
for arm in arms:
    manifest = root / arm / "manifest.json"
    payload = json.loads(manifest.read_text())
    rows.append({
        "arm": arm,
        "manifest": str(manifest),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "artifact_bytes": payload["artifact_bytes"],
        "payload_bytes": payload["payload_bytes"],
        "plan_sha256": payload["plan_sha256"],
    })
pathlib.Path(output).write_text(json.dumps({
    "format": "bw24-hy3-110gb-matched-repack-v1",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "arms": rows,
    "public_eval_data_used_for_construction": False,
}, indent=2, sort_keys=True) + "\n")
PY
date -u +%Y-%m-%dT%H:%M:%SZ >"$LOG_ROOT/complete"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] matched repack complete"
