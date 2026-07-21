#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:?}
PY=${PY:-/data/venvs/hy3-110gb/bin/python}
ROOT=${ROOT:-/data/experiments/hy3-110gb}
SOURCE=${SOURCE:-/opt/dlami/nvme/models/hy3-source}
SOURCE_RECEIPT=${SOURCE_RECEIPT:-$ROOT/receipts/source-download.json}
QUANTIZER_RECEIPT=${QUANTIZER_RECEIPT:-$ROOT/receipts/llama-cpp-quantizer.json}
ARM=${ARM:?}
PLAN=${PLAN:?}
LANES=${LANES:-8}
WORKERS_PER_LANE=${WORKERS_PER_LANE:-20}
LOG_ROOT=${LOG_ROOT:-$ROOT/logs/single-arm-repack/$ARM}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-$ROOT/artifacts/matched-repack}
BOUND_ROOT=${BOUND_ROOT:-$ROOT/plans/repack-bound}
IMPORTANCE_ROOT=${IMPORTANCE_ROOT:-$ROOT/calibration/repack-importance}

[[ "$LANES" == 8 ]]
mkdir -p "$LOG_ROOT" "$ARTIFACT_ROOT/$ARM" "$BOUND_ROOT" "$IMPORTANCE_ROOT"
exec 9>"$LOG_ROOT/transition.lock"
flock -n 9 || { echo "another repack owns $LOG_ROOT" >&2; exit 1; }
[[ ! -f "$LOG_ROOT/complete" ]] || exit 0
exec > >(tee -a "$LOG_ROOT/transition.log") 2>&1

jq -e '.complete == true and .shard_count == 99' "$SOURCE_RECEIPT" >/dev/null
ggml_lib=$(jq -r .library "$QUANTIZER_RECEIPT")
ggml_sha=$(jq -r .library_sha256 "$QUANTIZER_RECEIPT")
ggml_commit=$(jq -r .llama_cpp_commit "$QUANTIZER_RECEIPT")
[[ -f "$ggml_lib" && $(sha256sum "$ggml_lib" | awk '{print $1}') == "$ggml_sha" ]]
[[ -f "$PLAN" ]]
export LD_LIBRARY_PATH="$(dirname "$ggml_lib"):${LD_LIBRARY_PATH:-}"

"$PY" "$REPO/tools/build_hy3_repack_importance.py" --self-test
"$PY" "$REPO/tools/prepare_mixed_expert_repack.py" test
"$PY" "$REPO/tools/merge_expert_overlay_fragments.py" --self-test
"$PY" "$REPO/research/per-expert-quant/validate_artifact.py" --self-test

"$PY" "$REPO/tools/build_hy3_repack_importance.py" \
  --plan "$PLAN" --sidecar-dir "$IMPORTANCE_ROOT/sidecars" \
  --out-map "$IMPORTANCE_ROOT/$ARM.json" --out-plan "$BOUND_ROOT/$ARM.json" \
  --ggml-lib "$ggml_lib" --ggml-lib-sha256 "$ggml_sha" \
  --ggml-source-commit "$ggml_commit"

cpus_per_lane=$(( $(nproc) / LANES ))
((cpus_per_lane >= WORKERS_PER_LANE))
pids=()
fragment_root="$LOG_ROOT/fragments"
mkdir -p "$fragment_root"
for lane in $(seq 0 $((LANES - 1))); do
  layers=$(
    "$PY" - "$lane" "$LANES" <<'PY'
import sys
lane, lanes = map(int, sys.argv[1:])
print(",".join(str(layer) for layer in range(1, 80) if (layer - 1) % lanes == lane))
PY
  )
  cpu_start=$((lane * cpus_per_lane)); cpu_end=$((cpu_start + cpus_per_lane - 1))
  taskset -c "$cpu_start-$cpu_end" nice -n 5 \
    "$PY" "$REPO/tools/prepare_mixed_expert_repack.py" prepare \
      "$SOURCE" "$ARTIFACT_ROOT/$ARM" --fallback-dir "$SOURCE" \
      --plan "$BOUND_ROOT/$ARM.json" --layers "$layers" \
      --manifest-fragment "$fragment_root/lane-$lane.json" \
      --workers "$WORKERS_PER_LANE" --max-work-mb 512 --resume \
      --ggml-lib "$ggml_lib" --ggml-lib-sha256 "$ggml_sha" \
      --ggml-source-commit "$ggml_commit" \
      >"$LOG_ROOT/lane-$lane.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
((failed == 0)) || { echo "one or more repack lanes failed" >&2; exit 1; }

fragment_args=()
for lane in $(seq 0 $((LANES - 1))); do
  fragment_args+=(--fragment "$fragment_root/lane-$lane.json")
done
"$PY" "$REPO/tools/merge_expert_overlay_fragments.py" \
  "${fragment_args[@]}" --plan "$BOUND_ROOT/$ARM.json" \
  --out-dir "$ARTIFACT_ROOT/$ARM" --output "$ARTIFACT_ROOT/$ARM/manifest.json"
"$PY" "$REPO/research/per-expert-quant/validate_artifact.py" \
  "$ARTIFACT_ROOT/$ARM" --verify-sources

"$PY" - "$LOG_ROOT/receipt.json" "$ARTIFACT_ROOT/$ARM/manifest.json" "$PLAN" <<'PY'
import hashlib, json, pathlib, sys
from datetime import datetime, timezone
output, manifest_path, plan_path = map(pathlib.Path, sys.argv[1:])
manifest = json.loads(manifest_path.read_text())
output.write_text(json.dumps({
    "format": "bw24-hy3-single-arm-repack-v1",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "manifest": str(manifest_path),
    "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    "source_plan": str(plan_path),
    "source_plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
    "bound_plan_sha256": manifest["plan_sha256"],
    "artifact_bytes": manifest["artifact_bytes"],
    "payload_bytes": manifest["payload_bytes"],
    "public_eval_data_used_for_construction": False,
}, indent=2, sort_keys=True) + "\n")
PY
sha256sum "$PLAN" "$BOUND_ROOT/$ARM.json" "$ARTIFACT_ROOT/$ARM/manifest.json" \
  "$LOG_ROOT"/fragments/*.json "$LOG_ROOT"/lane-*.log >"$LOG_ROOT/evidence.sha256"
date -u +%FT%TZ >"$LOG_ROOT/complete"
echo "single-arm repack complete: $ARM"
