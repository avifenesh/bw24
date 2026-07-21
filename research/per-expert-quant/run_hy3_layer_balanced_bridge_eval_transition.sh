#!/usr/bin/env bash
set -euo pipefail

# Evaluate both privately frozen bridge budgets after their build/heal evidence is complete. Public
# results enter only the frozen promotion selector and never feed back into model construction.

ROOT=${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
PY=${PY:-python3}
EXPECTED_COMMIT=${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the detached orchestration commit}
BUILD_COMPLETE=${BUILD_COMPLETE:-/data/logs/layer-balanced-bridge-build/complete}
BUILD_EVIDENCE=${BUILD_EVIDENCE:-/data/logs/layer-balanced-bridge-build/evidence.sha256}
ART_ROOT=${ART_ROOT:-/scratch/bw24-artifacts-layer-balanced-bridge}
PLAN_ROOT=${PLAN_ROOT:-/data/plans/per-expert-quant-layer-balanced-bridge}
SELECTION_LOCK=${SELECTION_LOCK:-$ROOT/research/per-expert-quant/layer-balanced-bridge.lock.json}
PROMOTION_LOCK=${PROMOTION_LOCK:-$ROOT/research/per-expert-quant/layer-balanced-bridge-promotion-gates.lock.json}
BASE_DIRECTIONAL_ROOT=${BASE_DIRECTIONAL_ROOT:-/data/results/per-expert-quant/layer-balanced100-directional-v1}
BASE_DIRECTIONAL_COMPLETE=${BASE_DIRECTIONAL_COMPLETE:-/data/logs/layer-balanced100-directional-v1/complete}
EVAL_ROOT=${EVAL_ROOT:-/data/src/bw24-eval-valid-c8689ec}
EVAL_COMMIT=${EVAL_COMMIT:-c8689ec937c899fbdbd399432ec175e7d48b53ae}
BASE_EVAL_COMMIT=${BASE_EVAL_COMMIT:-ae89c11975d7d51bbce0a56ae5963ad42dc68a6a}
PANEL_LOCK=${PANEL_LOCK:-$EVAL_ROOT/research/per-expert-quant/expanded-capability-panel.lock.json}
PANEL_SHA256=${PANEL_SHA256:-33ca7c2a86ed52ab3ee06ec408ceda890e50447e5cc4a204a755afcd3368c64b}
SERVER_BIN=${SERVER_BIN:-/data/build/bw24-portable-ada-fix-target/release/bw24-server}
SERVER_SHA256=${SERVER_SHA256:-13a7ac6a15a5de17f0eb736ecb393a50ab9bf03145e86a878e66c86b2086f195}
OUT_ROOT=${OUT_ROOT:-/data/results/per-expert-quant/layer-balanced-bridge-directional-v1}
NORMALIZED_OUT_ROOT=${NORMALIZED_OUT_ROOT:-/data/results/per-expert-quant/layer-balanced-bridge-directional-normalized-v1}
LOG_ROOT=${LOG_ROOT:-/data/logs/layer-balanced-bridge-directional-v1}
CACHE_DIR=${CACHE_DIR:-/data/cache/per-expert-evals}
PRIVATE_GATE_ROOT=${PRIVATE_GATE_ROOT:-/data/logs/layer-balanced-bridge-private-artifact-gate}
SCORER_LOCK=${SCORER_LOCK:-/tmp/bw24-layer-balanced-bridge-scorer.lock}
RESUME_RUN_ID=${RESUME_RUN_ID:-}

arms=(layer_balanced120 layer_balanced137)
gpus=(0 1)
ports=(8080 8081)
cpu_starts=(0 12)
cpu_ends=(11 23)
numas=(0 0)

die() { echo "layer-balanced bridge eval transition: $*" >&2; exit 1; }
port_listening() { ss -H -ltn "sport = :$1" 2>/dev/null | grep -q .; }
gpu_busy() {
  nvidia-smi -i "$1" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | grep -Eq '^[0-9]+$'
}

mkdir -p "$OUT_ROOT/run-configs" "$NORMALIZED_OUT_ROOT" "$LOG_ROOT"
exec 9>"$LOG_ROOT/transition.lock"
flock -n 9 || die "another bridge directional transition owns the lock"
echo "$(date -u +%FT%TZ) layer-balanced bridge directional transition started" \
  | tee -a "$LOG_ROOT/transition.log"
[[ $(git -C "$ROOT" rev-parse HEAD) == "$EXPECTED_COMMIT" ]] || die "source commit mismatch"
[[ -z $(git -C "$ROOT" status --porcelain) ]] || die "source checkout is dirty"
[[ $(git -C "$EVAL_ROOT" rev-parse HEAD) == "$EVAL_COMMIT" ]] || die "eval commit mismatch"
[[ -z $(git -C "$EVAL_ROOT" status --porcelain) ]] || die "eval checkout is dirty"
[[ $(sha256sum "$PANEL_LOCK" | cut -d' ' -f1) == "$PANEL_SHA256" ]] || die "panel mismatch"
[[ $(sha256sum "$SERVER_BIN" | cut -d' ' -f1) == "$SERVER_SHA256" ]] || die "server mismatch"
"$PY" "$ROOT/research/per-expert-quant/validate_layer_balanced_bridge_lock.py" \
  --lock "$SELECTION_LOCK" --verify-inputs
"$PY" "$ROOT/research/per-expert-quant/select_layer_balanced_bridge_promotions.py" --self-test

while [[ ! -f "$BUILD_COMPLETE" ]]; do sleep 30; done
while [[ ! -f "$BASE_DIRECTIONAL_COMPLETE" ]]; do sleep 30; done
for arm in "${arms[@]}"; do
  for path in "$ART_ROOT/$arm/manifest.json" "$PLAN_ROOT/$arm.json"; do
    [[ -f "$path" ]] || die "missing bridge prerequisite $path"
  done
done
for path in "$BUILD_EVIDENCE" "$SELECTION_LOCK" "$PROMOTION_LOCK"; do
  [[ -f "$path" ]] || die "missing bridge evidence $path"
done
sha256sum -c "$BUILD_EVIDENCE" >"$LOG_ROOT/build-evidence-check.log"

if [[ -z "$RESUME_RUN_ID" ]]; then
  for lane in "${!arms[@]}"; do
    port_listening "${ports[$lane]}" && die "eval port ${ports[$lane]} is in use"
    port_listening "$((8070 + lane))" && die "private gate port $((8070 + lane)) is in use"
    gpu_busy "${gpus[$lane]}" && die "eval GPU ${gpus[$lane]} is in use"
  done
  if [[ ! -f "$PRIVATE_GATE_ROOT/complete" ]]; then
    SERVER_BIN="$SERVER_BIN" ART_ROOT="$ART_ROOT" \
      REQUESTS=/data/calibration/hy3-confidence-v1/requests.jsonl OUT_ROOT="$PRIVATE_GATE_ROOT" \
      PY="$PY" CAPTURE_TOOL="$EVAL_ROOT/research/per-expert-quant/capture_calibration.py" \
      HEALTH_TOOL="$EVAL_ROOT/research/per-expert-quant/validate_server_health.py" \
      ROUTE_VALIDATOR="$ROOT/research/per-expert-quant/validate_pruned_route_trace.py" \
      GPU_BASE=0 PORT_BASE=8070 CPU_BASE=0 ARMS_CSV=layer_balanced120,layer_balanced137 \
      "$ROOT/research/per-expert-quant/run_100gb_private_artifact_gate.sh"
  fi
  RUN_ID=layer-balanced-bridge-v1-$(git -C "$ROOT" rev-parse --short=7 HEAD)-$(date -u +%Y%m%dT%H%M%SZ)
  printf '%s\n' "$RUN_ID" >"$OUT_ROOT/_active-run-id"
  "$PY" - "$OUT_ROOT/run-configs/$RUN_ID.json" "$RUN_ID" "$ROOT" "$EVAL_ROOT" \
    "$PANEL_LOCK" "$SERVER_BIN" "$ART_ROOT" "$PLAN_ROOT" "$SELECTION_LOCK" \
    "$PROMOTION_LOCK" <<'PY'
import hashlib,json,pathlib,subprocess,sys
out,run_id,root,eval_root,panel,server,art_root,plan_root,selection_lock,promotion_lock=sys.argv[1:]
sha=lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
arms={}
for lane,arm in enumerate(("layer_balanced120","layer_balanced137")):
    artifact=pathlib.Path(art_root)/arm; plan=pathlib.Path(plan_root)/f"{arm}.json"
    arms[arm]={"lane":{"gpu":lane,"port":8080+lane,"cpu_start":lane*12,
                       "cpu_end":lane*12+11,"numa":0},
               "artifact":{"path":str(artifact.resolve()),
                           "manifest_sha256":sha(artifact/"manifest.json")},
               "plan":{"path":str(plan.resolve()),"sha256":sha(plan)}}
d={"format":"bw24-layer-balanced-bridge-directional-run-v1","run_id":run_id,
   "orchestration_commit":subprocess.check_output(
       ["git","-C",root,"rev-parse","HEAD"],text=True).strip(),
   "eval_commit":subprocess.check_output(
       ["git","-C",eval_root,"rev-parse","HEAD"],text=True).strip(),
   "panel":{"path":str(pathlib.Path(panel).resolve()),"sha256":sha(panel)},
   "server":{"path":str(pathlib.Path(server).resolve()),"sha256":sha(server)},
   "settings":{"mtp":False,"spec":False,"kv_reuse":False,"concurrency":1,"spill_depth":8},
   "selection_lock":{"path":str(pathlib.Path(selection_lock).resolve()),"sha256":sha(selection_lock)},
   "promotion_lock":{"path":str(pathlib.Path(promotion_lock).resolve()),"sha256":sha(promotion_lock)},
   "public_eval_data_used_for_model_construction":False,"arms":arms}
pathlib.Path(out).write_text(json.dumps(d,indent=2,sort_keys=True)+"\n")
PY
else
  [[ "$RESUME_RUN_ID" =~ ^layer-balanced-bridge-v1-[A-Za-z0-9._-]+$ ]] \
    || die "invalid RESUME_RUN_ID"
  [[ -f "$OUT_ROOT/_active-run-id" && $(<"$OUT_ROOT/_active-run-id") == "$RESUME_RUN_ID" ]] \
    || die "RESUME_RUN_ID does not match the active run"
  RUN_ID=$RESUME_RUN_ID
  [[ -f "$OUT_ROOT/run-configs/$RUN_ID.json" ]] || die "missing immutable run config"
fi

run_arm() {
  local lane=$1 arm=${arms[$lane]}
  CUDA_VISIBLE_DEVICES="${gpus[$lane]}" taskset -c "${cpu_starts[$lane]}-${cpu_ends[$lane]}" \
    numactl --membind="${numas[$lane]}" \
    env ARM="$arm" ARTIFACT="$ART_ROOT/$arm" RUN_ID="$RUN_ID" SERVER_BIN="$SERVER_BIN" \
      OUT_ROOT="$OUT_ROOT" CACHE_DIR="$CACHE_DIR" PANEL_LOCK="$PANEL_LOCK" \
      ADDR="127.0.0.1:${ports[$lane]}" BW24_SPILL_PREAD_DEPTH=8 \
      HOURISH_SCORER_LOCK="$SCORER_LOCK" \
      "$EVAL_ROOT/research/per-expert-quant/run_hourish_one_arm.sh" \
      >"$LOG_ROOT/$RUN_ID-$arm.launcher.log" 2>&1
}
pids=()
for lane in "${!arms[@]}"; do run_arm "$lane" & pids+=("$!"); done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
((failed == 0)) || die "one or more bridge eval arms failed"
for port in "${ports[@]}"; do port_listening "$port" && die "server remained on eval port $port"; done

summary="$OUT_ROOT/layer-balanced-bridge-summary-$RUN_ID.json"
if [[ ! -f "$summary" ]]; then
  "$PY" "$EVAL_ROOT/research/per-expert-quant/summarize_hourish_results.py" \
    --out-root "$OUT_ROOT" --run-id "$RUN_ID" --arms "${arms[@]}" --baseline layer_balanced120 \
    --panel-lock "$PANEL_LOCK" --suite-lock "$EVAL_ROOT/research/per-expert-quant/suite.lock.json" \
    --server-sha256 "$SERVER_SHA256" --output "$summary"
fi

base_run=$(<"$BASE_DIRECTIONAL_ROOT/_active-run-id")
base_frontier="$BASE_DIRECTIONAL_ROOT/layer-balanced100-frontier-$base_run.json"
[[ -f "$base_frontier" ]] || die "missing Layer100 combined frontier"
for arm in "${arms[@]}"; do
  SOURCE_RUN_DIR="$OUT_ROOT/$arm/$RUN_ID" OUT_ROOT="$NORMALIZED_OUT_ROOT" \
    ARM="$arm" RUN_ID="$RUN_ID" BASELINE_FRONTIER="$base_frontier" \
    BASELINE_ARM=plain_quant PANEL_LOCK="$PANEL_LOCK" \
    "$ROOT/research/per-expert-quant/normalize_hourish_scorer_evidence.sh"
done
normalized_summary="$NORMALIZED_OUT_ROOT/layer-balanced-bridge-summary-$RUN_ID.json"
if [[ ! -f "$normalized_summary" ]]; then
  "$PY" "$EVAL_ROOT/research/per-expert-quant/summarize_hourish_results.py" \
    --out-root "$NORMALIZED_OUT_ROOT" --run-id "$RUN_ID" --arms "${arms[@]}" \
    --baseline layer_balanced120 --panel-lock "$PANEL_LOCK" \
    --suite-lock "$EVAL_ROOT/research/per-expert-quant/suite.lock.json" \
    --server-sha256 "$SERVER_SHA256" --output "$normalized_summary"
fi
mapfile -t base_specs < <("$PY" - "$base_frontier" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); assert d["format"] == "bw24-cross-run-expanded-capability-frontier-v1"
for item in d["source_runs"]:
    print(f'{item["arm"]}={item["out_root"]}::{item["run_id"]}')
PY
)
frontier="$OUT_ROOT/layer-balanced-bridge-frontier-$RUN_ID.json"
frontier_args=(--panel-lock "$PANEL_LOCK" --server-sha256 "$SERVER_SHA256" \
  --compatible-bw24-commits "$BASE_EVAL_COMMIT=$EVAL_COMMIT")
for spec in "${base_specs[@]}"; do frontier_args+=(--arm "$spec"); done
for arm in "${arms[@]}"; do
  frontier_args+=(--arm "$arm=$NORMALIZED_OUT_ROOT::$RUN_ID")
done
frontier_args+=(--baseline plain_quant --output "$frontier")
if [[ ! -f "$frontier" ]]; then
  "$PY" "$ROOT/research/per-expert-quant/summarize_cross_run_hourish.py" "${frontier_args[@]}"
fi
promotion="$OUT_ROOT/layer-balanced-bridge-promotion-$RUN_ID.json"
if [[ ! -f "$promotion" ]]; then
  "$PY" "$ROOT/research/per-expert-quant/select_layer_balanced_bridge_promotions.py" \
    --frontier "$frontier" --lock "$PROMOTION_LOCK" --output "$promotion"
fi
sha256sum "$OUT_ROOT/run-configs/$RUN_ID.json" "$summary" "$normalized_summary" \
  "$NORMALIZED_OUT_ROOT"/*/"$RUN_ID"/scorer-normalization.evidence.sha256 \
  "$frontier" "$promotion" "$SELECTION_LOCK" "$PROMOTION_LOCK" \
  "$PRIVATE_GATE_ROOT/evidence.sha256" "$BUILD_EVIDENCE" >"$LOG_ROOT/evidence-$RUN_ID.sha256"
sha256sum -c "$LOG_ROOT/evidence-$RUN_ID.sha256"
date -u +%FT%TZ | tee "$LOG_ROOT/complete"
