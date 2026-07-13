#!/usr/bin/env bash
set -euo pipefail

# Evaluate the privately preregistered layer-balanced exact-100GB candidate on a lane that may
# overlap unrelated GPU work. Public results are consumed only after the immutable build/heal gate;
# they never flow back into the allocation, quantization, or healing inputs.

ROOT=${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
PY=${PY:-python3}
EXPECTED_COMMIT=${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the detached orchestration commit}
ARM=${ARM:-layer_balanced100}
BUILD_COMPLETE=${BUILD_COMPLETE:-/data/logs/layer-balanced100-3db293f/complete}
BUILD_EVIDENCE=${BUILD_EVIDENCE:-/data/logs/layer-balanced100-3db293f/evidence.sha256}
ART_ROOT=${ART_ROOT:-/scratch/bw24-artifacts-layer-balanced100-3db293f}
PLAN=${PLAN:-/data/plans/per-expert-quant-layer-balanced100-3db293f/layer-balanced100.plan.json}
CONSTRAINTS=${CONSTRAINTS:-/data/plans/per-expert-quant-layer-balanced100-3db293f/layer-balanced100.constraints.json}
BASE_DIRECTIONAL_ROOT=${BASE_DIRECTIONAL_ROOT:-/data/results/per-expert-quant/iq3-iq4-q4-pareto-directional-v1}
BASE_DIRECTIONAL_COMPLETE=${BASE_DIRECTIONAL_COMPLETE:-/data/logs/iq3-iq4-q4-pareto-directional-v1/complete}
EVAL_ROOT=${EVAL_ROOT:-/data/src/bw24-eval-valid-c8689ec}
EVAL_COMMIT=${EVAL_COMMIT:-c8689ec937c899fbdbd399432ec175e7d48b53ae}
PANEL_LOCK=${PANEL_LOCK:-$EVAL_ROOT/research/per-expert-quant/expanded-capability-panel.lock.json}
PANEL_SHA256=${PANEL_SHA256:-33ca7c2a86ed52ab3ee06ec408ceda890e50447e5cc4a204a755afcd3368c64b}
SERVER_BIN=${SERVER_BIN:-/data/build/bw24-portable-ada-fix-target/release/bw24-server}
SERVER_SHA256=${SERVER_SHA256:-13a7ac6a15a5de17f0eb736ecb393a50ab9bf03145e86a878e66c86b2086f195}
OUT_ROOT=${OUT_ROOT:-/data/results/per-expert-quant/layer-balanced100-directional-v1}
LOG_ROOT=${LOG_ROOT:-/data/logs/layer-balanced100-directional-v1}
CACHE_DIR=${CACHE_DIR:-/data/cache/per-expert-evals}
PRIVATE_GATE_ROOT=${PRIVATE_GATE_ROOT:-/data/logs/layer-balanced100-private-artifact-gate-3db293f}
PROMOTION_LOCK=${PROMOTION_LOCK:-$ROOT/research/per-expert-quant/layer-balanced100-promotion-gates.lock.json}
EVAL_GPU=${EVAL_GPU:-4}
EVAL_PORT=${EVAL_PORT:-8084}
PRIVATE_GATE_PORT=${PRIVATE_GATE_PORT:-8074}
EVAL_CPU_START=${EVAL_CPU_START:-48}
EVAL_CPU_END=${EVAL_CPU_END:-59}
EVAL_NUMA=${EVAL_NUMA:-1}
SCORER_LOCK=${SCORER_LOCK:-/tmp/bw24-layer-balanced100-scorer.lock}

die() { echo "layer-balanced100 eval transition: $*" >&2; exit 1; }
port_listening() { ss -H -ltn "sport = :$1" 2>/dev/null | grep -q .; }
gpu_busy() {
  nvidia-smi -i "$1" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | grep -Eq '^[0-9]+$'
}

mkdir -p "$OUT_ROOT/run-configs" "$LOG_ROOT"
exec 9>"$LOG_ROOT/transition.lock"
flock -n 9 || die "another layer-balanced100 directional transition owns the lock"
echo "$(date -u +%FT%TZ) layer-balanced100 directional transition started" \
  | tee -a "$LOG_ROOT/transition.log"
[[ $(git -C "$ROOT" rev-parse HEAD) == "$EXPECTED_COMMIT" ]] || die "source commit mismatch"
[[ -z $(git -C "$ROOT" status --porcelain) ]] || die "source checkout is dirty"
[[ $(git -C "$EVAL_ROOT" rev-parse HEAD) == "$EVAL_COMMIT" ]] || die "eval commit mismatch"
[[ -z $(git -C "$EVAL_ROOT" status --porcelain) ]] || die "eval checkout is dirty"
[[ $(sha256sum "$PANEL_LOCK" | cut -d' ' -f1) == "$PANEL_SHA256" ]] || die "panel mismatch"
[[ $(sha256sum "$SERVER_BIN" | cut -d' ' -f1) == "$SERVER_SHA256" ]] || die "server mismatch"
[[ -f "$PROMOTION_LOCK" ]] || die "missing frozen layer-balanced100 promotion lock"
[[ "$EVAL_GPU" =~ ^[0-9]+$ && "$EVAL_PORT" =~ ^[1-9][0-9]{0,4}$ \
  && "$PRIVATE_GATE_PORT" =~ ^[1-9][0-9]{0,4}$ && "$EVAL_CPU_START" =~ ^[0-9]+$ \
  && "$EVAL_CPU_END" =~ ^[0-9]+$ && "$EVAL_NUMA" =~ ^[0-9]+$ \
  && "$EVAL_PORT" -le 65535 && "$PRIVATE_GATE_PORT" -le 65535 \
  && "$EVAL_CPU_START" -le "$EVAL_CPU_END" ]] || die "invalid eval lane configuration"

while [[ ! -f "$BUILD_COMPLETE" ]]; do sleep 30; done
artifact="$ART_ROOT/$ARM"
for path in "$artifact/manifest.json" "$BUILD_EVIDENCE" "$PLAN" "$CONSTRAINTS"; do
  [[ -f "$path" ]] || die "missing layer-balanced100 prerequisite $path"
done
sha256sum -c "$BUILD_EVIDENCE" >"$LOG_ROOT/build-evidence-check.log"
port_listening "$PRIVATE_GATE_PORT" && die "private gate port $PRIVATE_GATE_PORT is already in use"
port_listening "$EVAL_PORT" && die "eval port $EVAL_PORT is already in use"
gpu_busy "$EVAL_GPU" && die "eval GPU $EVAL_GPU is already in use"

if [[ ! -f "$PRIVATE_GATE_ROOT/complete" ]]; then
  SERVER_BIN="$SERVER_BIN" ART_ROOT="$ART_ROOT" \
    REQUESTS=/data/calibration/hy3-confidence-v1/requests.jsonl OUT_ROOT="$PRIVATE_GATE_ROOT" \
    PY="$PY" CAPTURE_TOOL="$EVAL_ROOT/research/per-expert-quant/capture_calibration.py" \
    HEALTH_TOOL="$EVAL_ROOT/research/per-expert-quant/validate_server_health.py" \
    ROUTE_VALIDATOR="$ROOT/research/per-expert-quant/validate_pruned_route_trace.py" \
    GPU_BASE="$EVAL_GPU" PORT_BASE="$PRIVATE_GATE_PORT" CPU_BASE="$EVAL_CPU_START" \
    ARMS_CSV="$ARM" "$ROOT/research/per-expert-quant/run_100gb_private_artifact_gate.sh"
fi

RUN_ID=layer-balanced100-v1-$(git -C "$ROOT" rev-parse --short=7 HEAD)-$(date -u +%Y%m%dT%H%M%SZ)
printf '%s\n' "$RUN_ID" >"$OUT_ROOT/_active-run-id"
"$PY" - "$OUT_ROOT/run-configs/$RUN_ID.json" "$RUN_ID" "$ROOT" "$EVAL_ROOT" \
  "$PANEL_LOCK" "$SERVER_BIN" "$artifact" "$ARM" "$PLAN" "$CONSTRAINTS" \
  "$EVAL_GPU" "$EVAL_PORT" "$EVAL_CPU_START" "$EVAL_CPU_END" "$EVAL_NUMA" <<'PY'
import hashlib,json,pathlib,subprocess,sys
out,run_id,root,eval_root,panel,server,artifact,arm,plan,constraints,gpu,port,cpu_start,cpu_end,numa=sys.argv[1:]
sha=lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
d={"format":"bw24-layer-balanced100-directional-run-v1","run_id":run_id,
   "orchestration_commit":subprocess.check_output(
       ["git","-C",root,"rev-parse","HEAD"],text=True).strip(),
   "eval_commit":subprocess.check_output(
       ["git","-C",eval_root,"rev-parse","HEAD"],text=True).strip(),
   "panel":{"path":str(pathlib.Path(panel).resolve()),"sha256":sha(panel)},
   "server":{"path":str(pathlib.Path(server).resolve()),"sha256":sha(server)},
   "settings":{"mtp":False,"spec":False,"kv_reuse":False,"concurrency":1,"spill_depth":8},
   "lane":{"gpu":int(gpu),"port":int(port),"cpu_start":int(cpu_start),
           "cpu_end":int(cpu_end),"numa":int(numa)},
   "artifact":{"arm":arm,"path":str(pathlib.Path(artifact).resolve()),
               "manifest_sha256":sha(pathlib.Path(artifact)/"manifest.json")},
   "private_policy":{
       "plan":{"path":str(pathlib.Path(plan).resolve()),"sha256":sha(plan)},
       "constraints":{"path":str(pathlib.Path(constraints).resolve()),"sha256":sha(constraints)},
       "public_eval_data_used_for_selection":False}}
pathlib.Path(out).write_text(json.dumps(d,indent=2,sort_keys=True)+"\n")
PY

CUDA_VISIBLE_DEVICES="$EVAL_GPU" taskset -c "$EVAL_CPU_START-$EVAL_CPU_END" \
  numactl --membind="$EVAL_NUMA" \
  env ARM="$ARM" ARTIFACT="$artifact" RUN_ID="$RUN_ID" SERVER_BIN="$SERVER_BIN" \
    OUT_ROOT="$OUT_ROOT" CACHE_DIR="$CACHE_DIR" PANEL_LOCK="$PANEL_LOCK" \
    ADDR="127.0.0.1:$EVAL_PORT" BW24_SPILL_PREAD_DEPTH=8 \
    HOURISH_SCORER_LOCK="$SCORER_LOCK" \
    "$EVAL_ROOT/research/per-expert-quant/run_hourish_one_arm.sh" \
    >"$LOG_ROOT/$RUN_ID-$ARM.launcher.log" 2>&1
port_listening "$EVAL_PORT" && die "server remained on eval port $EVAL_PORT"
summary="$OUT_ROOT/layer-balanced100-summary-$RUN_ID.json"
"$PY" "$EVAL_ROOT/research/per-expert-quant/summarize_hourish_results.py" \
  --out-root "$OUT_ROOT" --run-id "$RUN_ID" --arms "$ARM" --baseline "$ARM" \
  --panel-lock "$PANEL_LOCK" --suite-lock "$EVAL_ROOT/research/per-expert-quant/suite.lock.json" \
  --server-sha256 "$SERVER_SHA256" --output "$summary"

while [[ ! -f "$BASE_DIRECTIONAL_COMPLETE" ]]; do sleep 30; done
base_run=$(<"$BASE_DIRECTIONAL_ROOT/_active-run-id")
base_frontier="$BASE_DIRECTIONAL_ROOT/iq3-iq4-q4-frontier-$base_run.json"
[[ -f "$base_frontier" ]] || die "missing previous combined frontier"
mapfile -t base_specs < <("$PY" - "$base_frontier" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
assert d["format"] == "bw24-cross-run-expanded-capability-frontier-v1"
for item in d["source_runs"]:
    print(f'{item["arm"]}={item["out_root"]}::{item["run_id"]}')
PY
)
frontier="$OUT_ROOT/layer-balanced100-frontier-$RUN_ID.json"
frontier_args=(--panel-lock "$PANEL_LOCK" --server-sha256 "$SERVER_SHA256")
for spec in "${base_specs[@]}"; do frontier_args+=(--arm "$spec"); done
frontier_args+=(--arm "$ARM=$OUT_ROOT::$RUN_ID" --baseline plain_quant --output "$frontier")
"$PY" "$ROOT/research/per-expert-quant/summarize_cross_run_hourish.py" "${frontier_args[@]}"
promotion="$OUT_ROOT/layer-balanced100-promotion-$RUN_ID.json"
"$PY" "$ROOT/research/per-expert-quant/select_smart100_promotions.py" \
  --frontier "$frontier" --lock "$PROMOTION_LOCK" --output "$promotion"
practical="$OUT_ROOT/layer-balanced100-practical-input-$RUN_ID.json"
"$PY" - "$promotion" "$practical" <<'PY'
import json,pathlib,sys
d=json.load(open(sys.argv[1])); arms=d["practical_arms"]
assert arms[:2] == ["plain_quant","traffic_nvfp4_53_q2_139"] and 2 <= len(arms) <= 3
pathlib.Path(sys.argv[2]).write_text(json.dumps({
    "format":"bw24-100gb-directional-promotion-v1",
    "source_format":d["format"],"practical_arms":arms,
    "selected_100gb_arms":d["selected_practical_candidates"],
},indent=2,sort_keys=True)+"\n")
PY
sha256sum "$OUT_ROOT/run-configs/$RUN_ID.json" "$summary" "$frontier" "$promotion" \
  "$practical" "$PROMOTION_LOCK" "$PLAN" "$CONSTRAINTS" \
  "$PRIVATE_GATE_ROOT/evidence.sha256" "$BUILD_EVIDENCE" >"$LOG_ROOT/evidence-$RUN_ID.sha256"
sha256sum -c "$LOG_ROOT/evidence-$RUN_ID.sha256"
date -u +%FT%TZ | tee "$LOG_ROOT/complete"
