#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
PY=${PY:-python3}
EXPECTED_COMMIT=${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the detached orchestration commit}
BUILD_COMPLETE=${BUILD_COMPLETE:-/data/logs/smart100-build-2605fde/complete}
ART_ROOT=${ART_ROOT:-/scratch/bw24-artifacts-smart100-2605fde}
CONTROL_ARTIFACT=${CONTROL_ARTIFACT:-/scratch/bw24-artifacts-100gb-5f02c37/prune100_joint_heal}
EVAL_ROOT=${EVAL_ROOT:-/data/src/bw24-expanded-panel}
EVAL_COMMIT=${EVAL_COMMIT:-ae89c11975d7d51bbce0a56ae5963ad42dc68a6a}
PANEL_LOCK=${PANEL_LOCK:-$EVAL_ROOT/research/per-expert-quant/expanded-capability-panel.lock.json}
PANEL_SHA256=${PANEL_SHA256:-33ca7c2a86ed52ab3ee06ec408ceda890e50447e5cc4a204a755afcd3368c64b}
SERVER_BIN=${SERVER_BIN:-/data/build/bw24-portable-ada-fix-target/release/bw24-server}
SERVER_SHA256=${SERVER_SHA256:-13a7ac6a15a5de17f0eb736ecb393a50ab9bf03145e86a878e66c86b2086f195}
OUT_ROOT=${OUT_ROOT:-/data/results/per-expert-quant/smart100-directional-v1}
LOG_ROOT=${LOG_ROOT:-/data/logs/smart100-directional-v1}
CACHE_DIR=${CACHE_DIR:-/data/cache/per-expert-evals}
PRIVATE_GATE_ROOT=${PRIVATE_GATE_ROOT:-/data/logs/smart100-private-artifact-gate-2605fde}

arms=(prune100_joint_heal smart100_empirical smart100_balanced smart100_rescue)
artifacts=("$CONTROL_ARTIFACT" "$ART_ROOT/smart100_empirical" "$ART_ROOT/smart100_balanced" "$ART_ROOT/smart100_rescue")
gpus=(0 2 4 6)
cpus=(0-11 24-35 48-59 72-83)
numas=(0 0 1 1)
ports=(8080 8081 8082 8083)

die() { echo "smart100 eval transition: $*" >&2; exit 1; }
mkdir -p "$OUT_ROOT" "$LOG_ROOT"
exec 9>"$LOG_ROOT/transition.lock"
flock -n 9 || die "another transition owns $LOG_ROOT/transition.lock"
echo "$(date -u +%FT%TZ) smart100 directional transition started" | tee -a "$LOG_ROOT/transition.log"
[[ $(git -C "$ROOT" rev-parse HEAD) == "$EXPECTED_COMMIT" ]] || die "orchestration commit mismatch"
while [[ ! -f "$BUILD_COMPLETE" ]]; do sleep 30; done
[[ $(git -C "$EVAL_ROOT" rev-parse HEAD) == "$EVAL_COMMIT" ]] || die "eval tooling commit mismatch"
[[ $(sha256sum "$PANEL_LOCK" | cut -d' ' -f1) == "$PANEL_SHA256" ]] || die "panel hash mismatch"
[[ $(sha256sum "$SERVER_BIN" | cut -d' ' -f1) == "$SERVER_SHA256" ]] || die "server hash mismatch"
for artifact in "${artifacts[@]}"; do [[ -f "$artifact/manifest.json" ]] || die "missing $artifact"; done
if pgrep -x bw24-server >/dev/null || pgrep -af '/harbor run ' >/dev/null; then die "server/eval active"; fi
[[ -z $(docker ps -q) ]] || die "task containers active"

# Two private short/long prompts plus full 79-layer routing traces prove that the new masks load,
# answer, and never dispatch a pruned expert before public capability data is generated.
if [[ ! -f "$PRIVATE_GATE_ROOT/complete" ]]; then
  SERVER_BIN="$SERVER_BIN" ART_ROOT="$ART_ROOT" \
    REQUESTS=/data/calibration/hy3-confidence-v1/requests.jsonl OUT_ROOT="$PRIVATE_GATE_ROOT" \
    PY="$PY" CAPTURE_TOOL="$EVAL_ROOT/research/per-expert-quant/capture_calibration.py" \
    HEALTH_TOOL="$EVAL_ROOT/research/per-expert-quant/validate_server_health.py" \
    ROUTE_VALIDATOR="$ROOT/research/per-expert-quant/validate_pruned_route_trace.py" \
    ARMS_CSV=smart100_empirical,smart100_balanced,smart100_rescue \
    "$ROOT/research/per-expert-quant/run_100gb_private_artifact_gate.sh"
fi

RUN_ID=smart100-v1-$(git -C "$ROOT" rev-parse --short=7 HEAD)-$(date -u +%Y%m%dT%H%M%SZ)
[[ ! -e "$OUT_ROOT/run-configs/$RUN_ID.json" ]] || die "run id collision"
mkdir -p "$OUT_ROOT/run-configs"
printf '%s\n' "$RUN_ID" >"$OUT_ROOT/_active-run-id"
"$PY" - "$OUT_ROOT/run-configs/$RUN_ID.json" "$RUN_ID" "$ROOT" "$EVAL_ROOT" \
  "$PANEL_LOCK" "$SERVER_BIN" "${artifacts[@]}" <<'PY'
import hashlib,json,pathlib,subprocess,sys
out,run_id,root,eval_root,panel,server,*artifacts=sys.argv[1:]
sha=lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
arms=["prune100_joint_heal","smart100_empirical","smart100_balanced","smart100_rescue"]
d={"format":"bw24-smart100-directional-run-v1","run_id":run_id,
   "orchestration_commit":subprocess.check_output(["git","-C",root,"rev-parse","HEAD"],text=True).strip(),
   "eval_commit":subprocess.check_output(["git","-C",eval_root,"rev-parse","HEAD"],text=True).strip(),
   "panel":{"path":str(pathlib.Path(panel).resolve()),"sha256":sha(panel)},
   "server":{"path":str(pathlib.Path(server).resolve()),"sha256":sha(server)},
   "settings":{"mtp":False,"spec":False,"kv_reuse":False,"concurrency":1,"spill_depth":8},
   "artifacts":{a:{"path":str(pathlib.Path(p).resolve()),"manifest_sha256":sha(pathlib.Path(p)/"manifest.json")}
                for a,p in zip(arms,artifacts,strict=True)}}
pathlib.Path(out).write_text(json.dumps(d,indent=2,sort_keys=True)+"\n")
PY

pids=()
for lane in "${!arms[@]}"; do
  arm=${arms[$lane]}; artifact=${artifacts[$lane]}
  CUDA_VISIBLE_DEVICES=${gpus[$lane]} taskset -c "${cpus[$lane]}" numactl --membind="${numas[$lane]}" \
    env ARM="$arm" ARTIFACT="$artifact" RUN_ID="$RUN_ID" SERVER_BIN="$SERVER_BIN" \
      OUT_ROOT="$OUT_ROOT" CACHE_DIR="$CACHE_DIR" PANEL_LOCK="$PANEL_LOCK" \
      ADDR="127.0.0.1:${ports[$lane]}" BW24_SPILL_PREAD_DEPTH=8 \
      HOURISH_SCORER_LOCK=/tmp/bw24-smart100-scorer.lock \
      "$EVAL_ROOT/research/per-expert-quant/run_hourish_one_arm.sh" \
      >"$LOG_ROOT/$RUN_ID-$arm.launcher.log" 2>&1 &
  pids+=("$!")
done
failed=0; for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
((failed == 0)) || die "one or more directional arms failed"
pgrep -x bw24-server >/dev/null && die "server remained after directional completion"

arms_csv=$(IFS=,; echo "${arms[*]}")
"$PY" "$EVAL_ROOT/research/per-expert-quant/summarize_hourish_results.py" \
  --out-root "$OUT_ROOT" --run-id "$RUN_ID" --arms "$arms_csv" \
  --baseline prune100_joint_heal --panel-lock "$PANEL_LOCK" \
  --suite-lock "$EVAL_ROOT/research/per-expert-quant/suite.lock.json" \
  --server-sha256 "$SERVER_SHA256" --output "$OUT_ROOT/smart100-summary-$RUN_ID.json"
OLD_RUN_ID=$(cat /data/results/per-expert-quant/expanded-v2/_active-run-id)
FRONTIER="$OUT_ROOT/smart100-frontier-$RUN_ID.json"
"$PY" "$ROOT/research/per-expert-quant/summarize_cross_run_hourish.py" \
  --panel-lock "$PANEL_LOCK" --server-sha256 "$SERVER_SHA256" \
  --arm "plain_quant=/data/results/per-expert-quant/expanded-v2::$OLD_RUN_ID" \
  --arm "traffic_nvfp4_53_q2_139=/data/results/per-expert-quant/expanded-v2::$OLD_RUN_ID" \
  --arm "prune100_joint_heal=$OUT_ROOT::$RUN_ID" \
  --arm "smart100_empirical=$OUT_ROOT::$RUN_ID" \
  --arm "smart100_balanced=$OUT_ROOT::$RUN_ID" \
  --arm "smart100_rescue=$OUT_ROOT::$RUN_ID" \
  --baseline plain_quant --output "$FRONTIER"
PROMOTION="$OUT_ROOT/smart100-promotion-$RUN_ID.json"
"$PY" "$ROOT/research/per-expert-quant/select_smart100_promotions.py" \
  --frontier "$FRONTIER" --lock "$ROOT/research/per-expert-quant/smart100-promotion-gates.lock.json" \
  --output "$PROMOTION"
sha256sum "$OUT_ROOT/run-configs/$RUN_ID.json" "$OUT_ROOT/smart100-summary-$RUN_ID.json" \
  "$FRONTIER" "$PROMOTION" \
  "$PRIVATE_GATE_ROOT/evidence.sha256" >"$LOG_ROOT/evidence-$RUN_ID.sha256"
date -u +%FT%TZ | tee "$LOG_ROOT/complete"
