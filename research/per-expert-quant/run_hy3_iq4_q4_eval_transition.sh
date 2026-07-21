#!/usr/bin/env bash
set -euo pipefail

# Evaluate the pre-frozen seven-format arm only after its private build/heal gate and the original
# smart100 directional run are complete.  The combined frontier is rebuilt from immutable run
# receipts; public results never flow back into allocation or healing.

ROOT=${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
PY=${PY:-python3}
EXPECTED_COMMIT=${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the detached orchestration commit}
ARM=${ARM:-smart100_iq3_iq4_q4_empirical}
BUILD_COMPLETE=${BUILD_COMPLETE:-/data/logs/iq3-iq4-q4-extension-99f3dc3/complete}
ART_ROOT=${ART_ROOT:-/scratch/bw24-artifacts-iq3-iq4-q4-99f3dc3}
OLD_DIRECTIONAL_ROOT=${OLD_DIRECTIONAL_ROOT:-/data/results/per-expert-quant/smart100-directional-v1}
OLD_DIRECTIONAL_COMPLETE=${OLD_DIRECTIONAL_COMPLETE:-/data/logs/smart100-directional-v1/complete}
EVAL_ROOT=${EVAL_ROOT:-/data/src/bw24-expanded-panel}
EVAL_COMMIT=${EVAL_COMMIT:-ae89c11975d7d51bbce0a56ae5963ad42dc68a6a}
PANEL_LOCK=${PANEL_LOCK:-$EVAL_ROOT/research/per-expert-quant/expanded-capability-panel.lock.json}
PANEL_SHA256=${PANEL_SHA256:-33ca7c2a86ed52ab3ee06ec408ceda890e50447e5cc4a204a755afcd3368c64b}
SERVER_BIN=${SERVER_BIN:-/data/build/bw24-portable-ada-fix-target/release/bw24-server}
SERVER_SHA256=${SERVER_SHA256:-13a7ac6a15a5de17f0eb736ecb393a50ab9bf03145e86a878e66c86b2086f195}
OUT_ROOT=${OUT_ROOT:-/data/results/per-expert-quant/iq3-iq4-q4-directional-v1}
LOG_ROOT=${LOG_ROOT:-/data/logs/iq3-iq4-q4-directional-v1}
CACHE_DIR=${CACHE_DIR:-/data/cache/per-expert-evals}
PRIVATE_GATE_ROOT=${PRIVATE_GATE_ROOT:-/data/logs/iq3-iq4-q4-private-artifact-gate-99f3dc3}
PROMOTION_LOCK=${PROMOTION_LOCK:-$ROOT/research/per-expert-quant/iq4-q4-promotion-gates.lock.json}

die() { echo "IQ4/Q4 eval transition: $*" >&2; exit 1; }
port_listening() { ss -H -ltn "sport = :$1" 2>/dev/null | grep -q .; }
mkdir -p "$OUT_ROOT/run-configs" "$LOG_ROOT"
exec 9>"$LOG_ROOT/transition.lock"
flock -n 9 || die "another IQ4/Q4 eval transition owns the lock"
echo "$(date -u +%FT%TZ) IQ4/Q4 directional transition started" | tee -a "$LOG_ROOT/transition.log"
[[ $(git -C "$ROOT" rev-parse HEAD) == "$EXPECTED_COMMIT" ]] || die "source commit mismatch"
[[ $(git -C "$EVAL_ROOT" rev-parse HEAD) == "$EVAL_COMMIT" ]] || die "eval commit mismatch"
[[ $(sha256sum "$PANEL_LOCK" | cut -d' ' -f1) == "$PANEL_SHA256" ]] || die "panel mismatch"
[[ $(sha256sum "$SERVER_BIN" | cut -d' ' -f1) == "$SERVER_SHA256" ]] || die "server mismatch"
[[ -f "$PROMOTION_LOCK" ]] || die "missing frozen promotion lock"
while [[ ! -f "$BUILD_COMPLETE" || ! -f "$OLD_DIRECTIONAL_COMPLETE" ]]; do sleep 30; done
artifact="$ART_ROOT/$ARM"; [[ -f "$artifact/manifest.json" ]] || die "missing candidate artifact"
if pgrep -x bw24-server >/dev/null || pgrep -af '/harbor run ' >/dev/null \
  || [[ -n $(docker ps -q) ]]; then
  die "GPU evaluation work remains active after prerequisite completion"
fi

if [[ ! -f "$PRIVATE_GATE_ROOT/complete" ]]; then
  SERVER_BIN="$SERVER_BIN" ART_ROOT="$ART_ROOT" \
    REQUESTS=/data/calibration/hy3-confidence-v1/requests.jsonl OUT_ROOT="$PRIVATE_GATE_ROOT" \
    PY="$PY" CAPTURE_TOOL="$EVAL_ROOT/research/per-expert-quant/capture_calibration.py" \
    HEALTH_TOOL="$EVAL_ROOT/research/per-expert-quant/validate_server_health.py" \
    ROUTE_VALIDATOR="$ROOT/research/per-expert-quant/validate_pruned_route_trace.py" \
    ARMS_CSV="$ARM" "$ROOT/research/per-expert-quant/run_100gb_private_artifact_gate.sh"
fi

RUN_ID=iq3-iq4-q4-v1-$(git -C "$ROOT" rev-parse --short=7 HEAD)-$(date -u +%Y%m%dT%H%M%SZ)
printf '%s\n' "$RUN_ID" >"$OUT_ROOT/_active-run-id"
"$PY" - "$OUT_ROOT/run-configs/$RUN_ID.json" "$RUN_ID" "$ROOT" "$EVAL_ROOT" \
  "$PANEL_LOCK" "$SERVER_BIN" "$artifact" "$ARM" <<'PY'
import hashlib,json,pathlib,subprocess,sys
out,run_id,root,eval_root,panel,server,artifact,arm=sys.argv[1:]
sha=lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
d={"format":"bw24-iq3-iq4-q4-directional-run-v1","run_id":run_id,
   "orchestration_commit":subprocess.check_output(["git","-C",root,"rev-parse","HEAD"],text=True).strip(),
   "eval_commit":subprocess.check_output(["git","-C",eval_root,"rev-parse","HEAD"],text=True).strip(),
   "panel":{"path":str(pathlib.Path(panel).resolve()),"sha256":sha(panel)},
   "server":{"path":str(pathlib.Path(server).resolve()),"sha256":sha(server)},
   "settings":{"mtp":False,"spec":False,"kv_reuse":False,"concurrency":1,"spill_depth":8},
   "artifact":{"arm":arm,"path":str(pathlib.Path(artifact).resolve()),
               "manifest_sha256":sha(pathlib.Path(artifact)/"manifest.json")}}
pathlib.Path(out).write_text(json.dumps(d,indent=2,sort_keys=True)+"\n")
PY

CUDA_VISIBLE_DEVICES=0 taskset -c 0-11 numactl --membind=0 \
  env ARM="$ARM" ARTIFACT="$artifact" RUN_ID="$RUN_ID" SERVER_BIN="$SERVER_BIN" \
    OUT_ROOT="$OUT_ROOT" CACHE_DIR="$CACHE_DIR" PANEL_LOCK="$PANEL_LOCK" \
    ADDR=127.0.0.1:8080 BW24_SPILL_PREAD_DEPTH=8 \
    HOURISH_SCORER_LOCK=/tmp/bw24-iq3-iq4-q4-scorer.lock \
    "$EVAL_ROOT/research/per-expert-quant/run_hourish_one_arm.sh" \
    >"$LOG_ROOT/$RUN_ID-$ARM.launcher.log" 2>&1
port_listening 8080 && die "server remained on candidate eval port 8080"
"$PY" "$EVAL_ROOT/research/per-expert-quant/summarize_hourish_results.py" \
  --out-root "$OUT_ROOT" --run-id "$RUN_ID" --arms "$ARM" --baseline "$ARM" \
  --panel-lock "$PANEL_LOCK" --suite-lock "$EVAL_ROOT/research/per-expert-quant/suite.lock.json" \
  --server-sha256 "$SERVER_SHA256" --output "$OUT_ROOT/iq3-iq4-q4-summary-$RUN_ID.json"

old_run=$(cat "$OLD_DIRECTIONAL_ROOT/_active-run-id")
old_frontier="$OLD_DIRECTIONAL_ROOT/smart100-frontier-$old_run.json"
[[ -f "$old_frontier" ]] || die "missing original smart100 frontier"
mapfile -t old_specs < <("$PY" - "$old_frontier" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
assert d["format"] == "bw24-cross-run-expanded-capability-frontier-v1"
for item in d["source_runs"]:
    print(f'{item["arm"]}={item["out_root"]}::{item["run_id"]}')
PY
)
frontier="$OUT_ROOT/iq3-iq4-q4-frontier-$RUN_ID.json"
frontier_args=(--panel-lock "$PANEL_LOCK" --server-sha256 "$SERVER_SHA256")
for spec in "${old_specs[@]}"; do frontier_args+=(--arm "$spec"); done
frontier_args+=(--arm "$ARM=$OUT_ROOT::$RUN_ID" --baseline plain_quant --output "$frontier")
"$PY" "$ROOT/research/per-expert-quant/summarize_cross_run_hourish.py" "${frontier_args[@]}"
promotion="$OUT_ROOT/iq3-iq4-q4-promotion-$RUN_ID.json"
"$PY" "$ROOT/research/per-expert-quant/select_smart100_promotions.py" \
  --frontier "$frontier" --lock "$PROMOTION_LOCK" --output "$promotion"
practical="$OUT_ROOT/iq3-iq4-q4-practical-input-$RUN_ID.json"
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
sha256sum "$OUT_ROOT/run-configs/$RUN_ID.json" "$OUT_ROOT/iq3-iq4-q4-summary-$RUN_ID.json" \
  "$frontier" "$promotion" "$practical" "$PRIVATE_GATE_ROOT/evidence.sha256" \
  >"$LOG_ROOT/evidence-$RUN_ID.sha256"
date -u +%FT%TZ | tee "$LOG_ROOT/complete"
