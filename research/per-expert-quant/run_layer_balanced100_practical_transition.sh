#!/usr/bin/env bash
set -euo pipefail

# Run only the newly qualified layer-balanced candidate, then compare it with immutable reference
# evidence from the matched plain/compact practical run. Candidate generation may overlap the
# references on isolated loopback ports; the strict comparator normalizes only that port field.

ROOT=${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
EXPECTED_COMMIT=${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the detached orchestration commit}
EVAL_ROOT=${EVAL_ROOT:-/data/src/bw24-eval-valid-c8689ec}
EVAL_COMMIT=${EVAL_COMMIT:-c8689ec937c899fbdbd399432ec175e7d48b53ae}
DIRECTIONAL_ROOT=${DIRECTIONAL_ROOT:-/data/results/per-expert-quant/layer-balanced100-directional-v1}
DIRECTIONAL_READY=${DIRECTIONAL_READY:-/data/logs/layer-balanced100-directional-v1/complete}
SOURCE_ROOT=${SOURCE_ROOT:-/data/results/per-expert-quant/practical-iq3-iq4-q4-pareto-v1}
SOURCE_READY=${SOURCE_READY:-/data/logs/practical-iq3-iq4-q4-pareto-v1/complete}
OUT_ROOT=${OUT_ROOT:-/data/results/per-expert-quant/practical-layer-balanced100-v1}
LOG_ROOT=${LOG_ROOT:-/data/logs/practical-layer-balanced100-v1}
ART_ROOT=${ART_ROOT:-/scratch/bw24-artifacts-layer-balanced100-3db293f}
ARM=${ARM:-layer_balanced100}
PRACTICAL_LOCK=${PRACTICAL_LOCK:-$EVAL_ROOT/research/per-expert-quant/practical-evals.lock.json}
GATE_LOCK=${GATE_LOCK:-$EVAL_ROOT/research/per-expert-quant/smart100-practical-gates.lock.json}
SELECTOR=${SELECTOR:-$ROOT/research/per-expert-quant/select_practical_promotions.py}
SUMMARIZER=${SUMMARIZER:-$ROOT/research/per-expert-quant/summarize_practical_results.py}
SERVER_BIN=${SERVER_BIN:-/data/build/bw24-portable-ada-fix-target/release/bw24-server}
SERVER_SHA256=${SERVER_SHA256:-13a7ac6a15a5de17f0eb736ecb393a50ab9bf03145e86a878e66c86b2086f195}
HARBOR_BIN=${HARBOR_BIN:-/data/bin/harbor-0.18.0-0a01ad6/harbor}
HARBOR_HOME=${HARBOR_HOME:-/data/cache/harbor-home}
SPILL_DEPTH=${SPILL_DEPTH:-8}
VRAM_FRAC=${VRAM_FRAC:-0.75}
WAIT_INTERVAL_S=${WAIT_INTERVAL_S:-30}
SERVER_HEALTH_TIMEOUT_S=${SERVER_HEALTH_TIMEOUT_S:-1800}

die() { echo "layer-balanced100 practical transition: $*" >&2; exit 2; }
port_listening() { ss -H -ltn "sport = :$1" 2>/dev/null | grep -q .; }
gpu_busy() {
  nvidia-smi -i "$1" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | grep -Eq '^[0-9]+$'
}

[[ -x "$SERVER_BIN" && -x "$HARBOR_BIN" ]] || die "missing server or Harbor"
[[ $(sha256sum "$SERVER_BIN" | cut -d' ' -f1) == "$SERVER_SHA256" ]] \
  || die "server hash mismatch"
for path in "$PRACTICAL_LOCK" "$GATE_LOCK" "$SELECTOR" "$SUMMARIZER"; do
  [[ -f "$path" ]] || die "missing practical prerequisite $path"
done
[[ "$SPILL_DEPTH" =~ ^[1-9][0-9]*$ ]] || die "invalid spill depth"
python3 - "$VRAM_FRAC" <<'PY'
import sys
value=float(sys.argv[1])
if not .5 <= value < .9:
    raise SystemExit("VRAM_FRAC must be in [0.5, 0.9)")
PY

mkdir -p "$OUT_ROOT/run-configs" "$LOG_ROOT"
exec 9>"$LOG_ROOT/transition.lock"
flock -n 9 || die "another layer-balanced100 practical transition owns the lock"
exec >>"$LOG_ROOT/transition.log" 2>&1
echo "$(date -u +%FT%TZ) layer-balanced100 practical transition started"
[[ $(git -C "$ROOT" rev-parse HEAD) == "$EXPECTED_COMMIT" ]] || die "source commit mismatch"
[[ -z $(git -C "$ROOT" status --porcelain) ]] || die "source checkout is dirty"
[[ $(git -C "$EVAL_ROOT" rev-parse HEAD) == "$EVAL_COMMIT" ]] || die "eval commit mismatch"
[[ -z $(git -C "$EVAL_ROOT" status --porcelain) ]] || die "eval checkout is dirty"

while [[ ! -f "$DIRECTIONAL_READY" || ! -s "$DIRECTIONAL_ROOT/_active-run-id" \
  || ! -s "$SOURCE_ROOT/_active-run-id" ]]; do
  sleep "$WAIT_INTERVAL_S"
done
DIRECTIONAL_RUN_ID=$(<"$DIRECTIONAL_ROOT/_active-run-id")
SOURCE_RUN_ID=$(<"$SOURCE_ROOT/_active-run-id")
PROMOTION="$DIRECTIONAL_ROOT/layer-balanced100-practical-input-$DIRECTIONAL_RUN_ID.json"
FRONTIER="$DIRECTIONAL_ROOT/layer-balanced100-frontier-$DIRECTIONAL_RUN_ID.json"
SOURCE_CONFIG="$SOURCE_ROOT/run-configs/$SOURCE_RUN_ID.json"
for path in "$PROMOTION" "$FRONTIER" "$SOURCE_CONFIG"; do
  [[ -f "$path" ]] || die "missing prerequisite $path"
done

selected=$(python3 - "$PROMOTION" "$ARM" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
if d.get("format") != "bw24-100gb-directional-promotion-v1":
    raise SystemExit("wrong directional promotion format")
arms=d.get("practical_arms")
if not isinstance(arms,list) or arms[:2] != ["plain_quant","traffic_nvfp4_53_q2_139"]:
    raise SystemExit("directional promotion lost frozen references")
selected=d.get("selected_100gb_arms")
if selected not in ([],[sys.argv[2]]):
    raise SystemExit("unexpected layer-balanced100 selection")
print(1 if selected else 0)
PY
)

python3 - "$SOURCE_CONFIG" "$SOURCE_ROOT" "$SOURCE_RUN_ID" "$PRACTICAL_LOCK" \
  "$SERVER_SHA256" <<'PY'
import hashlib,json,pathlib,sys
config_path,root,run_id,lock_path,server_sha=sys.argv[1:]
root=pathlib.Path(root); d=json.load(open(config_path))
sha=lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
if d.get("format") != "bw24-practical-transition-run-v1" or d.get("run_id") != run_id:
    raise SystemExit("wrong source practical config")
if d.get("arms",[])[:2] != ["plain_quant","traffic_nvfp4_53_q2_139"]:
    raise SystemExit("source practical references differ")
if d["practical_lock"]["sha256"] != sha(lock_path) or d["server"]["sha256"] != server_sha:
    raise SystemExit("source practical protocol hash differs")
PY

RUN_ID="practical-layer-balanced100-v1-$(date -u +%Y%m%dT%H%M%SZ)"
RUN_CONFIG="$OUT_ROOT/run-configs/$RUN_ID.json"
COMPARE_ROOT="$OUT_ROOT/comparisons/$RUN_ID"
mkdir -p "$COMPARE_ROOT"
artifact="$ART_ROOT/$ARM"
if [[ "$selected" == 1 ]]; then
  [[ -f "$artifact/manifest.json" ]] || die "missing selected candidate artifact"
fi
export ROOT RUN_ID ARM selected PROMOTION FRONTIER SOURCE_CONFIG SOURCE_ROOT SOURCE_RUN_ID \
  PRACTICAL_LOCK GATE_LOCK SERVER_BIN HARBOR_BIN artifact
python3 - "$RUN_CONFIG" <<'PY'
import hashlib,json,os,pathlib,subprocess,sys
sha=lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
payload={
  "format":"bw24-layer-balanced100-practical-reuse-run-v1",
  "run_id":os.environ["RUN_ID"],"candidate":os.environ["ARM"],
  "candidate_selected":os.environ["selected"] == "1",
  "protocol":"overlap candidate on isolated ports; reuse immutable matched references",
  "directional_promotion":{"path":os.environ["PROMOTION"],"sha256":sha(os.environ["PROMOTION"])},
  "directional_frontier":{"path":os.environ["FRONTIER"],"sha256":sha(os.environ["FRONTIER"])},
  "source_practical":{"run_id":os.environ["SOURCE_RUN_ID"],
      "config":{"path":os.environ["SOURCE_CONFIG"],"sha256":sha(os.environ["SOURCE_CONFIG"])}},
  "practical_lock":{"path":os.environ["PRACTICAL_LOCK"],"sha256":sha(os.environ["PRACTICAL_LOCK"])},
  "gate_lock":{"path":os.environ["GATE_LOCK"],"sha256":sha(os.environ["GATE_LOCK"])},
  "server":{"path":os.environ["SERVER_BIN"],"sha256":sha(os.environ["SERVER_BIN"])},
  "harbor":{"path":os.environ["HARBOR_BIN"],"sha256":sha(os.environ["HARBOR_BIN"])},
  "bw24_commit":subprocess.check_output(
      ["git","-C",os.environ["ROOT"],"rev-parse","HEAD"],text=True).strip(),
}
if payload["candidate_selected"]:
  manifest=pathlib.Path(os.environ["artifact"])/"manifest.json"
  payload["artifact"]={"path":str(manifest.parent.resolve()),"manifest_sha256":sha(manifest)}
pathlib.Path(sys.argv[1]).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY
sha256sum "$RUN_CONFIG" >"$RUN_CONFIG.sha256"
printf '%s\n' "$RUN_ID" >"$OUT_ROOT/_active-run-id"

declare -a WORKER_PIDS=()
cleanup_all() {
  status=$?
  trap - EXIT INT TERM
  for pid in "${WORKER_PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  exit "$status"
}
trap cleanup_all EXIT INT TERM

run_panel() (
  local panel=$1 gpu=$2 cpus=$3 port=$4 pilot=$5
  local arm_log="$LOG_ROOT/$RUN_ID-$ARM-$panel" server_log server_pid=""
  server_log="$arm_log/server.log"
  mkdir -p "$arm_log"
  echo "$port" >"$arm_log/port"
  port_listening "$port" && die "practical port $port is already in use"
  gpu_busy "$gpu" && die "practical GPU $gpu is already in use"
  stop_server() {
    local pid=${server_pid:-}
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      for _ in {1..100}; do kill -0 "$pid" 2>/dev/null || break; sleep .1; done
      kill -KILL "$pid" 2>/dev/null || true
    fi
    [[ -z "$pid" ]] || wait "$pid" 2>/dev/null || true
  }
  trap stop_server EXIT INT TERM
  env PATH="$PATH" CUDA_VISIBLE_DEVICES="$gpu" \
    BW24_COMPAT=openai BW24_SERVE_SPEC=0 BW24_KV_REUSE=0 BW24_CTX=8192 \
    BW24_FAST=1 BW24_MMVQ=1 BW24_MOE_CACHE=1 BW24_MOE_GROUPED=1 \
    BW24_MOE_PREWARM=1 BW24_MOE_PREFETCH=1 BW24_MOE_PAGE_PREFETCH=1 \
    BW24_MOE_PAGE_PREFETCH_WINDOW=8 BW24_MOE_MMAP_ADVICE=normal \
    BW24_MOE_RESIDENT=1 BW24_MOE_VRAM_FRAC="$VRAM_FRAC" BW24_SPILL_IO=worker \
    BW24_SPILL_PREAD_DEPTH="$SPILL_DEPTH" BW24_SPILL_STATS=1 \
    BW24_MODELS="$ARM=$artifact" BW24_ADDR="127.0.0.1:$port" \
    taskset -c "$cpus" "$SERVER_BIN" >"$server_log" 2>&1 &
  server_pid=$!
  echo "$server_pid" >"$arm_log/server.pid"
  deadline=$((SECONDS + SERVER_HEALTH_TIMEOUT_S))
  while ((SECONDS < deadline)); do
    if curl -fsS --max-time 5 "http://127.0.0.1:$port/health" >"$arm_log/health.json.tmp" 2>/dev/null \
      && python3 "$EVAL_ROOT/research/per-expert-quant/validate_server_health.py" \
        "$arm_log/health.json.tmp" "$ARM" --exact; then
      mv "$arm_log/health.json.tmp" "$arm_log/health.json"; break
    fi
    sleep 1
  done
  [[ -f "$arm_log/health.json" ]] || die "$panel server health timeout"

  pilot_root="$OUT_ROOT/_pilots"
  taskset -c "$cpus" env HOME="$HARBOR_HOME" PATH="$PATH" HF_HUB_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ARM="$ARM" PANEL="$panel" \
    ARTIFACT="$artifact" SERVER_BIN="$SERVER_BIN" SERVER_LOG="$server_log" \
    HARBOR_BIN="$HARBOR_BIN" LOCK="$PRACTICAL_LOCK" OUT_ROOT="$pilot_root" \
    RUN_ID="$RUN_ID" PILOT_TASK="$pilot" BASE_URL="http://127.0.0.1:$port/v1" \
    BW24_SPILL_IO=worker BW24_SPILL_PREAD_DEPTH="$SPILL_DEPTH" BW24_SPILL_STATS=1 \
    BW24_SERVE_SPEC=0 "$EVAL_ROOT/research/per-expert-quant/run_practical_evals.sh" \
    | tee "$arm_log/$panel-pilot.log"
  python3 "$EVAL_ROOT/research/per-expert-quant/validate_practical_pilot.py" \
    --run-dir "$pilot_root/$ARM/$panel/$RUN_ID" --lock "$PRACTICAL_LOCK" \
    --arm "$ARM" --panel "$panel" --expected-task "$pilot" \
    | tee "$pilot_root/$ARM/$panel/$RUN_ID/pilot-validation.json"

  taskset -c "$cpus" env HOME="$HARBOR_HOME" PATH="$PATH" HF_HUB_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ARM="$ARM" PANEL="$panel" \
    ARTIFACT="$artifact" SERVER_BIN="$SERVER_BIN" SERVER_LOG="$server_log" \
    HARBOR_BIN="$HARBOR_BIN" LOCK="$PRACTICAL_LOCK" OUT_ROOT="$OUT_ROOT" RUN_ID="$RUN_ID" \
    BASE_URL="http://127.0.0.1:$port/v1" BW24_SPILL_IO=worker \
    BW24_SPILL_PREAD_DEPTH="$SPILL_DEPTH" BW24_SPILL_STATS=1 BW24_SERVE_SPEC=0 \
    "$EVAL_ROOT/research/per-expert-quant/run_practical_evals.sh" \
    | tee "$arm_log/$panel.log"
  date -u +%FT%TZ >"$arm_log/complete"
)

if [[ "$selected" == 1 ]]; then
  mapfile -t pilots < <(python3 - "$SOURCE_CONFIG" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); p=d.get("pilot_tasks",{})
print(p["swe"]); print(p["terminal"])
PY
)
  [[ ${#pilots[@]} == 2 ]] || die "source pilot resolution failed"
  run_panel swe 4 48-59 8084 "${pilots[0]}" & WORKER_PIDS+=("$!")
  run_panel terminal 5 60-71 8085 "${pilots[1]}" & WORKER_PIDS+=("$!")
  failed=0; for pid in "${WORKER_PIDS[@]}"; do wait "$pid" || failed=1; done
  ((failed == 0)) || die "candidate practical worker failed"
fi

while [[ ! -f "$SOURCE_READY" ]]; do sleep "$WAIT_INTERVAL_S"; done
python3 - "$SOURCE_ROOT" "$SOURCE_RUN_ID" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]); run_id=sys.argv[2]
expected={"swe":"http://127.0.0.1:8082/v1","terminal":"http://127.0.0.1:8083/v1"}
for panel,base_url in expected.items():
    receipt=root/"traffic_nvfp4_53_q2_139"/panel/run_id/"run-metadata.json"
    row=json.loads(receipt.read_text())
    if not row.get("completed_successfully") or row.get("base_url") != base_url:
        raise SystemExit(f"source compact {panel} receipt is not reusable")
PY
SOURCE_INVENTORY="$OUT_ROOT/source-reference-$RUN_ID.sha256"
find \
  "$SOURCE_ROOT/plain_quant/swe/$SOURCE_RUN_ID" \
  "$SOURCE_ROOT/plain_quant/terminal/$SOURCE_RUN_ID" \
  "$SOURCE_ROOT/traffic_nvfp4_53_q2_139/swe/$SOURCE_RUN_ID" \
  "$SOURCE_ROOT/traffic_nvfp4_53_q2_139/terminal/$SOURCE_RUN_ID" \
  -type f -print0 | sort -z | xargs -0 sha256sum >"$SOURCE_INVENTORY"
python3 - "$OUT_ROOT/reuse-receipt-$RUN_ID.json" "$SOURCE_ROOT" "$SOURCE_RUN_ID" \
  "$SOURCE_CONFIG" "$PRACTICAL_LOCK" "$SOURCE_INVENTORY" <<'PY'
import hashlib,json,pathlib,sys
out,root,run_id,config,lock,inventory=map(pathlib.Path,sys.argv[1:])
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
refs={}
for arm in ("plain_quant","traffic_nvfp4_53_q2_139"):
  refs[arm]={}
  for panel in ("swe","terminal"):
    path=root/arm/panel/run_id/"run-metadata.json"
    refs[arm][panel]={"path":str(path.resolve()),"sha256":sha(path)}
payload={"format":"bw24-practical-reference-reuse-v1","source_run_id":str(run_id),
         "source_config":{"path":str(config.resolve()),"sha256":sha(config)},
         "practical_lock_sha256":sha(lock),
         "source_inventory":{"path":str(inventory.resolve()),"sha256":sha(inventory)},
         "references":refs}
out.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY

if [[ "$selected" == 1 ]]; then
  for baseline in plain_quant traffic_nvfp4_53_q2_139; do
    for panel in swe terminal; do
      python3 "$SUMMARIZER" \
        --baseline "$SOURCE_ROOT/$baseline/$panel/$SOURCE_RUN_ID" \
        --candidate "$OUT_ROOT/$ARM/$panel/$RUN_ID" --panel "$panel" \
        --lock "$PRACTICAL_LOCK" \
        --json-out "$COMPARE_ROOT/$baseline-vs-$ARM.$panel.json" \
        --markdown-out "$COMPARE_ROOT/$baseline-vs-$ARM.$panel.md"
    done
  done
fi

python3 "$SELECTOR" --promotion "$PROMOTION" --gate-lock "$GATE_LOCK" \
  --comparison-root "$COMPARE_ROOT" --output "$OUT_ROOT/practical-promotion-$RUN_ID.json"
find "$OUT_ROOT/_pilots" -path "*/$RUN_ID/*" -type f -print0 2>/dev/null \
  | sort -z | xargs -0 -r sha256sum >"$LOG_ROOT/$RUN_ID-pilot-evidence.sha256"
find "$COMPARE_ROOT" -type f -print0 | sort -z | xargs -0 -r sha256sum \
  >"$LOG_ROOT/$RUN_ID-comparison-evidence.sha256"
sha256sum "$RUN_CONFIG" "$PROMOTION" "$FRONTIER" "$GATE_LOCK" "$PRACTICAL_LOCK" \
  "$OUT_ROOT/reuse-receipt-$RUN_ID.json" "$LOG_ROOT/$RUN_ID-pilot-evidence.sha256" \
  "$SOURCE_INVENTORY" "$LOG_ROOT/$RUN_ID-comparison-evidence.sha256" \
  "$OUT_ROOT/practical-promotion-$RUN_ID.json" >"$LOG_ROOT/$RUN_ID-evidence.sha256"
sha256sum -c "$LOG_ROOT/$RUN_ID-evidence.sha256"
date -u +%FT%TZ | tee "$LOG_ROOT/complete"
trap - EXIT INT TERM
