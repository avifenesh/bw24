#!/usr/bin/env bash
set -euo pipefail

# Run only bridge arms selected by the frozen directional gate, then compare them and the already
# running Layer100 arm against immutable plain/Traffic137 practical references.

ROOT=${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
EXPECTED_COMMIT=${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the detached orchestration commit}
EVAL_ROOT=${EVAL_ROOT:-/data/src/bw24-eval-valid-c8689ec}
EVAL_COMMIT=${EVAL_COMMIT:-c8689ec937c899fbdbd399432ec175e7d48b53ae}
DIRECTIONAL_ROOT=${DIRECTIONAL_ROOT:-/data/results/per-expert-quant/layer-balanced-bridge-directional-v1}
DIRECTIONAL_READY=${DIRECTIONAL_READY:-/data/logs/layer-balanced-bridge-directional-v1/complete}
SOURCE_ROOT=${SOURCE_ROOT:-/data/results/per-expert-quant/practical-iq3-iq4-q4-pareto-v1}
SOURCE_READY=${SOURCE_READY:-/data/logs/practical-iq3-iq4-q4-pareto-v1/complete}
LAYER100_ROOT=${LAYER100_ROOT:-/data/results/per-expert-quant/practical-layer-balanced100-v1}
LAYER100_READY=${LAYER100_READY:-/data/logs/practical-layer-balanced100-v1/complete}
OUT_ROOT=${OUT_ROOT:-/data/results/per-expert-quant/practical-layer-balanced-bridge-v1}
LOG_ROOT=${LOG_ROOT:-/data/logs/practical-layer-balanced-bridge-v1}
ART_ROOT=${ART_ROOT:-/scratch/bw24-artifacts-layer-balanced-bridge}
PRACTICAL_LOCK=${PRACTICAL_LOCK:-$EVAL_ROOT/research/per-expert-quant/practical-evals.lock.json}
GATE_LOCK=${GATE_LOCK:-$ROOT/research/per-expert-quant/layer-balanced-bridge-practical-gates.lock.json}
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

die() { echo "layer-balanced bridge practical transition: $*" >&2; exit 2; }
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
python3 - "$VRAM_FRAC" <<'PY'
import sys
value=float(sys.argv[1])
if not .5 <= value < .9:
    raise SystemExit("VRAM_FRAC must be in [0.5, 0.9)")
PY

mkdir -p "$OUT_ROOT/run-configs" "$LOG_ROOT"
exec 9>"$LOG_ROOT/transition.lock"
flock -n 9 || die "another bridge practical transition owns the lock"
exec >>"$LOG_ROOT/transition.log" 2>&1
echo "$(date -u +%FT%TZ) layer-balanced bridge practical transition started"
[[ $(git -C "$ROOT" rev-parse HEAD) == "$EXPECTED_COMMIT" ]] || die "source commit mismatch"
[[ -z $(git -C "$ROOT" status --porcelain) ]] || die "source checkout is dirty"
[[ $(git -C "$EVAL_ROOT" rev-parse HEAD) == "$EVAL_COMMIT" ]] || die "eval commit mismatch"
[[ -z $(git -C "$EVAL_ROOT" status --porcelain) ]] || die "eval checkout is dirty"

while [[ ! -f "$DIRECTIONAL_READY" || ! -f "$SOURCE_READY" || ! -f "$LAYER100_READY" \
  || ! -s "$DIRECTIONAL_ROOT/_active-run-id" || ! -s "$SOURCE_ROOT/_active-run-id" \
  || ! -s "$LAYER100_ROOT/_active-run-id" ]]; do
  sleep "$WAIT_INTERVAL_S"
done
DIRECTIONAL_RUN_ID=$(<"$DIRECTIONAL_ROOT/_active-run-id")
SOURCE_RUN_ID=$(<"$SOURCE_ROOT/_active-run-id")
LAYER100_RUN_ID=$(<"$LAYER100_ROOT/_active-run-id")
PROMOTION="$DIRECTIONAL_ROOT/layer-balanced-bridge-promotion-$DIRECTIONAL_RUN_ID.json"
FRONTIER="$DIRECTIONAL_ROOT/layer-balanced-bridge-frontier-$DIRECTIONAL_RUN_ID.json"
SOURCE_CONFIG="$SOURCE_ROOT/run-configs/$SOURCE_RUN_ID.json"
LAYER100_CONFIG="$LAYER100_ROOT/run-configs/$LAYER100_RUN_ID.json"
for path in "$PROMOTION" "$FRONTIER" "$SOURCE_CONFIG" "$LAYER100_CONFIG"; do
  [[ -f "$path" ]] || die "missing practical input $path"
done

mapfile -t bridge_arms < <(python3 - "$PROMOTION" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
if d.get("format") != "bw24-layer-balanced-bridge-directional-promotion-v1":
    raise SystemExit("wrong bridge promotion format")
selected=d.get("selected_practical_candidates")
if not isinstance(selected,list) or len(selected)>2 or any(
    arm not in ("layer_balanced120","layer_balanced137") for arm in selected
):
    raise SystemExit("invalid selected bridge candidates")
print("\n".join(selected))
PY
)

RUN_ID=practical-layer-balanced-bridge-v1-$(date -u +%Y%m%dT%H%M%SZ)
RUN_CONFIG="$OUT_ROOT/run-configs/$RUN_ID.json"
COMBINED_PROMOTION="$OUT_ROOT/combined-directional-promotion-$RUN_ID.json"
COMPARE_ROOT="$OUT_ROOT/comparisons/$RUN_ID"
mkdir -p "$COMPARE_ROOT"
python3 - "$PROMOTION" "$COMBINED_PROMOTION" "${bridge_arms[@]}" <<'PY'
import json,pathlib,sys
source=pathlib.Path(sys.argv[1]); output=pathlib.Path(sys.argv[2]); selected=sys.argv[3:]
d=json.loads(source.read_text())
if selected != d["selected_practical_candidates"]:
    raise SystemExit("bridge selection changed while constructing practical input")
payload={"format":"bw24-100gb-directional-promotion-v1",
         "source_format":d["format"],
         "practical_arms":["plain_quant","traffic_nvfp4_53_q2_139",
                           "layer_balanced100",*selected],
         "selected_100gb_arms":["layer_balanced100",*selected]}
output.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY

for arm in "${bridge_arms[@]}"; do
  [[ -f "$ART_ROOT/$arm/manifest.json" ]] || die "missing selected artifact $arm"
done
export ROOT RUN_ID PROMOTION FRONTIER COMBINED_PROMOTION SOURCE_CONFIG LAYER100_CONFIG \
  SOURCE_RUN_ID LAYER100_RUN_ID PRACTICAL_LOCK GATE_LOCK SERVER_BIN HARBOR_BIN ART_ROOT
python3 - "$RUN_CONFIG" "${bridge_arms[@]}" <<'PY'
import hashlib,json,os,pathlib,subprocess,sys
sha=lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
selected=sys.argv[2:]
payload={"format":"bw24-layer-balanced-bridge-practical-run-v1",
  "run_id":os.environ["RUN_ID"],"selected_bridge_arms":selected,
  "protocol":"reuse immutable plain, Traffic137, and Layer100 evidence; generate selected bridge arms",
  "directional_promotion":{"path":os.environ["PROMOTION"],"sha256":sha(os.environ["PROMOTION"])},
  "directional_frontier":{"path":os.environ["FRONTIER"],"sha256":sha(os.environ["FRONTIER"])},
  "combined_promotion":{"path":os.environ["COMBINED_PROMOTION"],
                        "sha256":sha(os.environ["COMBINED_PROMOTION"])},
  "source_practical":{"run_id":os.environ["SOURCE_RUN_ID"],
      "config":{"path":os.environ["SOURCE_CONFIG"],"sha256":sha(os.environ["SOURCE_CONFIG"])}},
  "layer100_practical":{"run_id":os.environ["LAYER100_RUN_ID"],
      "config":{"path":os.environ["LAYER100_CONFIG"],"sha256":sha(os.environ["LAYER100_CONFIG"])}},
  "practical_lock":{"path":os.environ["PRACTICAL_LOCK"],"sha256":sha(os.environ["PRACTICAL_LOCK"])},
  "gate_lock":{"path":os.environ["GATE_LOCK"],"sha256":sha(os.environ["GATE_LOCK"])},
  "server":{"path":os.environ["SERVER_BIN"],"sha256":sha(os.environ["SERVER_BIN"])},
  "harbor":{"path":os.environ["HARBOR_BIN"],"sha256":sha(os.environ["HARBOR_BIN"])},
  "bw24_commit":subprocess.check_output(
      ["git","-C",os.environ["ROOT"],"rev-parse","HEAD"],text=True).strip(),
  "artifacts":{}}
for arm in selected:
    manifest=pathlib.Path(os.environ["ART_ROOT"])/arm/"manifest.json"
    payload["artifacts"][arm]={"path":str(manifest.parent.resolve()),"manifest_sha256":sha(manifest)}
pathlib.Path(sys.argv[1]).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY
sha256sum "$RUN_CONFIG" >"$RUN_CONFIG.sha256"
printf '%s\n' "$RUN_ID" >"$OUT_ROOT/_active-run-id"

mapfile -t pilots < <(python3 - "$SOURCE_CONFIG" <<'PY'
import json,sys
p=json.load(open(sys.argv[1])).get("pilot_tasks",{})
print(p["swe"]); print(p["terminal"])
PY
)
[[ ${#pilots[@]} == 2 ]] || die "source pilot resolution failed"

declare -a WORKER_PIDS=()
cleanup_all() {
  status=$?; trap - EXIT INT TERM
  for pid in "${WORKER_PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  exit "$status"
}
trap cleanup_all EXIT INT TERM

run_panel() (
  local arm=$1 panel=$2 gpu=$3 cpus=$4 port=$5 pilot=$6
  local artifact="$ART_ROOT/$arm" arm_log="$LOG_ROOT/$RUN_ID-$arm-$panel"
  local server_log="$arm_log/server.log" server_pid=""
  mkdir -p "$arm_log"; echo "$port" >"$arm_log/port"
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
    BW24_MODELS="$arm=$artifact" BW24_ADDR="127.0.0.1:$port" \
    taskset -c "$cpus" "$SERVER_BIN" >"$server_log" 2>&1 &
  server_pid=$!
  deadline=$((SECONDS + SERVER_HEALTH_TIMEOUT_S))
  while ((SECONDS < deadline)); do
    if curl -fsS --max-time 5 "http://127.0.0.1:$port/health" >"$arm_log/health.json.tmp" 2>/dev/null \
      && python3 "$EVAL_ROOT/research/per-expert-quant/validate_server_health.py" \
        "$arm_log/health.json.tmp" "$arm" --exact; then
      mv "$arm_log/health.json.tmp" "$arm_log/health.json"; break
    fi
    sleep 1
  done
  [[ -f "$arm_log/health.json" ]] || die "$arm/$panel server health timeout"
  pilot_root="$OUT_ROOT/_pilots"
  taskset -c "$cpus" env HOME="$HARBOR_HOME" PATH="$PATH" HF_HUB_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ARM="$arm" PANEL="$panel" \
    ARTIFACT="$artifact" SERVER_BIN="$SERVER_BIN" SERVER_LOG="$server_log" \
    HARBOR_BIN="$HARBOR_BIN" LOCK="$PRACTICAL_LOCK" OUT_ROOT="$pilot_root" \
    RUN_ID="$RUN_ID" PILOT_TASK="$pilot" BASE_URL="http://127.0.0.1:$port/v1" \
    BW24_SPILL_IO=worker BW24_SPILL_PREAD_DEPTH="$SPILL_DEPTH" BW24_SPILL_STATS=1 \
    BW24_SERVE_SPEC=0 "$EVAL_ROOT/research/per-expert-quant/run_practical_evals.sh" \
    | tee "$arm_log/$panel-pilot.log"
  python3 "$EVAL_ROOT/research/per-expert-quant/validate_practical_pilot.py" \
    --run-dir "$pilot_root/$arm/$panel/$RUN_ID" --lock "$PRACTICAL_LOCK" \
    --arm "$arm" --panel "$panel" --expected-task "$pilot" \
    | tee "$pilot_root/$arm/$panel/$RUN_ID/pilot-validation.json"
  taskset -c "$cpus" env HOME="$HARBOR_HOME" PATH="$PATH" HF_HUB_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ARM="$arm" PANEL="$panel" \
    ARTIFACT="$artifact" SERVER_BIN="$SERVER_BIN" SERVER_LOG="$server_log" \
    HARBOR_BIN="$HARBOR_BIN" LOCK="$PRACTICAL_LOCK" OUT_ROOT="$OUT_ROOT" RUN_ID="$RUN_ID" \
    BASE_URL="http://127.0.0.1:$port/v1" BW24_SPILL_IO=worker \
    BW24_SPILL_PREAD_DEPTH="$SPILL_DEPTH" BW24_SPILL_STATS=1 BW24_SERVE_SPEC=0 \
    "$EVAL_ROOT/research/per-expert-quant/run_practical_evals.sh" \
    | tee "$arm_log/$panel.log"
  date -u +%FT%TZ >"$arm_log/complete"
)

for index in "${!bridge_arms[@]}"; do
  arm=${bridge_arms[$index]}; base_gpu=$((index * 2)); base_cpu=$((index * 24))
  run_panel "$arm" swe "$base_gpu" "$base_cpu-$((base_cpu + 11))" \
    "$((8080 + base_gpu))" "${pilots[0]}" & WORKER_PIDS+=("$!")
  run_panel "$arm" terminal "$((base_gpu + 1))" "$((base_cpu + 12))-$((base_cpu + 23))" \
    "$((8081 + base_gpu))" "${pilots[1]}" & WORKER_PIDS+=("$!")
done
failed=0
for pid in "${WORKER_PIDS[@]}"; do wait "$pid" || failed=1; done
((failed == 0)) || die "one or more bridge practical workers failed"

candidate_source() {
  case "$1" in
    layer_balanced100) printf '%s\n' "$LAYER100_ROOT/$1/$2/$LAYER100_RUN_ID" ;;
    layer_balanced120|layer_balanced137) printf '%s\n' "$OUT_ROOT/$1/$2/$RUN_ID" ;;
    *) die "unexpected candidate $1" ;;
  esac
}
all_candidates=(layer_balanced100 "${bridge_arms[@]}")
for candidate in "${all_candidates[@]}"; do
  for baseline in plain_quant traffic_nvfp4_53_q2_139; do
    for panel in swe terminal; do
      candidate_dir=$(candidate_source "$candidate" "$panel")
      python3 "$SUMMARIZER" \
        --baseline "$SOURCE_ROOT/$baseline/$panel/$SOURCE_RUN_ID" \
        --candidate "$candidate_dir" --panel "$panel" --lock "$PRACTICAL_LOCK" \
        --json-out "$COMPARE_ROOT/$baseline-vs-$candidate.$panel.json" \
        --markdown-out "$COMPARE_ROOT/$baseline-vs-$candidate.$panel.md"
    done
  done
done

python3 "$SELECTOR" --promotion "$COMBINED_PROMOTION" --gate-lock "$GATE_LOCK" \
  --comparison-root "$COMPARE_ROOT" --output "$OUT_ROOT/practical-promotion-$RUN_ID.json"
find "$OUT_ROOT/_pilots" -path "*/$RUN_ID/*" -type f -print0 2>/dev/null \
  | sort -z | xargs -0 -r sha256sum >"$LOG_ROOT/$RUN_ID-pilot-evidence.sha256"
find "$COMPARE_ROOT" -type f -print0 | sort -z | xargs -0 sha256sum \
  >"$LOG_ROOT/$RUN_ID-comparison-evidence.sha256"
sha256sum "$RUN_CONFIG" "$PROMOTION" "$FRONTIER" "$COMBINED_PROMOTION" \
  "$GATE_LOCK" "$PRACTICAL_LOCK" "$SOURCE_CONFIG" "$LAYER100_CONFIG" \
  "$LOG_ROOT/$RUN_ID-pilot-evidence.sha256" "$LOG_ROOT/$RUN_ID-comparison-evidence.sha256" \
  "$OUT_ROOT/practical-promotion-$RUN_ID.json" >"$LOG_ROOT/$RUN_ID-evidence.sha256"
sha256sum -c "$LOG_ROOT/$RUN_ID-evidence.sha256"
date -u +%FT%TZ | tee "$LOG_ROOT/complete"
trap - EXIT INT TERM
