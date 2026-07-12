#!/usr/bin/env bash
set -euo pipefail

ROOT=${BW24_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
HERE="$ROOT/research/per-expert-quant"
PROMOTION=${PROMOTION:-/data/results/per-expert-quant/100gb-heal-v1/directional-promotion.json}
PROMOTION_READY=${PROMOTION_READY:-/data/logs/100gb-promotion-5f02c37/complete}
GATE_LOCK=${GATE_LOCK:-$HERE/100gb-promotion-gates.lock.json}
PRACTICAL_LOCK=${PRACTICAL_LOCK:-$HERE/practical-evals.lock.json}
PRACTICAL_SELECTOR=${PRACTICAL_SELECTOR:-$HERE/select_practical_promotions.py}
CONFIRMATION_LOCK=${CONFIRMATION_LOCK:-}
FRONTIER=${FRONTIER:-}
OUT_ROOT=${OUT_ROOT:-/data/results/per-expert-quant/practical-v1}
LOG_ROOT=${LOG_ROOT:-/data/logs/practical-v1}
SERVER_BIN=${SERVER_BIN:-/data/build/bw24-portable-ada-fix-target/release/bw24-server}
HARBOR_BIN=${HARBOR_BIN:-/data/bin/harbor-0.18.0-0a01ad6/harbor}
HARBOR_HOME=${HARBOR_HOME:-/data/cache/harbor-home}
SPILL_DEPTH=${SPILL_DEPTH:-8}
WAIT_INTERVAL_S=${WAIT_INTERVAL_S:-30}
SERVER_HEALTH_TIMEOUT_S=${SERVER_HEALTH_TIMEOUT_S:-1800}

die() { echo "error: $*" >&2; exit 2; }

[[ -x "$SERVER_BIN" ]] || die "missing practical server"
[[ -x "$HARBOR_BIN" ]] || die "missing pinned Harbor"
[[ -f "$GATE_LOCK" && -f "$PRACTICAL_LOCK" ]] || die "missing frozen lock"
[[ -f "$PRACTICAL_SELECTOR" ]] || die "missing practical promotion selector"
if [[ -n "$CONFIRMATION_LOCK" ]]; then
  [[ -f "$CONFIRMATION_LOCK" && -n "$FRONTIER" ]] \
    || die "surprise confirmation requires its lock and strict-frontier path"
fi
[[ "$SPILL_DEPTH" =~ ^[1-9][0-9]*$ ]] || die "invalid spill depth"

mkdir -p "$LOG_ROOT" "$OUT_ROOT/run-configs"
exec 9>"$LOG_ROOT/transition.lock"
flock -n 9 || exit 0
exec >>"$LOG_ROOT/transition.log" 2>&1
echo "$(date -u +%FT%TZ) promoted practical transition started"

while [[ ! -f "$PROMOTION_READY" || ! -f "$PROMOTION" ]]; do sleep "$WAIT_INTERVAL_S"; done
if [[ -n "$CONFIRMATION_LOCK" ]]; then
  [[ -f "$FRONTIER" ]] || die "strict surprise-confirmation frontier is missing"
fi

mapfile -t ARMS < <(python3 - "$PROMOTION" "$CONFIRMATION_LOCK" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
if d.get("format") != "bw24-100gb-directional-promotion-v1":
    raise SystemExit("wrong directional promotion format")
arms = d.get("practical_arms")
if not isinstance(arms, list) or not 2 <= len(arms) <= 3 or len(set(arms)) != len(arms):
    raise SystemExit("expected two or three unique practical arms")
if sys.argv[2]:
    confirmation = json.load(open(sys.argv[2]))
    if confirmation.get("format") != "bw24-surprise-practical-confirmation-v1":
        raise SystemExit("wrong surprise-confirmation lock format")
    arms = list(dict.fromkeys(arms + confirmation.get("candidate_arms", [])))
if len(arms) > 4:
    raise SystemExit("at most four practical arms are supported")
for arm in arms:
    if not isinstance(arm, str) or not arm.replace("_", "").replace("-", "").isalnum():
        raise SystemExit(f"invalid arm: {arm!r}")
    print(arm)
PY
)

artifact_for() {
  case "$1" in
    plain_quant) printf '%s\n' /scratch/bw24-artifacts/plain-quant ;;
    traffic_nvfp4_53_q2_139) printf '%s\n' /scratch/bw24-artifacts/traffic-nvfp4-53-q2-139 ;;
    prune100_unhealed|prune100_router_repair|prune100_joint_heal)
      printf '/scratch/bw24-artifacts-100gb-5f02c37/%s\n' "$1" ;;
    smart100_empirical|smart100_balanced|smart100_rescue)
      printf '/scratch/bw24-artifacts-smart100-2605fde/%s\n' "$1" ;;
    *) die "no frozen artifact mapping for $1" ;;
  esac
}

for arm in "${ARMS[@]}"; do
  artifact=$(artifact_for "$arm")
  [[ -f "$artifact/manifest.json" ]] || die "missing manifest for $arm"
done

RUN_ID="practical-v1-$(date -u +%Y%m%dT%H%M%SZ)"
RUN_CONFIG="$OUT_ROOT/run-configs/$RUN_ID.json"
[[ ! -e "$RUN_CONFIG" ]] || die "run config already exists"
export RUN_ID PROMOTION GATE_LOCK PRACTICAL_LOCK SERVER_BIN HARBOR_BIN \
  CONFIRMATION_LOCK FRONTIER
python3 - "$RUN_CONFIG" "${ARMS[@]}" <<'PY'
import hashlib, json, os, pathlib, sys

def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()

arms = sys.argv[2:]
payload = {
    "format": "bw24-practical-transition-run-v1",
    "run_id": os.environ["RUN_ID"],
    "arms": arms,
    "panels": ["swe", "terminal"],
    "protocol": "parallel unique localhost ports; one GPU and concurrency one per panel",
    "directional_promotion": {"path": os.environ["PROMOTION"], "sha256": sha(os.environ["PROMOTION"])},
    "gate_lock": {"path": os.environ["GATE_LOCK"], "sha256": sha(os.environ["GATE_LOCK"])},
    "practical_lock": {"path": os.environ["PRACTICAL_LOCK"], "sha256": sha(os.environ["PRACTICAL_LOCK"])},
    "server": {"path": os.environ["SERVER_BIN"], "sha256": sha(os.environ["SERVER_BIN"])},
    "harbor": {"path": os.environ["HARBOR_BIN"], "sha256": sha(os.environ["HARBOR_BIN"])},
    "artifacts": {},
}
if os.environ.get("CONFIRMATION_LOCK"):
    payload["surprise_confirmation"] = {
        "path": os.environ["CONFIRMATION_LOCK"],
        "sha256": sha(os.environ["CONFIRMATION_LOCK"]),
        "frontier": {
            "path": os.environ["FRONTIER"],
            "sha256": sha(os.environ["FRONTIER"]),
        },
    }
for arm in arms:
    if arm.startswith("prune100_"):
        root = pathlib.Path("/scratch/bw24-artifacts-100gb-5f02c37") / arm
    elif arm.startswith("smart100_"):
        root = pathlib.Path("/scratch/bw24-artifacts-smart100-2605fde") / arm
    else:
        root = pathlib.Path("/scratch/bw24-artifacts") / (arm.replace("_", "-") if arm in ("plain_quant",) else arm)
    if arm == "plain_quant": root = pathlib.Path("/scratch/bw24-artifacts/plain-quant")
    if arm == "traffic_nvfp4_53_q2_139": root = pathlib.Path("/scratch/bw24-artifacts/traffic-nvfp4-53-q2-139")
    manifest = root / "manifest.json"
    payload["artifacts"][arm] = {"path": str(root), "manifest_sha256": sha(manifest)}
pathlib.Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
sha256sum "$RUN_CONFIG" > "$RUN_CONFIG.sha256"
printf '%s\n' "$RUN_ID" > "$OUT_ROOT/_active-run-id"

USER_PATH=$PATH
declare -a WORKER_PIDS=()

cleanup_all() {
  status=$?
  trap - EXIT INT TERM
  for pid in "${WORKER_PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  exit "$status"
}
trap cleanup_all EXIT INT TERM

run_panel() (
  local arm=$1 panel=$2 gpu=$3 cpus=$4
  local artifact arm_log server_log server_pid port
  server_pid=""
  artifact=$(artifact_for "$arm")
  port=$((8080 + gpu))
  arm_log="$LOG_ROOT/$RUN_ID-$arm-$panel"
  server_log="$arm_log/server.log"
  mkdir -p "$arm_log"
  echo "$port" > "$arm_log/port"
  if curl -fsS --connect-timeout 1 --max-time 2 "http://127.0.0.1:$port/health" \
    >/dev/null 2>&1; then
    die "a server is already answering on practical port $port"
  fi

  stop_server() {
    if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
      kill "$server_pid" 2>/dev/null || true
      for _ in {1..100}; do
        kill -0 "$server_pid" 2>/dev/null || break
        sleep 0.1
      done
      kill -KILL "$server_pid" 2>/dev/null || true
    fi
    [[ -z "$server_pid" ]] || wait "$server_pid" 2>/dev/null || true
  }
  trap stop_server EXIT INT TERM

  env \
    PATH="$USER_PATH" CUDA_VISIBLE_DEVICES="$gpu" \
    BW24_COMPAT=openai BW24_SERVE_SPEC=0 BW24_KV_REUSE=0 BW24_CTX=8192 \
    BW24_FAST=1 BW24_MMVQ=1 BW24_MOE_CACHE=1 BW24_MOE_GROUPED=1 \
    BW24_MOE_PREWARM=1 BW24_MOE_PREFETCH=1 BW24_MOE_PAGE_PREFETCH=1 \
    BW24_MOE_PAGE_PREFETCH_WINDOW=8 BW24_MOE_MMAP_ADVICE=normal \
    BW24_MOE_RESIDENT=1 BW24_MOE_VRAM_FRAC=0.85 BW24_SPILL_IO=worker \
    BW24_SPILL_PREAD_DEPTH="$SPILL_DEPTH" BW24_SPILL_STATS=1 \
    BW24_MODELS="$arm=$artifact" BW24_ADDR="127.0.0.1:$port" \
    taskset -c "$cpus" "$SERVER_BIN" >"$server_log" 2>&1 &
  server_pid=$!
  echo "$server_pid" > "$arm_log/server.pid"

  local deadline=$((SECONDS + SERVER_HEALTH_TIMEOUT_S))
  while ((SECONDS < deadline)); do
    if curl -fsS --max-time 5 "http://127.0.0.1:$port/health" \
      >"$arm_log/health.json.tmp" 2>/dev/null \
      && python3 "$HERE/validate_server_health.py" "$arm_log/health.json.tmp" "$arm" --exact
    then
      mv "$arm_log/health.json.tmp" "$arm_log/health.json"
      break
    fi
    sleep 1
  done
  [[ -f "$arm_log/health.json" ]] || die "$arm practical server health timeout"

  taskset -c "$cpus" env \
    HOME="$HARBOR_HOME" PATH="$USER_PATH" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 ARM="$arm" PANEL="$panel" ARTIFACT="$artifact" \
    SERVER_BIN="$SERVER_BIN" SERVER_LOG="$server_log" HARBOR_BIN="$HARBOR_BIN" \
    LOCK="$PRACTICAL_LOCK" OUT_ROOT="$OUT_ROOT" RUN_ID="$RUN_ID" \
    BASE_URL="http://127.0.0.1:$port/v1" BW24_SPILL_IO=worker \
    BW24_SPILL_PREAD_DEPTH="$SPILL_DEPTH" BW24_SPILL_STATS=1 BW24_SERVE_SPEC=0 \
    "$HERE/run_practical_evals.sh" | tee "$arm_log/$panel.log"
  date -u +%FT%TZ > "$arm_log/complete"
)

gpu=0
for arm in "${ARMS[@]}"; do
  for panel in swe terminal; do
    cpu_start=$((gpu * 12))
    cpu_end=$((cpu_start + 11))
    run_panel "$arm" "$panel" "$gpu" "$cpu_start-$cpu_end" &
    WORKER_PIDS+=("$!")
    gpu=$((gpu + 1))
  done
done

failed=0
for pid in "${WORKER_PIDS[@]}"; do wait "$pid" || failed=1; done
((failed == 0)) || die "one or more practical workers failed"

COMPARE_ROOT="$OUT_ROOT/comparisons/$RUN_ID"
mkdir -p "$COMPARE_ROOT"
for baseline in "${ARMS[@]:0:2}"; do
  for candidate in "${ARMS[@]}"; do
    [[ "$baseline" != "$candidate" ]] || continue
    for panel in swe terminal; do
      python3 "$HERE/summarize_practical_results.py" \
        --baseline "$OUT_ROOT/$baseline/$panel/$RUN_ID" \
        --candidate "$OUT_ROOT/$candidate/$panel/$RUN_ID" \
        --panel "$panel" --lock "$PRACTICAL_LOCK" \
        --json-out "$COMPARE_ROOT/$baseline-vs-$candidate.$panel.json" \
        --markdown-out "$COMPARE_ROOT/$baseline-vs-$candidate.$panel.md"
    done
  done
done

selector_args=()
evidence_paths=()
if [[ -n "$CONFIRMATION_LOCK" ]]; then
  selector_args+=(--confirmation-lock "$CONFIRMATION_LOCK" --frontier "$FRONTIER")
  evidence_paths+=("$CONFIRMATION_LOCK" "$FRONTIER")
fi
python3 "$PRACTICAL_SELECTOR" \
  --promotion "$PROMOTION" --gate-lock "$GATE_LOCK" \
  --comparison-root "$COMPARE_ROOT" \
  "${selector_args[@]}" \
  --output "$OUT_ROOT/practical-promotion-$RUN_ID.json"
sha256sum "$RUN_CONFIG" "$PROMOTION" "$GATE_LOCK" "$PRACTICAL_LOCK" \
  "${evidence_paths[@]}" \
  "$OUT_ROOT/practical-promotion-$RUN_ID.json" > "$LOG_ROOT/$RUN_ID-evidence.sha256"
date -u +%FT%TZ | tee "$LOG_ROOT/complete"
trap - EXIT INT TERM
