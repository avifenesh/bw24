#!/usr/bin/env bash
set -euo pipefail

ROOT=${BW24_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
HERE="$ROOT/research/per-expert-quant"
TRUSTED_ROOT=${TRUSTED_ROOT:-/data/results/per-expert-quant/trusted-full-v1}
TRUSTED_READY=${TRUSTED_READY:-/data/logs/trusted-full-v1/complete}
OUT_ROOT=${OUT_ROOT:-/data/results/per-expert-quant/full-agentic-v1}
LOG_ROOT=${LOG_ROOT:-/data/logs/full-agentic-v1}
SERVER_BIN=${SERVER_BIN:-/data/build/bw24-portable-ada-fix-target/release/bw24-server}
HARBOR_BIN=${HARBOR_BIN:-/data/bin/harbor-0.18.0-0a01ad6/harbor}
HARBOR_HOME=${HARBOR_HOME:-/data/cache/harbor-home}
FULL_RUNNER=${FULL_RUNNER:-$HERE/run_full_practical_evals.sh}
FULL_SUMMARIZER=${FULL_SUMMARIZER:-$HERE/summarize_full_practical_results.py}
FULL_TASK_LOCK=${FULL_TASK_LOCK:-$HERE/full-practical-tasks.lock.json}
SPILL_DEPTH=${SPILL_DEPTH:-8}
WAIT_INTERVAL_S=${WAIT_INTERVAL_S:-30}
SERVER_HEALTH_TIMEOUT_S=${SERVER_HEALTH_TIMEOUT_S:-1800}

die() { echo "error: $*" >&2; exit 2; }

[[ -x "$SERVER_BIN" && -x "$HARBOR_BIN" ]] || die "missing server or Harbor"
[[ -x "$FULL_RUNNER" ]] || die "missing full practical runner"
[[ -f "$FULL_SUMMARIZER" ]] || die "missing full practical summarizer"
[[ -f "$HERE/practical-evals.lock.json" ]] || die "missing practical lock"
[[ -f "$FULL_TASK_LOCK" ]] || die "missing full practical task lock"
mkdir -p "$LOG_ROOT" "$OUT_ROOT/run-configs"
exec 9>"$LOG_ROOT/transition.lock"
flock -n 9 || exit 0
exec >>"$LOG_ROOT/transition.log" 2>&1
echo "$(date -u +%FT%TZ) complete agentic transition started"

while [[ ! -f "$TRUSTED_READY" || ! -s "$TRUSTED_ROOT/_active-run-id" ]]; do
  sleep "$WAIT_INTERVAL_S"
done
TRUSTED_RUN_ID=$(<"$TRUSTED_ROOT/_active-run-id")
TRUSTED_REPORT="$TRUSTED_ROOT/_runs/$TRUSTED_RUN_ID/trusted-full-results.json"
while [[ ! -f "$TRUSTED_REPORT" ]]; do sleep "$WAIT_INTERVAL_S"; done

mapfile -t ARMS < <(python3 - "$TRUSTED_REPORT" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
if d.get("format") != "bw24-promoted-candidate-v1" or d.get("n_per_task") != "all":
    raise SystemExit("wrong trusted-full report")
arms = d.get("selection", {}).get("full_eval_arms")
if not isinstance(arms, list) or len(arms) != 2 or len(set(arms)) != 2:
    raise SystemExit("expected baseline plus one full-agentic finalist")
if arms[0] != "plain_quant":
    raise SystemExit("plain_quant must be full-agentic baseline")
for arm in arms:
    print(arm)
PY
)

artifact_for() {
  case "$1" in
    plain_quant) printf '%s\n' /scratch/bw24-artifacts/plain-quant ;;
    traffic_nvfp4_53_q2_139) printf '%s\n' /scratch/bw24-artifacts/traffic-nvfp4-53-q2-139 ;;
    prune100_unhealed|prune100_router_repair|prune100_joint_heal)
      printf '/scratch/bw24-artifacts-100gb-5f02c37/%s\n' "$1" ;;
    *) die "no artifact mapping for $1" ;;
  esac
}

for arm in "${ARMS[@]}"; do [[ -f "$(artifact_for "$arm")/manifest.json" ]] || die "missing $arm"; done

RUN_ID="full-agentic-v1-$(date -u +%Y%m%dT%H%M%SZ)"
RUN_CONFIG="$OUT_ROOT/run-configs/$RUN_ID.json"
export RUN_ID TRUSTED_REPORT SERVER_BIN HARBOR_BIN ROOT HERE FULL_TASK_LOCK
python3 - "$RUN_CONFIG" "${ARMS[@]}" <<'PY'
import hashlib, json, os, pathlib, subprocess, sys

def sha(path): return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
payload = {
    "format": "bw24-full-agentic-run-v1", "run_id": os.environ["RUN_ID"],
    "arms": sys.argv[2:], "baseline": "plain_quant",
    "suites": {"swe": 500, "terminal": 89},
    "trusted_full_report": {"path": os.environ["TRUSTED_REPORT"], "sha256": sha(os.environ["TRUSTED_REPORT"])},
    "practical_lock": {"path": str(pathlib.Path(os.environ["HERE"]) / "practical-evals.lock.json"),
                       "sha256": sha(pathlib.Path(os.environ["HERE"]) / "practical-evals.lock.json")},
    "full_task_lock": {"path": os.environ["FULL_TASK_LOCK"],
                       "sha256": sha(os.environ["FULL_TASK_LOCK"])},
    "server": {"path": os.environ["SERVER_BIN"], "sha256": sha(os.environ["SERVER_BIN"])},
    "harbor": {"path": os.environ["HARBOR_BIN"], "sha256": sha(os.environ["HARBOR_BIN"])},
    "bw24_commit": subprocess.check_output(["git", "-C", os.environ["ROOT"], "rev-parse", "HEAD"], text=True).strip(),
}
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

run_lane() (
  local arm=$1 gpu=$2 cpus=$3 lane=$4 lane_count=$5 artifact arm_log server_log server_pid port
  artifact=$(artifact_for "$arm")
  port=$((8080 + gpu))
  server_pid=""
  arm_log="$LOG_ROOT/$RUN_ID-$arm-lane$lane"
  server_log="$arm_log/server.log"
  mkdir -p "$arm_log"
  echo "$port" > "$arm_log/port"
  if curl -fsS --connect-timeout 1 --max-time 2 "http://127.0.0.1:$port/health" \
    >/dev/null 2>&1; then
    die "a server is already answering on full-agentic port $port"
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

  env PATH="$USER_PATH" \
    CUDA_VISIBLE_DEVICES="$gpu" BW24_COMPAT=openai BW24_SERVE_SPEC=0 BW24_KV_REUSE=0 \
    BW24_CTX=8192 BW24_FAST=1 BW24_MMVQ=1 BW24_MOE_CACHE=1 BW24_MOE_GROUPED=1 \
    BW24_MOE_PREWARM=1 BW24_MOE_PREFETCH=1 BW24_MOE_PAGE_PREFETCH=1 \
    BW24_MOE_PAGE_PREFETCH_WINDOW=8 BW24_MOE_MMAP_ADVICE=normal BW24_MOE_RESIDENT=1 \
    BW24_MOE_VRAM_FRAC=0.85 BW24_SPILL_IO=worker BW24_SPILL_PREAD_DEPTH="$SPILL_DEPTH" \
    BW24_SPILL_STATS=1 BW24_MODELS="$arm=$artifact" BW24_ADDR="127.0.0.1:$port" \
    taskset -c "$cpus" "$SERVER_BIN" >"$server_log" 2>&1 &
  server_pid=$!
  echo "$server_pid" > "$arm_log/server.pid"

  local deadline=$((SECONDS + SERVER_HEALTH_TIMEOUT_S))
  while ((SECONDS < deadline)); do
    if curl -fsS --max-time 5 "http://127.0.0.1:$port/health" \
      >"$arm_log/health.json.tmp" 2>/dev/null \
      && python3 "$HERE/validate_server_health.py" "$arm_log/health.json.tmp" "$arm" --exact
    then mv "$arm_log/health.json.tmp" "$arm_log/health.json"; break; fi
    sleep 1
  done
  [[ -f "$arm_log/health.json" ]] || die "$arm full-agentic health timeout"

  for panel in swe terminal; do
    tasks_json=$(python3 - "$FULL_TASK_LOCK" "$panel" "$lane" "$lane_count" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
rows = d[sys.argv[2]]["tasks"]
lane, count = map(int, sys.argv[3:])
selected = [row["name"] for index, row in enumerate(rows) if index % count == lane]
if not selected:
    raise SystemExit("empty full practical lane")
print(json.dumps(selected, separators=(",", ":")))
PY
)
    env \
      HOME="$HARBOR_HOME" PATH="$USER_PATH" ARM="$arm" PANEL="$panel" ARTIFACT="$artifact" \
      BW24_ROOT="$ROOT" \
      SERVER_BIN="$SERVER_BIN" SERVER_LOG="$server_log" HARBOR_BIN="$HARBOR_BIN" \
      LOCK="$HERE/practical-evals.lock.json" FULL_TASK_LOCK="$FULL_TASK_LOCK" \
      TASKS_JSON="$tasks_json" SHARD_ID="lane$lane-of-$lane_count" \
      OUT_ROOT="$OUT_ROOT" RUN_ID="$RUN_ID" \
      BASE_URL="http://127.0.0.1:$port/v1" BW24_SPILL_IO=worker \
      BW24_SPILL_PREAD_DEPTH="$SPILL_DEPTH" BW24_SPILL_STATS=1 BW24_SERVE_SPEC=0 \
      "$FULL_RUNNER" | tee "$arm_log/$panel.log"
  done
  date -u +%FT%TZ > "$arm_log/complete"
)

LANES_PER_ARM=4
gpu=0
for arm in "${ARMS[@]}"; do
  for ((lane=0; lane<LANES_PER_ARM; lane++)); do
    cpu_start=$((gpu * 12))
    cpu_end=$((cpu_start + 11))
    run_lane "$arm" "$gpu" "$cpu_start-$cpu_end" "$lane" "$LANES_PER_ARM" &
    WORKER_PIDS+=("$!")
    gpu=$((gpu + 1))
  done
done
failed=0
for pid in "${WORKER_PIDS[@]}"; do wait "$pid" || failed=1; done
((failed == 0)) || die "full-agentic worker failed"

COMPARE_ROOT="$OUT_ROOT/comparisons/$RUN_ID"
mkdir -p "$COMPARE_ROOT"
for panel in swe terminal; do
  python3 "$FULL_SUMMARIZER" \
    --baseline "$OUT_ROOT/${ARMS[0]}/$panel/$RUN_ID" \
    --candidate "$OUT_ROOT/${ARMS[1]}/$panel/$RUN_ID" \
    --panel "$panel" --lock "$HERE/practical-evals.lock.json" \
    --full-task-lock "$FULL_TASK_LOCK" \
    --output "$COMPARE_ROOT/$panel.json"
done
python3 - "$COMPARE_ROOT/swe.json" "$COMPARE_ROOT/terminal.json" "$COMPARE_ROOT/combined.json" <<'PY'
import json, pathlib, sys
swe, terminal = (json.load(open(path)) for path in sys.argv[1:3])
if swe["baseline"]["arm"] != terminal["baseline"]["arm"] or swe["candidate"]["arm"] != terminal["candidate"]["arm"]:
    raise SystemExit("full practical panel arms differ")
out = {
    "format": "bw24-full-agentic-comparison-v1", "baseline": swe["baseline"]["arm"],
    "candidate": swe["candidate"]["arm"], "swe": swe, "terminal": terminal,
    "total_tasks": swe["n_tasks"] + terminal["n_tasks"],
    "candidate_total_solved_delta": swe["candidate_solved_delta"] + terminal["candidate_solved_delta"],
    "note": "Complete SWE-Bench Verified and Terminal-Bench 2 at one attempt per task.",
}
pathlib.Path(sys.argv[3]).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
PY
sha256sum "$RUN_CONFIG" "$TRUSTED_REPORT" "$HERE/practical-evals.lock.json" \
  "$FULL_TASK_LOCK" \
  "$COMPARE_ROOT/swe.json" "$COMPARE_ROOT/terminal.json" "$COMPARE_ROOT/combined.json" \
  > "$LOG_ROOT/$RUN_ID-evidence.sha256"
date -u +%FT%TZ | tee "$LOG_ROOT/complete"
trap - EXIT INT TERM
