#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
HERE="$ROOT/research/per-expert-quant"
LOCK=${LOCK:-$HERE/practical-evals.lock.json}
HARBOR_BIN=${HARBOR_BIN:-$(command -v harbor || true)}
HARBOR_TASK_CACHE_ROOT=${HARBOR_TASK_CACHE_ROOT:-$HOME/.cache/harbor/tasks/packages}
BASE_URL=${BASE_URL:-http://127.0.0.1:8080/v1}
OUT_ROOT=${OUT_ROOT:-$HERE/results/practical}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
SERVER_BIN=${SERVER_BIN:-}
PILOT_TASK=${PILOT_TASK:-}

: "${ARM:?set ARM to the exact served model name}"
: "${PANEL:?set PANEL to swe or terminal}"
: "${ARTIFACT:?set ARTIFACT to the served overlay directory}"
: "${SERVER_BIN:?set SERVER_BIN to the active practical server binary}"
: "${SERVER_LOG:?set SERVER_LOG to the active practical server log}"
: "${BW24_SPILL_IO:?declare the active spill backend}"
: "${BW24_SPILL_PREAD_DEPTH:?declare the active worker depth}"
: "${BW24_SPILL_STATS:?declare spill telemetry state}"
: "${BW24_SERVE_SPEC:?declare speculative serving state}"

die() {
  echo "error: $*" >&2
  exit 2
}

[[ "$ARM" =~ ^[A-Za-z0-9._-]+$ ]] || die "ARM may contain only letters, digits, dot, underscore, and dash"
[[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || die "RUN_ID may contain only letters, digits, dot, underscore, and dash"
[[ "$PANEL" == swe || "$PANEL" == terminal ]] || die "PANEL must be swe or terminal"
[[ -x "$HARBOR_BIN" ]] || die "Harbor is required"
[[ -d "$HARBOR_TASK_CACHE_ROOT/swe-bench" ]] || die "missing cached SWE tasks: $HARBOR_TASK_CACHE_ROOT/swe-bench"
[[ -d "$HARBOR_TASK_CACHE_ROOT/terminal-bench" ]] || die "missing cached Terminal tasks: $HARBOR_TASK_CACHE_ROOT/terminal-bench"
[[ -f "$LOCK" ]] || die "missing practical eval lock: $LOCK"
[[ -x "$SERVER_BIN" ]] || die "missing executable server: $SERVER_BIN"
[[ -f "$SERVER_LOG" ]] || die "missing server log: $SERVER_LOG"
[[ -f "$ARTIFACT/manifest.json" ]] || die "missing artifact manifest: $ARTIFACT/manifest.json"
[[ "$BW24_SPILL_IO" == worker ]] || die "practical eval requires BW24_SPILL_IO=worker"
[[ "$BW24_SPILL_PREAD_DEPTH" =~ ^[1-9][0-9]*$ ]] || die "invalid spill depth"
[[ "$BW24_SPILL_STATS" == 1 ]] || die "practical eval requires BW24_SPILL_STATS=1"
[[ "$BW24_SERVE_SPEC" == 0 ]] || die "practical eval requires BW24_SERVE_SPEC=0"
command -v curl >/dev/null || die "curl is required"
command -v docker >/dev/null || die "Docker is required"
docker info >/dev/null 2>&1 || die "Docker daemon is unavailable"

HARBOR_VERSION=$($HARBOR_BIN --version)
[[ "$HARBOR_VERSION" == 0.18.0 ]] || die "expected Harbor 0.18.0, got $HARBOR_VERSION"
python3 "$HERE/validate_practical_eval_lock.py" --lock "$LOCK" \
  --swe-harbor-root "$HARBOR_TASK_CACHE_ROOT/swe-bench" \
  --terminal-root "$HARBOR_TASK_CACHE_ROOT/terminal-bench"
MAX_TURNS=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["protocol"]["agent_scaffold"]["max_turns"])' "$LOCK")

mapfile -t selection < <(python3 - "$LOCK" "$PANEL" <<'PY'
import json, sys
lock = json.load(open(sys.argv[1]))
if sys.argv[2] == "swe":
    suite = lock["swe_bench_verified"]
    print(f"{suite['harbor_dataset']}@{suite['harbor_dataset_digest']}")
    for task in suite["tasks"]:
        print(task["harbor_task"])
else:
    suite = lock["terminal_bench_2"]
    print(f"{suite['dataset']}@{suite['dataset_digest']}")
    for task in suite["tasks"]:
        print(task["name"])
PY
)
(( ${#selection[@]} == 13 )) || die "lock did not resolve one dataset plus 12 tasks"
DATASET=${selection[0]}
TASKS=("${selection[@]:1}")
if [[ -n "$PILOT_TASK" ]]; then
  found=0
  for task in "${TASKS[@]}"; do [[ "$task" == "$PILOT_TASK" ]] && found=1; done
  ((found == 1)) || die "pilot task is outside the frozen panel"
  TASKS=("$PILOT_TASK")
fi

SERVER_ROOT=${BASE_URL%/v1}
RUN_DIR="$OUT_ROOT/$ARM/$PANEL/$RUN_ID"
[[ ! -e "$RUN_DIR" ]] || die "refusing to overwrite practical evidence: $RUN_DIR"
mkdir -p "$(dirname "$RUN_DIR")"
mkdir "$RUN_DIR"
cp "$LOCK" "$RUN_DIR/practical-evals.lock.json"
cp "$ARTIFACT/manifest.json" "$RUN_DIR/artifact-manifest.json"

curl -fsS --max-time 10 "$SERVER_ROOT/health" > "$RUN_DIR/server-health.json"
python3 "$HERE/validate_server_health.py" "$RUN_DIR/server-health.json" "$ARM" --exact

AUTH_ARGS=()
if [[ -n ${BW24_API_KEY:-} ]]; then
  AUTH_ARGS=(-H "Authorization: Bearer $BW24_API_KEY")
fi
CHAT_STATUS=$(curl -sS --max-time 10 -o "$RUN_DIR/chat-route-probe.json" -w '%{http_code}' \
  "${AUTH_ARGS[@]}" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$ARM\",\"messages\":[]}" "$BASE_URL/chat/completions")
[[ "$CHAT_STATUS" == 400 ]] || die "chat route probe expected HTTP 400, got $CHAT_STATUS"

export OPENAI_API_KEY=${BW24_API_KEY:-dummy}
MODEL_INFO='{"max_input_tokens":5120,"max_output_tokens":3072,"input_cost_per_token":0,"output_cost_per_token":0}'
CALL_KWARGS='{"max_tokens":3072,"timeout":7200}'
CMD=(
  "$HARBOR_BIN" run
  --dataset "$DATASET"
  --agent terminus-2
  --model "openai/$ARM"
  --job-name "$RUN_ID"
  --jobs-dir "$RUN_DIR/jobs"
  --env docker
  --cpus limit
  --memory limit
  --n-concurrent 1
  --n-concurrent-agents 1
  --n-attempts 1
  --max-retries 0
  --agent-timeout-multiplier 4.0
  --yes
  --agent-kwarg "api_base=$BASE_URL"
  --agent-kwarg temperature=0
  --agent-kwarg "max_turns=$MAX_TURNS"
  --agent-kwarg parser_name=json
  --agent-kwarg proactive_summarization_threshold=1024
  --agent-kwarg enable_summarize=true
  --agent-kwarg store_all_messages=true
  --agent-kwarg record_terminal_session=true
  --agent-kwarg "model_info=$MODEL_INFO"
  --agent-kwarg "llm_call_kwargs=$CALL_KWARGS"
)
for task in "${TASKS[@]}"; do
  CMD+=(--include-task-name "$task")
done

"${CMD[@]}" --print-config > "$RUN_DIR/resolved-harbor-config.json"
STARTED_UTC=$(date -u +%FT%TZ)
STARTED_NS=$(date +%s%N)
export ROOT LOCK ARM PANEL RUN_ID RUN_DIR DATASET BASE_URL HARBOR_VERSION SERVER_BIN SERVER_LOG ARTIFACT STARTED_UTC BW24_SPILL_IO BW24_SPILL_PREAD_DEPTH BW24_SPILL_STATS BW24_SERVE_SPEC PILOT_TASK
python3 - "$RUN_DIR/run-metadata.json" <<'PY'
import hashlib, json, os, pathlib, re, subprocess, sys

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

server_raw = os.environ.get("SERVER_BIN")
server = pathlib.Path(server_raw).resolve() if server_raw else None
server_log = pathlib.Path(os.environ["SERVER_LOG"]).resolve()
artifact = pathlib.Path(os.environ["ARTIFACT"]).resolve()
manifest = artifact / "manifest.json"
spill_keys = ("reads", "bytes", "errors", "short_reads", "fallbacks", "buffer_waits", "ring_full")

def spill_snapshot(path):
    for line in reversed(path.read_text(errors="replace").splitlines()):
        if "[spill-pread] snapshot" not in line:
            continue
        values = {key: int(value) for key, value in re.findall(r"([a-z_]+)=([0-9]+)", line)}
        if all(key in values for key in spill_keys):
            return {key: values[key] for key in spill_keys}
    return {key: 0 for key in spill_keys}

manifest_payload = json.loads(manifest.read_text())
payload = {
    "format": "bw24-practical-run-v1",
    "arm": os.environ["ARM"], "panel": os.environ["PANEL"],
    "run_id": os.environ["RUN_ID"], "dataset": os.environ["DATASET"],
    "base_url": os.environ["BASE_URL"], "harbor_version": os.environ["HARBOR_VERSION"],
    "pilot_task": os.environ.get("PILOT_TASK") or None,
    "bw24_commit": subprocess.check_output(
        ["git", "-C", os.environ["ROOT"], "rev-parse", "HEAD"], text=True
    ).strip(),
    "lock_sha256": sha256(pathlib.Path(os.environ["LOCK"])),
    "artifact": str(artifact), "artifact_manifest_sha256": sha256(manifest),
    "artifact_bytes": manifest_payload.get("artifact_bytes"),
    "server_binary": str(server) if server else None,
    "server_binary_sha256": sha256(server) if server and server.is_file() else None,
    "server_log_source": str(server_log),
    "server_log": str(pathlib.Path(os.environ["RUN_DIR"]) / "server.log"),
    "server_log_sha256": None,
    "declared_spill_io": os.environ["BW24_SPILL_IO"],
    "declared_spill_pread_depth": os.environ["BW24_SPILL_PREAD_DEPTH"],
    "declared_spill_stats": os.environ["BW24_SPILL_STATS"],
    "declared_serve_spec": os.environ["BW24_SERVE_SPEC"],
    "spill_snapshot_start": spill_snapshot(server_log),
    "spill_snapshot_end": None, "spill_delta": None,
    "started_utc": os.environ["STARTED_UTC"], "completed_utc": None,
    "elapsed_seconds": None, "harbor_exit_code": None, "tee_exit_code": None,
    "completed_successfully": False,
}
pathlib.Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

set +e
"${CMD[@]}" 2>&1 | tee "$RUN_DIR/harbor.log"
statuses=("${PIPESTATUS[@]}")
set -e
HARBOR_EXIT=${statuses[0]}
TEE_EXIT=${statuses[1]}
COMPLETED_UTC=$(date -u +%FT%TZ)
COMPLETED_NS=$(date +%s%N)
ELAPSED_SECONDS=$(python3 -c 'import sys; print((int(sys.argv[2])-int(sys.argv[1]))/1e9)' "$STARTED_NS" "$COMPLETED_NS")
cp "$SERVER_LOG" "$RUN_DIR/server.log"
docker image ls --no-trunc --digests --format '{{json .}}' | sort > "$RUN_DIR/container-images.jsonl"
export HARBOR_EXIT TEE_EXIT COMPLETED_UTC ELAPSED_SECONDS
python3 - "$RUN_DIR/run-metadata.json" <<'PY'
import hashlib, json, os, pathlib, re, sys
path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text())
spill_keys = ("reads", "bytes", "errors", "short_reads", "fallbacks", "buffer_waits", "ring_full")
server_log = pathlib.Path(payload["server_log"])

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

def spill_snapshot(path):
    for line in reversed(path.read_text(errors="replace").splitlines()):
        if "[spill-pread] snapshot" not in line:
            continue
        values = {key: int(value) for key, value in re.findall(r"([a-z_]+)=([0-9]+)", line)}
        if all(key in values for key in spill_keys):
            return {key: values[key] for key in spill_keys}
    return {key: 0 for key in spill_keys}

spill_end = spill_snapshot(server_log)
spill_start = payload["spill_snapshot_start"]
payload.update({
    "completed_utc": os.environ["COMPLETED_UTC"],
    "elapsed_seconds": float(os.environ["ELAPSED_SECONDS"]),
    "harbor_exit_code": int(os.environ["HARBOR_EXIT"]),
    "tee_exit_code": int(os.environ["TEE_EXIT"]),
    "completed_successfully": int(os.environ["HARBOR_EXIT"]) == 0 and int(os.environ["TEE_EXIT"]) == 0,
    "server_log_sha256": sha256(server_log),
    "container_images_sha256": sha256(path.parent / "container-images.jsonl"),
    "spill_snapshot_end": spill_end,
    "spill_delta": {key: spill_end[key] - spill_start[key] for key in spill_keys},
})
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

(( HARBOR_EXIT == 0 )) || exit "$HARBOR_EXIT"
(( TEE_EXIT == 0 )) || exit "$TEE_EXIT"
echo "$RUN_DIR"
