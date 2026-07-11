#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
HERE="$ROOT/research/per-expert-quant"
LOCK=${LOCK:-$HERE/practical-evals.lock.json}
HARBOR_BIN=${HARBOR_BIN:-$(command -v harbor || true)}
BASE_URL=${BASE_URL:-http://127.0.0.1:8080/v1}
OUT_ROOT=${OUT_ROOT:-$HERE/results/practical}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
SERVER_BIN=${SERVER_BIN:-}

: "${ARM:?set ARM to the exact served model name}"
: "${PANEL:?set PANEL to swe or terminal}"

die() {
  echo "error: $*" >&2
  exit 2
}

[[ "$ARM" =~ ^[A-Za-z0-9._-]+$ ]] || die "ARM may contain only letters, digits, dot, underscore, and dash"
[[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || die "RUN_ID may contain only letters, digits, dot, underscore, and dash"
[[ "$PANEL" == swe || "$PANEL" == terminal ]] || die "PANEL must be swe or terminal"
[[ -x "$HARBOR_BIN" ]] || die "Harbor is required"
[[ -f "$LOCK" ]] || die "missing practical eval lock: $LOCK"
command -v curl >/dev/null || die "curl is required"
command -v docker >/dev/null || die "Docker is required"
docker info >/dev/null 2>&1 || die "Docker daemon is unavailable"

HARBOR_VERSION=$($HARBOR_BIN --version)
[[ "$HARBOR_VERSION" == 0.18.0 ]] || die "expected Harbor 0.18.0, got $HARBOR_VERSION"
python3 "$HERE/validate_practical_eval_lock.py" --lock "$LOCK"

mapfile -t selection < <(python3 - "$LOCK" "$PANEL" <<'PY'
import json, sys
lock = json.load(open(sys.argv[1]))
if sys.argv[2] == "swe":
    suite = lock["swe_bench_verified"]
    print(f"{suite['harbor_dataset']}@{suite['harbor_dataset_digest']}")
    for task in suite["tasks"]:
        print(task["instance_id"])
else:
    suite = lock["terminal_bench_2"]
    print(f"{suite['dataset']}@{suite['dataset_digest']}")
    for task in suite["tasks"]:
        print(task["name"].split("/", 1)[1])
PY
)
(( ${#selection[@]} == 13 )) || die "lock did not resolve one dataset plus 12 tasks"
DATASET=${selection[0]}
TASKS=("${selection[@]:1}")

SERVER_ROOT=${BASE_URL%/v1}
RUN_DIR="$OUT_ROOT/$ARM/$PANEL/$RUN_ID"
[[ ! -e "$RUN_DIR" ]] || die "refusing to overwrite practical evidence: $RUN_DIR"
mkdir -p "$(dirname "$RUN_DIR")"
mkdir "$RUN_DIR"
cp "$LOCK" "$RUN_DIR/practical-evals.lock.json"

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
MODEL_INFO='{"max_input_tokens":8192,"max_output_tokens":512,"input_cost_per_token":0,"output_cost_per_token":0}'
CALL_KWARGS='{"max_tokens":512}'
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
  --yes
  --agent-kwarg "api_base=$BASE_URL"
  --agent-kwarg temperature=0
  --agent-kwarg max_turns=20
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
export ROOT LOCK ARM PANEL RUN_ID DATASET BASE_URL HARBOR_VERSION SERVER_BIN STARTED_UTC
python3 - "$RUN_DIR/run-metadata.json" <<'PY'
import hashlib, json, os, pathlib, subprocess, sys

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

server_raw = os.environ.get("SERVER_BIN")
server = pathlib.Path(server_raw).resolve() if server_raw else None
payload = {
    "format": "bw24-practical-run-v1",
    "arm": os.environ["ARM"], "panel": os.environ["PANEL"],
    "run_id": os.environ["RUN_ID"], "dataset": os.environ["DATASET"],
    "base_url": os.environ["BASE_URL"], "harbor_version": os.environ["HARBOR_VERSION"],
    "bw24_commit": subprocess.check_output(
        ["git", "-C", os.environ["ROOT"], "rev-parse", "HEAD"], text=True
    ).strip(),
    "lock_sha256": sha256(pathlib.Path(os.environ["LOCK"])),
    "server_binary": str(server) if server else None,
    "server_binary_sha256": sha256(server) if server and server.is_file() else None,
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
export HARBOR_EXIT TEE_EXIT COMPLETED_UTC ELAPSED_SECONDS
python3 - "$RUN_DIR/run-metadata.json" <<'PY'
import json, os, pathlib, sys
path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text())
payload.update({
    "completed_utc": os.environ["COMPLETED_UTC"],
    "elapsed_seconds": float(os.environ["ELAPSED_SECONDS"]),
    "harbor_exit_code": int(os.environ["HARBOR_EXIT"]),
    "tee_exit_code": int(os.environ["TEE_EXIT"]),
    "completed_successfully": int(os.environ["HARBOR_EXIT"]) == 0 and int(os.environ["TEE_EXIT"]) == 0,
})
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

(( HARBOR_EXIT == 0 )) || exit "$HARBOR_EXIT"
(( TEE_EXIT == 0 )) || exit "$TEE_EXIT"
echo "$RUN_DIR"
