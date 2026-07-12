#!/usr/bin/env bash
set -euo pipefail

ROOT=${BW24_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
HERE="$ROOT/research/per-expert-quant"
LOCK=${LOCK:-$HERE/practical-evals.lock.json}
FULL_TASK_LOCK=${FULL_TASK_LOCK:-$HERE/full-practical-tasks.lock.json}
HARBOR_BIN=${HARBOR_BIN:-$(command -v harbor || true)}
BASE_URL=${BASE_URL:-http://127.0.0.1:8080/v1}
OUT_ROOT=${OUT_ROOT:-$HERE/results/practical-full}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
RESUME_ATTEMPTS=${RESUME_ATTEMPTS:-3}
TASKS_JSON=${TASKS_JSON:-}
SHARD_ID=${SHARD_ID:-}

: "${ARM:?set the exact served arm}"
: "${PANEL:?set PANEL to swe or terminal}"
: "${ARTIFACT:?set the served artifact}"
: "${SERVER_BIN:?set the active server binary}"
: "${SERVER_LOG:?set the active server log}"
: "${BW24_SPILL_IO:?declare spill backend}"
: "${BW24_SPILL_PREAD_DEPTH:?declare worker depth}"
: "${BW24_SPILL_STATS:?declare spill telemetry}"
: "${BW24_SERVE_SPEC:?declare speculative serving state}"

die() { echo "error: $*" >&2; exit 2; }

[[ "$ARM" =~ ^[A-Za-z0-9._-]+$ ]] || die "invalid ARM"
[[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || die "invalid RUN_ID"
[[ "$PANEL" == swe || "$PANEL" == terminal ]] || die "PANEL must be swe or terminal"
[[ -x "$HARBOR_BIN" && -x "$SERVER_BIN" ]] || die "missing Harbor or server"
[[ -f "$LOCK" && -f "$FULL_TASK_LOCK" && -f "$SERVER_LOG" \
  && -f "$ARTIFACT/manifest.json" ]] || die "missing evidence input"
if [[ -n "$SHARD_ID" ]]; then
  [[ "$SHARD_ID" =~ ^[A-Za-z0-9._-]+$ ]] || die "invalid SHARD_ID"
fi
[[ "$BW24_SPILL_IO" == worker && "$BW24_SPILL_STATS" == 1 && "$BW24_SERVE_SPEC" == 0 ]] \
  || die "full practical protocol differs"
[[ "$BW24_SPILL_PREAD_DEPTH" =~ ^[1-9][0-9]*$ ]] || die "invalid spill depth"
[[ "$RESUME_ATTEMPTS" =~ ^[0-9]+$ ]] || die "invalid resume attempt count"
docker info >/dev/null 2>&1 || die "Docker daemon unavailable"
[[ "$($HARBOR_BIN --version)" == 0.18.0 ]] || die "Harbor version differs"
python3 "$HERE/validate_practical_eval_lock.py" --lock "$LOCK" >/dev/null

readarray -t suite < <(python3 - "$LOCK" "$FULL_TASK_LOCK" "$PANEL" "$TASKS_JSON" <<'PY'
import hashlib, json, sys
lock = json.load(open(sys.argv[1]))
full = json.load(open(sys.argv[2]))
panel = sys.argv[3]
if full.get("format") != "bw24-full-practical-task-lock-v1":
    raise SystemExit("wrong full task lock format")
if full.get("source_practical_lock_sha256") != hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest():
    raise SystemExit("full task lock does not bind practical lock")
if panel == "swe":
    x, fx, prefix = lock["swe_bench_verified"], full["swe"], "swe-bench/"
    name, digest = x["harbor_dataset"], x["harbor_dataset_digest"]
else:
    x, fx, prefix = lock["terminal_bench_2"], full["terminal"], "terminal-bench/"
    name, digest = x["dataset"], x["dataset_digest"]
if fx.get("dataset") != name or fx.get("dataset_digest") != digest:
    raise SystemExit("full task lock dataset differs")
available = {row["name"]: row["digest"] for row in fx["tasks"]}
if sys.argv[4]:
    selected_names = json.loads(sys.argv[4])
    if not isinstance(selected_names, list) or not selected_names or len(set(selected_names)) != len(selected_names):
        raise SystemExit("TASKS_JSON must be a non-empty unique list")
else:
    selected_names = sorted(available)
if any(task not in available for task in selected_names):
    raise SystemExit("TASKS_JSON contains task outside full lock")
selected = [{"name": task, "digest": available[task]} for task in selected_names]
print(f"{name}@{digest}")
print(name)
print(digest)
print(len(selected))
print(json.dumps(selected, sort_keys=True, separators=(",", ":")))
for row in selected:
    if not row["name"].startswith(prefix):
        raise SystemExit("task prefix differs")
    print(row["name"])
PY
)
(( ${#suite[@]} >= 6 )) || die "full suite did not resolve"
DATASET=${suite[0]}
DATASET_NAME=${suite[1]}
DATASET_DIGEST=${suite[2]}
EXPECTED_TASKS=${suite[3]}
SELECTED_TASKS_JSON=${suite[4]}
INCLUDE_TASKS=("${suite[@]:5}")
(( ${#INCLUDE_TASKS[@]} == EXPECTED_TASKS )) || die "selected task count differs"

SERVER_ROOT=${BASE_URL%/v1}
RUN_DIR="$OUT_ROOT/$ARM/$PANEL/$RUN_ID"
if [[ -n "$SHARD_ID" ]]; then RUN_DIR="$RUN_DIR/shards/$SHARD_ID"; fi
[[ ! -e "$RUN_DIR" ]] || die "refusing to overwrite $RUN_DIR"
mkdir -p "$(dirname "$RUN_DIR")"
mkdir "$RUN_DIR"
cp "$LOCK" "$RUN_DIR/practical-evals.lock.json"
cp "$FULL_TASK_LOCK" "$RUN_DIR/full-practical-tasks.lock.json"
cp "$ARTIFACT/manifest.json" "$RUN_DIR/artifact-manifest.json"
printf '%s\n' "$SELECTED_TASKS_JSON" > "$RUN_DIR/selected-tasks.json"
curl -fsS --max-time 10 "$SERVER_ROOT/health" > "$RUN_DIR/server-health.json"
python3 "$HERE/validate_server_health.py" "$RUN_DIR/server-health.json" "$ARM" --exact

export OPENAI_API_KEY=${BW24_API_KEY:-dummy}
MODEL_INFO='{"max_input_tokens":8192,"max_output_tokens":512,"input_cost_per_token":0,"output_cost_per_token":0}'
CALL_KWARGS='{"max_tokens":512}'
CMD=(
  "$HARBOR_BIN" run --dataset "$DATASET" --agent terminus-2 --model "openai/$ARM"
  --job-name "$RUN_ID" --jobs-dir "$RUN_DIR/jobs" --env docker --cpus limit --memory limit
  --n-concurrent 1 --n-concurrent-agents 1 --n-attempts 1 --max-retries 0
  --agent-timeout-multiplier 4.0 --yes
  --agent-kwarg "api_base=$BASE_URL" --agent-kwarg temperature=0
  --agent-kwarg max_turns=20 --agent-kwarg parser_name=json
  --agent-kwarg proactive_summarization_threshold=1024
  --agent-kwarg enable_summarize=true --agent-kwarg store_all_messages=true
  --agent-kwarg record_terminal_session=true --agent-kwarg "model_info=$MODEL_INFO"
  --agent-kwarg "llm_call_kwargs=$CALL_KWARGS"
)
for task in "${INCLUDE_TASKS[@]}"; do CMD+=(--include-task-name "$task"); done
"${CMD[@]}" --print-config > "$RUN_DIR/resolved-harbor-config.json"

STARTED_UTC=$(date -u +%FT%TZ)
STARTED_NS=$(date +%s%N)
export ROOT LOCK FULL_TASK_LOCK ARM PANEL RUN_ID RUN_DIR SHARD_ID DATASET DATASET_NAME \
  DATASET_DIGEST EXPECTED_TASKS SELECTED_TASKS_JSON \
  BASE_URL SERVER_BIN SERVER_LOG ARTIFACT STARTED_UTC BW24_SPILL_IO \
  BW24_SPILL_PREAD_DEPTH BW24_SPILL_STATS BW24_SERVE_SPEC HARBOR_BIN
python3 - "$RUN_DIR/run-metadata.json" <<'PY'
import hashlib, json, os, pathlib, re, subprocess, sys

def sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(16 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def spill(path):
    keys = ("reads", "bytes", "errors", "short_reads", "fallbacks", "buffer_waits", "ring_full")
    for line in reversed(pathlib.Path(path).read_text(errors="replace").splitlines()):
        if "[spill-pread] snapshot" not in line:
            continue
        values = {k: int(v) for k, v in re.findall(r"([a-z_]+)=([0-9]+)", line)}
        if all(k in values for k in keys):
            return {k: values[k] for k in keys}
    return {k: 0 for k in keys}

manifest = pathlib.Path(os.environ["ARTIFACT"]) / "manifest.json"
server = pathlib.Path(os.environ["SERVER_BIN"]).resolve()
server_log = pathlib.Path(os.environ["SERVER_LOG"]).resolve()
payload = {
    "format": "bw24-full-practical-run-v1", "arm": os.environ["ARM"],
    "panel": os.environ["PANEL"], "run_id": os.environ["RUN_ID"],
    "shard_id": os.environ.get("SHARD_ID") or None,
    "dataset": os.environ["DATASET"], "dataset_name": os.environ["DATASET_NAME"],
    "dataset_digest": os.environ["DATASET_DIGEST"],
    "expected_tasks": int(os.environ["EXPECTED_TASKS"]), "base_url": os.environ["BASE_URL"],
    "harbor_version": subprocess.check_output([os.environ["HARBOR_BIN"], "--version"], text=True).strip(),
    "bw24_commit": subprocess.check_output(["git", "-C", os.environ["ROOT"], "rev-parse", "HEAD"], text=True).strip(),
    "lock_sha256": sha(os.environ["LOCK"]),
    "full_task_lock_sha256": sha(os.environ["FULL_TASK_LOCK"]),
    "selected_tasks_sha256": sha(pathlib.Path(os.environ["RUN_DIR"]) / "selected-tasks.json"),
    "artifact_manifest_sha256": sha(manifest),
    "artifact_bytes": json.loads(manifest.read_text()).get("artifact_bytes"),
    "server_binary": str(server), "server_binary_sha256": sha(server),
    "server_log_source": str(server_log),
    "server_log": str(pathlib.Path(os.environ["RUN_DIR"]) / "server.log"),
    "declared_spill_io": os.environ["BW24_SPILL_IO"],
    "declared_spill_pread_depth": os.environ["BW24_SPILL_PREAD_DEPTH"],
    "declared_spill_stats": os.environ["BW24_SPILL_STATS"],
    "declared_serve_spec": os.environ["BW24_SERVE_SPEC"],
    "spill_snapshot_start": spill(server_log), "spill_snapshot_end": None,
    "spill_delta": None, "started_utc": os.environ["STARTED_UTC"],
    "completed_utc": None, "elapsed_seconds": None, "harbor_exit_code": None,
    "tee_exit_code": None, "completed_successfully": False,
}
pathlib.Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

set +e
"${CMD[@]}" 2>&1 | tee "$RUN_DIR/harbor.log"
statuses=("${PIPESTATUS[@]}")
set -e
HARBOR_EXIT=${statuses[0]}
TEE_EXIT=${statuses[1]}
PROCESS_RESUMES=0
JOB_PATH="$RUN_DIR/jobs/$RUN_ID"
while (( HARBOR_EXIT != 0 && PROCESS_RESUMES < RESUME_ATTEMPTS )) \
  && [[ -f "$JOB_PATH/config.json" ]]; do
  PROCESS_RESUMES=$((PROCESS_RESUMES + 1))
  echo "process-level Harbor resume $PROCESS_RESUMES/$RESUME_ATTEMPTS" | tee -a "$RUN_DIR/harbor.log"
  set +e
  "$HARBOR_BIN" job resume --job-path "$JOB_PATH" 2>&1 | tee -a "$RUN_DIR/harbor.log"
  resumed=("${PIPESTATUS[@]}")
  set -e
  HARBOR_EXIT=${resumed[0]}
  (( resumed[1] == 0 )) || TEE_EXIT=${resumed[1]}
done
COMPLETED_UTC=$(date -u +%FT%TZ)
COMPLETED_NS=$(date +%s%N)
ELAPSED_SECONDS=$(python3 -c 'import sys; print((int(sys.argv[2])-int(sys.argv[1]))/1e9)' \
  "$STARTED_NS" "$COMPLETED_NS")
cp "$SERVER_LOG" "$RUN_DIR/server.log"
docker image ls --no-trunc --digests --format '{{json .}}' | sort > "$RUN_DIR/container-images.jsonl"
export HARBOR_EXIT TEE_EXIT PROCESS_RESUMES COMPLETED_UTC ELAPSED_SECONDS
python3 - "$RUN_DIR" <<'PY'
import hashlib, json, math, os, pathlib, re, sys

root = pathlib.Path(sys.argv[1])
metadata_path = root / "run-metadata.json"
metadata = json.loads(metadata_path.read_text())
keys = ("reads", "bytes", "errors", "short_reads", "fallbacks", "buffer_waits", "ring_full")

def sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(16 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def spill(path):
    for line in reversed(pathlib.Path(path).read_text(errors="replace").splitlines()):
        if "[spill-pread] snapshot" not in line:
            continue
        values = {k: int(v) for k, v in re.findall(r"([a-z_]+)=([0-9]+)", line)}
        if all(k in values for k in keys):
            return {k: values[k] for k in keys}
    return {k: 0 for k in keys}

job = root / "jobs" / metadata["run_id"]
result_path = job / "result.json"
if not result_path.is_file():
    raise SystemExit(f"missing Harbor result {result_path}")
result = json.loads(result_path.read_text())
expected = metadata["expected_tasks"]
selected = {row["name"]: row["digest"] for row in json.load(open(root / "selected-tasks.json"))}
if len(selected) != expected:
    raise SystemExit("selected task lock count differs")
stats = result.get("stats", {})
if not (
    result.get("n_total_trials") == expected
    and stats.get("n_completed_trials") == expected
    and stats.get("n_errored_trials") == 0
    and stats.get("n_cancelled_trials") == 0
    and stats.get("n_retries") == 0
):
    raise SystemExit("Harbor full-suite completion counts differ")
trials = []
trial_dirs = sorted(path for path in job.iterdir() if path.is_dir())
if len(trial_dirs) != expected:
    raise SystemExit("full practical trial directory count differs")
for trial_dir in trial_dirs:
    path = trial_dir / "result.json"
    if not path.is_file():
        raise SystemExit(f"missing full practical trial result {path}")
    row = json.loads(path.read_text())
    task = row.get("task_name")
    reward = row.get("verifier_result", {}).get("rewards", {}).get("reward")
    ref = row.get("task_id", {}).get("ref")
    if (
        not isinstance(task, str) or not task or not isinstance(ref, str) or not ref
        or row.get("exception_info") is not None
        or not isinstance(reward, (int, float)) or isinstance(reward, bool)
        or not math.isfinite(float(reward)) or not 0 <= reward <= 1
    ):
        raise SystemExit(f"invalid full practical trial {path}")
    trials.append({"task": task, "task_digest": ref, "reward": float(reward), "result_sha256": sha(path)})
if len(trials) != expected or len({row["task"] for row in trials}) != expected:
    raise SystemExit("full practical trial set differs")
trials.sort(key=lambda row: row["task"])
if {row["task"]: row["task_digest"] for row in trials} != selected:
    raise SystemExit("full practical task names/digests differ from selected lock")
(root / "validated-trials.json").write_text(json.dumps(trials, indent=2, sort_keys=True) + "\n")
end = spill(metadata["server_log_source"])
start = metadata["spill_snapshot_start"]
metadata.update({
    "completed_utc": os.environ["COMPLETED_UTC"],
    "elapsed_seconds": float(os.environ["ELAPSED_SECONDS"]),
    "harbor_exit_code": int(os.environ["HARBOR_EXIT"]), "tee_exit_code": int(os.environ["TEE_EXIT"]),
    "completed_successfully": int(os.environ["HARBOR_EXIT"]) == 0 and int(os.environ["TEE_EXIT"]) == 0,
    "process_resume_attempts": int(os.environ["PROCESS_RESUMES"]),
    "server_log_sha256": sha(root / "server.log"),
    "resolved_harbor_config_sha256": sha(root / "resolved-harbor-config.json"),
    "container_images_sha256": sha(root / "container-images.jsonl"),
    "harbor_result_sha256": sha(result_path), "validated_trials_sha256": sha(root / "validated-trials.json"),
    "spill_snapshot_end": end, "spill_delta": {key: end[key] - start[key] for key in keys},
    "solved": sum(row["reward"] for row in trials),
})
metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
PY

(( HARBOR_EXIT == 0 )) || exit "$HARBOR_EXIT"
(( TEE_EXIT == 0 )) || exit "$TEE_EXIT"
echo "$RUN_DIR"
