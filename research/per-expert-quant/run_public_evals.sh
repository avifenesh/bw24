#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
HERE="$ROOT/research/per-expert-quant"
LOCK="$HERE/suite.lock.json"

: "${ARM:?set ARM to the experiment arm name, e.g. uniform_q4k_control or reap50_plus25}"
: "${MODEL:?set MODEL to the name configured in BW24_MODELS}"
: "${ARTIFACT:?set ARTIFACT to the served model file or overlay directory}"
: "${SERVER_BIN:?set SERVER_BIN to the active server binary}"
: "${SERVER_LOG:?set SERVER_LOG to the active server log}"
: "${BW24_SPILL_IO:?declare the active spill backend}"
: "${BW24_SPILL_PREAD_DEPTH:?declare the active worker depth}"
: "${BW24_SPILL_STATS:?declare spill telemetry state}"
: "${BW24_SERVE_SPEC:?declare speculative serving state}"
BASE_URL=${BASE_URL:-http://127.0.0.1:8080/v1/completions}
SUITE=${SUITE:-core}
OUT_ROOT=${OUT_ROOT:-$HERE/results}
CACHE_DIR=${CACHE_DIR:-$HERE/.cache}
EVAL_TIMEOUT_S=${EVAL_TIMEOUT_S:-}
NUM_CONCURRENT=${NUM_CONCURRENT:-1}
HARNESS_COMMIT=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["lm_eval_commit"])' "$LOCK")
HARNESS_DIR="$CACHE_DIR/lm-eval-${HARNESS_COMMIT:0:12}"
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}

[[ -e "$ARTIFACT" ]] || { echo "ARTIFACT does not exist: $ARTIFACT" >&2; exit 2; }
[[ -x "$SERVER_BIN" ]] || { echo "SERVER_BIN is not executable: $SERVER_BIN" >&2; exit 2; }
[[ -f "$SERVER_LOG" ]] || { echo "SERVER_LOG does not exist: $SERVER_LOG" >&2; exit 2; }
[[ "$BW24_SPILL_IO" == worker ]] || { echo "BW24_SPILL_IO must be worker" >&2; exit 2; }
[[ "$BW24_SPILL_PREAD_DEPTH" =~ ^[1-9][0-9]*$ ]] || {
  echo "BW24_SPILL_PREAD_DEPTH must be a positive integer" >&2
  exit 2
}
[[ "$BW24_SPILL_STATS" == 1 ]] || { echo "BW24_SPILL_STATS must be 1" >&2; exit 2; }
[[ "$BW24_SERVE_SPEC" == 0 ]] || { echo "BW24_SERVE_SPEC must be 0" >&2; exit 2; }

case "$SUITE" in
  core) TASKS=ifeval,gsm8k_cot,bbh_cot_fewshot,drop ;;
  candidate)
    TASKS=gpqa_diamond_cot_zeroshot,hendrycks_math500,mmlu_pro_history,mmlu_pro_other,mmlu_pro_economics,mmlu_pro_law,mmlu_pro_psychology
    if [[ ${LIMIT:-} != all ]]; then
      LIMIT=${LIMIT:-3}
    fi
    MAX_GEN_TOKS=${MAX_GEN_TOKS:-256}
    ;;
  code)
    if [[ ${BW24_UNSAFE_EVALS:-0} != 1 ]]; then
      echo "code evals execute model-generated Python; run in an isolated sandbox and set BW24_UNSAFE_EVALS=1" >&2
      exit 2
    fi
    TASKS=humaneval_instruct,mbpp_instruct
    ;;
  *) echo "unknown SUITE=$SUITE (expected candidate, core, or code)" >&2; exit 2 ;;
esac

SUITE_TASKS=$TASKS
requested_tasks=()
if [[ -n ${TASKS_OVERRIDE:-} ]]; then
  IFS=',' read -r -a requested_tasks <<< "$TASKS_OVERRIDE"
  declare -A seen_tasks=()
  for task in "${requested_tasks[@]}"; do
    [[ -n "$task" && ",$SUITE_TASKS," == *",$task,"* ]] || {
      echo "TASKS_OVERRIDE contains task outside SUITE=$SUITE: $task" >&2
      exit 2
    }
    [[ -z ${seen_tasks[$task]+x} ]] || {
      echo "TASKS_OVERRIDE contains duplicate task: $task" >&2
      exit 2
    }
    seen_tasks[$task]=1
  done
  TASKS=$TASKS_OVERRIDE
fi
if [[ -n ${SHARD_ID:-} && (
  ! ${SHARD_ID} =~ ^[A-Za-z0-9._-]+$ || ${SHARD_ID} == "." || ${SHARD_ID} == ".."
) ]]; then
  echo "SHARD_ID must contain only letters, digits, dot, underscore, or dash" >&2
  exit 2
fi
if [[ -n ${SHARD_ID:-} ]]; then
  [[ ${#requested_tasks[@]} == 1 && "$SHARD_ID" == "${requested_tasks[0]}" ]] || {
    echo "SHARD_ID requires exactly one matching TASKS_OVERRIDE task" >&2
    exit 2
  }
fi

RUN_DIR="$OUT_ROOT/$ARM/$RUN_ID"
if [[ -n ${SHARD_ID:-} ]]; then
  RUN_DIR="$RUN_DIR/shards/$SHARD_ID"
fi

if [[ -z "$EVAL_TIMEOUT_S" ]]; then
  if [[ "$SUITE" == candidate && ${LIMIT:-} == all ]]; then
    EVAL_TIMEOUT_S=432000
  else
    EVAL_TIMEOUT_S=14400
  fi
fi
[[ "$EVAL_TIMEOUT_S" =~ ^[1-9][0-9]*$ ]] || {
  echo "EVAL_TIMEOUT_S must be a positive integer (got $EVAL_TIMEOUT_S)" >&2
  exit 2
}
[[ "$NUM_CONCURRENT" =~ ^[1-9][0-9]*$ ]] || {
  echo "NUM_CONCURRENT must be a positive integer (got $NUM_CONCURRENT)" >&2
  exit 2
}
command -v timeout >/dev/null || {
  echo "GNU timeout is required" >&2
  exit 2
}

if [[ -e "$RUN_DIR" ]]; then
  if find "$RUN_DIR" -name 'results_*.json' -type f -print -quit 2>/dev/null | grep -q .; then
    echo "refusing to rerun completed output directory: $RUN_DIR" >&2
  else
    echo "refusing to overwrite existing partial output directory: $RUN_DIR" >&2
    echo "quarantine it with a timestamp before retrying this run or shard" >&2
  fi
  exit 3
fi
SERVER_ROOT=${BASE_URL%/v1/completions}
HEALTH_TMP=$(mktemp)
if ! curl -fsS --max-time 10 "$SERVER_ROOT/health" > "$HEALTH_TMP"; then
  rm -f "$HEALTH_TMP"
  echo "server health request failed: $SERVER_ROOT/health" >&2
  exit 2
fi
if ! python3 "$HERE/validate_server_health.py" "$HEALTH_TMP" "$MODEL" --exact; then
  rm -f "$HEALTH_TMP"
  exit 2
fi
mkdir -p "$CACHE_DIR" "$(dirname "$RUN_DIR")"
if ! mkdir "$RUN_DIR"; then
  rm -f "$HEALTH_TMP"
  echo "lost exclusive ownership race for output directory: $RUN_DIR" >&2
  exit 3
fi
mv "$HEALTH_TMP" "$RUN_DIR/health.json"
if [[ ! -d "$HARNESS_DIR/.git" ]]; then
  git init --quiet "$HARNESS_DIR"
  git -C "$HARNESS_DIR" remote add origin https://github.com/EleutherAI/lm-evaluation-harness.git
  git -C "$HARNESS_DIR" fetch --quiet --depth=1 origin "$HARNESS_COMMIT"
  git -C "$HARNESS_DIR" checkout --quiet --detach FETCH_HEAD
fi
python3 "$HERE/prepare_harness.py" "$HARNESS_DIR" --lock "$LOCK"
HARNESS_PYTHON="$HARNESS_DIR/.venv/bin/python"
HARNESS_CLI="$HARNESS_DIR/.venv/bin/lm_eval"
if [[ ! -x "$HARNESS_CLI" ]]; then
  UV_BIN=${UV_BIN:-$(command -v uv || true)}
  if [[ -z "$UV_BIN" && -x "${HOME:-}/.local/bin/uv" ]]; then
    UV_BIN="${HOME}/.local/bin/uv"
  fi
  [[ -n "$UV_BIN" && -x "$UV_BIN" ]] || {
    echo "uv is required to create the pinned lm-eval environment (set UV_BIN or install ~/.local/bin/uv)" >&2
    exit 2
  }
  "$UV_BIN" venv --python 3.12 "$HARNESS_DIR/.venv"
  # `uv sync --extra api` resolves every optional extra in lm-eval's universal lock; the pinned
  # checkout currently has mutually exclusive acpbench/vllm lark constraints. Install only the
  # backend and task dependency set this suite actually uses.
  "$UV_BIN" pip install --python "$HARNESS_PYTHON" -e "$HARNESS_DIR[api,ifeval]"
fi

cp "$LOCK" "$RUN_DIR/suite.lock.json"
if [[ -f "$ARTIFACT/manifest.json" ]]; then
  cp "$ARTIFACT/manifest.json" "$RUN_DIR/artifact-manifest.json"
fi
RUN_STARTED_UTC=$(date -u +%FT%TZ)
RUN_STARTED_NS=$(date +%s%N)
export ROOT RUN_DIR ARM MODEL SUITE TASKS LIMIT SHARD_ID BASE_URL HARNESS_COMMIT ARTIFACT MAX_GEN_TOKS EVAL_TIMEOUT_S NUM_CONCURRENT SERVER_BIN SERVER_LOG BW24_SPILL_IO BW24_SPILL_PREAD_DEPTH BW24_SPILL_STATS BW24_SERVE_SPEC RUN_STARTED_UTC
python3 - "$RUN_DIR/run-metadata.json" <<'PY'
import hashlib, json, os, pathlib, platform, re, subprocess, sys

def command(*args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unavailable: {exc}"

root = pathlib.Path(os.environ["ROOT"])
artifact = pathlib.Path(os.environ["ARTIFACT"]).resolve()
identity = artifact / "manifest.json" if artifact.is_dir() else artifact
server_bin_raw = os.environ.get("SERVER_BIN")
server_bin = pathlib.Path(server_bin_raw).resolve() if server_bin_raw else None
server_log_raw = os.environ.get("SERVER_LOG")
server_log = pathlib.Path(server_log_raw).resolve() if server_log_raw else None

spill_keys = ("reads", "bytes", "errors", "short_reads", "fallbacks", "buffer_waits", "ring_full")
cache_keys = ("hits", "misses", "staged_bytes", "slots")

def spill_snapshot(path):
    if path is None or not path.is_file():
        return None
    for line in reversed(path.read_text(errors="replace").splitlines()):
        if "[spill-pread] snapshot" not in line:
            continue
        values = {key: int(value) for key, value in re.findall(r"([a-z_]+)=([0-9]+)", line)}
        if all(key in values for key in spill_keys):
            return {key: values[key] for key in spill_keys}
    return {key: 0 for key in spill_keys}

def cache_snapshot(path):
    if path is None or not path.is_file():
        return None
    for line in reversed(path.read_text(errors="replace").splitlines()):
        if "[moe-cache] snapshot" not in line:
            continue
        values = {key: int(value) for key, value in re.findall(r"([a-z_]+)=([0-9]+)", line)}
        if all(key in values for key in cache_keys):
            return {key: values[key] for key in cache_keys}
    return {key: 0 for key in cache_keys}

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

metadata = {
    "arm": os.environ["ARM"],
    "model": os.environ["MODEL"],
    "suite": os.environ["SUITE"],
    "tasks": os.environ["TASKS"].split(","),
    "limit": os.environ.get("LIMIT") or None,
    "shard_id": os.environ.get("SHARD_ID") or None,
    "base_url": os.environ["BASE_URL"],
    "artifact": str(artifact),
    "artifact_identity_file": str(identity),
    "artifact_identity_sha256": sha256(identity),
    "bw24_commit": command("git", "-C", str(root), "rev-parse", "HEAD"),
    "lm_eval_commit": os.environ["HARNESS_COMMIT"],
    "eval_timeout_s": int(os.environ["EVAL_TIMEOUT_S"]),
    "num_concurrent": int(os.environ["NUM_CONCURRENT"]),
    "declared_spill_io": os.environ.get("BW24_SPILL_IO") or None,
    "declared_spill_pread_depth": os.environ.get("BW24_SPILL_PREAD_DEPTH") or None,
    "declared_spill_stats": os.environ.get("BW24_SPILL_STATS") or None,
    "declared_serve_spec": os.environ.get("BW24_SERVE_SPEC") or None,
    "started_utc": os.environ["RUN_STARTED_UTC"],
    "completed_utc": None,
    "elapsed_seconds": None,
    "evaluator_exit_code": None,
    "tee_exit_code": None,
    "completed_successfully": False,
    "max_gen_toks_override": (
        int(os.environ["MAX_GEN_TOKS"]) if os.environ.get("MAX_GEN_TOKS") else None
    ),
    "server_binary": str(server_bin) if server_bin else None,
    "server_binary_sha256": (
        sha256(server_bin) if server_bin and server_bin.is_file() else None
    ),
    "server_log_source": str(server_log) if server_log else None,
    "server_log": str(pathlib.Path(os.environ["RUN_DIR"]) / "server.log"),
    "server_log_sha256": None,
    "spill_snapshot_start": spill_snapshot(server_log),
    "spill_snapshot_end": None,
    "moe_cache_snapshot_start": cache_snapshot(server_log),
    "moe_cache_snapshot_end": None,
    "moe_cache_delta": None,
    "spill_delta": None,
    "platform": platform.platform(),
    "nvidia_smi": command("nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"),
}
pathlib.Path(sys.argv[1]).write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
PY

ARGS=(
  --model local-completions
  --model_args "model=$MODEL,base_url=$BASE_URL,num_concurrent=$NUM_CONCURRENT,max_retries=3,tokenized_requests=False,tokenizer_backend=none"
  --tasks "$TASKS"
  --batch_size 1
  --log_samples
  --output_path "$RUN_DIR"
)
if [[ -n ${LIMIT:-} && ${LIMIT} != all ]]; then ARGS+=(--limit "$LIMIT"); fi
if [[ -n ${MAX_GEN_TOKS:-} ]]; then ARGS+=(--gen_kwargs "max_gen_toks=$MAX_GEN_TOKS"); fi
if [[ "$SUITE" == code ]]; then ARGS+=(--confirm_run_unsafe_code); fi

set +e
timeout --signal=INT --kill-after=60s "${EVAL_TIMEOUT_S}s" \
  "$HARNESS_CLI" "${ARGS[@]}" 2>&1 | tee "$RUN_DIR/lm-eval.log"
pipeline_status=("${PIPESTATUS[@]}")
set -e
evaluator_status=${pipeline_status[0]}
tee_status=${pipeline_status[1]}
RUN_COMPLETED_UTC=$(date -u +%FT%TZ)
RUN_COMPLETED_NS=$(date +%s%N)
RUN_ELAPSED_SECONDS=$(python3 -c 'import sys; print((int(sys.argv[2]) - int(sys.argv[1])) / 1e9)' \
  "$RUN_STARTED_NS" "$RUN_COMPLETED_NS")
cp "$SERVER_LOG" "$RUN_DIR/server.log"
export RUN_COMPLETED_UTC RUN_ELAPSED_SECONDS evaluator_status tee_status
python3 - "$RUN_DIR/run-metadata.json" <<'PY'
import hashlib, json, os, pathlib, re, sys

path = pathlib.Path(sys.argv[1])
metadata = json.loads(path.read_text())
evaluator_status = int(os.environ["evaluator_status"])
tee_status = int(os.environ["tee_status"])
spill_keys = ("reads", "bytes", "errors", "short_reads", "fallbacks", "buffer_waits", "ring_full")
cache_keys = ("hits", "misses", "staged_bytes", "slots")

def spill_snapshot(raw_path):
    if not raw_path:
        return None
    spill_log = pathlib.Path(raw_path)
    if not spill_log.is_file():
        return None
    for line in reversed(spill_log.read_text(errors="replace").splitlines()):
        if "[spill-pread] snapshot" not in line:
            continue
        values = {key: int(value) for key, value in re.findall(r"([a-z_]+)=([0-9]+)", line)}
        if all(key in values for key in spill_keys):
            return {key: values[key] for key in spill_keys}
    return {key: 0 for key in spill_keys}

def cache_snapshot(raw_path):
    if not raw_path:
        return None
    cache_log = pathlib.Path(raw_path)
    if not cache_log.is_file():
        return None
    for line in reversed(cache_log.read_text(errors="replace").splitlines()):
        if "[moe-cache] snapshot" not in line:
            continue
        values = {key: int(value) for key, value in re.findall(r"([a-z_]+)=([0-9]+)", line)}
        if all(key in values for key in cache_keys):
            return {key: values[key] for key in cache_keys}
    return {key: 0 for key in cache_keys}

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

spill_end = spill_snapshot(metadata.get("server_log_source"))
spill_start = metadata.get("spill_snapshot_start")
spill_delta = None
if spill_start is not None and spill_end is not None:
    spill_delta = {key: spill_end[key] - spill_start[key] for key in spill_keys}
cache_end = cache_snapshot(metadata.get("server_log_source"))
cache_start = metadata.get("moe_cache_snapshot_start")
cache_delta = None
if cache_start is not None and cache_end is not None:
    cache_delta = {
        key: cache_end[key] - cache_start[key]
        for key in ("hits", "misses", "staged_bytes")
    }
metadata.update({
    "completed_utc": os.environ["RUN_COMPLETED_UTC"],
    "elapsed_seconds": float(os.environ["RUN_ELAPSED_SECONDS"]),
    "evaluator_exit_code": evaluator_status,
    "tee_exit_code": tee_status,
    "completed_successfully": evaluator_status == 0 and tee_status == 0,
    "server_log_sha256": sha256(pathlib.Path(metadata["server_log"])),
    "spill_snapshot_end": spill_end,
    "spill_delta": spill_delta,
    "moe_cache_snapshot_end": cache_end,
    "moe_cache_delta": cache_delta,
})
path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
PY
if (( evaluator_status != 0 )); then
  exit "$evaluator_status"
fi
if (( tee_status != 0 )); then
  exit "$tee_status"
fi
