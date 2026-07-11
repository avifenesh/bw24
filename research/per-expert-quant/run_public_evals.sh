#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
HERE="$ROOT/research/per-expert-quant"
LOCK="$HERE/suite.lock.json"

: "${ARM:?set ARM to the experiment arm name, e.g. uniform_q4k_control or reap50_plus25}"
: "${MODEL:?set MODEL to the name configured in BW24_MODELS}"
: "${ARTIFACT:?set ARTIFACT to the served model file or overlay directory}"
BASE_URL=${BASE_URL:-http://127.0.0.1:8080/v1/completions}
SUITE=${SUITE:-core}
OUT_ROOT=${OUT_ROOT:-$HERE/results}
CACHE_DIR=${CACHE_DIR:-$HERE/.cache}
EVAL_TIMEOUT_S=${EVAL_TIMEOUT_S:-}
NUM_CONCURRENT=${NUM_CONCURRENT:-1}
HARNESS_COMMIT=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["lm_eval_commit"])' "$LOCK")
HARNESS_DIR="$CACHE_DIR/lm-eval-${HARNESS_COMMIT:0:12}"
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}

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
mkdir -p "$CACHE_DIR" "$RUN_DIR"
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

SERVER_ROOT=${BASE_URL%/v1/completions}
curl -fsS --max-time 10 "$SERVER_ROOT/health" > "$RUN_DIR/health.json"
python3 "$HERE/validate_server_health.py" "$RUN_DIR/health.json" "$MODEL"
cp "$LOCK" "$RUN_DIR/suite.lock.json"
if [[ -f "$ARTIFACT/manifest.json" ]]; then
  cp "$ARTIFACT/manifest.json" "$RUN_DIR/artifact-manifest.json"
fi
RUN_STARTED_UTC=$(date -u +%FT%TZ)
RUN_STARTED_NS=$(date +%s%N)
export ROOT ARM MODEL SUITE TASKS LIMIT SHARD_ID BASE_URL HARNESS_COMMIT ARTIFACT MAX_GEN_TOKS EVAL_TIMEOUT_S NUM_CONCURRENT SERVER_BIN BW24_SPILL_IO BW24_SPILL_PREAD_DEPTH BW24_SPILL_STATS BW24_SERVE_SPEC RUN_STARTED_UTC
python3 - "$RUN_DIR/run-metadata.json" <<'PY'
import hashlib, json, os, pathlib, platform, subprocess, sys

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
export RUN_COMPLETED_UTC RUN_ELAPSED_SECONDS evaluator_status tee_status
python3 - "$RUN_DIR/run-metadata.json" <<'PY'
import json, os, pathlib, sys

path = pathlib.Path(sys.argv[1])
metadata = json.loads(path.read_text())
evaluator_status = int(os.environ["evaluator_status"])
tee_status = int(os.environ["tee_status"])
metadata.update({
    "completed_utc": os.environ["RUN_COMPLETED_UTC"],
    "elapsed_seconds": float(os.environ["RUN_ELAPSED_SECONDS"]),
    "evaluator_exit_code": evaluator_status,
    "tee_exit_code": tee_status,
    "completed_successfully": evaluator_status == 0 and tee_status == 0,
})
path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
PY
if (( evaluator_status != 0 )); then
  exit "$evaluator_status"
fi
if (( tee_status != 0 )); then
  exit "$tee_status"
fi
