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
HARNESS_COMMIT=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["lm_eval_commit"])' "$LOCK")
HARNESS_DIR="$CACHE_DIR/lm-eval-${HARNESS_COMMIT:0:12}"
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
RUN_DIR="$OUT_ROOT/$ARM/$RUN_ID"

case "$SUITE" in
  core) TASKS=ifeval,gsm8k_cot,bbh_cot_fewshot,drop ;;
  code)
    if [[ ${BW24_UNSAFE_EVALS:-0} != 1 ]]; then
      echo "code evals execute model-generated Python; run in an isolated sandbox and set BW24_UNSAFE_EVALS=1" >&2
      exit 2
    fi
    TASKS=humaneval_instruct,mbpp_instruct
    ;;
  *) echo "unknown SUITE=$SUITE (expected core or code)" >&2; exit 2 ;;
esac

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
  uv venv --python 3.12 "$HARNESS_DIR/.venv"
  # `uv sync --extra api` resolves every optional extra in lm-eval's universal lock; the pinned
  # checkout currently has mutually exclusive acpbench/vllm lark constraints. Install only the
  # backend and task dependency set this suite actually uses.
  uv pip install --python "$HARNESS_PYTHON" -e "$HARNESS_DIR[api,ifeval]"
fi

SERVER_ROOT=${BASE_URL%/v1/completions}
curl -fsS "$SERVER_ROOT/health" > "$RUN_DIR/health.json"
cp "$LOCK" "$RUN_DIR/suite.lock.json"
if [[ -f "$ARTIFACT/manifest.json" ]]; then
  cp "$ARTIFACT/manifest.json" "$RUN_DIR/artifact-manifest.json"
fi
export ROOT ARM MODEL SUITE BASE_URL HARNESS_COMMIT ARTIFACT
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
    "base_url": os.environ["BASE_URL"],
    "artifact": str(artifact),
    "artifact_identity_file": str(identity),
    "artifact_identity_sha256": sha256(identity),
    "bw24_commit": command("git", "-C", str(root), "rev-parse", "HEAD"),
    "lm_eval_commit": os.environ["HARNESS_COMMIT"],
    "platform": platform.platform(),
    "nvidia_smi": command("nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"),
}
pathlib.Path(sys.argv[1]).write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
PY

ARGS=(
  --model local-completions
  --model_args "model=$MODEL,base_url=$BASE_URL,num_concurrent=1,max_retries=3,tokenized_requests=False,tokenizer_backend=none"
  --tasks "$TASKS"
  --batch_size 1
  --log_samples
  --output_path "$RUN_DIR"
)
if [[ -n ${LIMIT:-} ]]; then ARGS+=(--limit "$LIMIT"); fi
if [[ "$SUITE" == code ]]; then ARGS+=(--confirm_run_unsafe_code); fi

"$HARNESS_CLI" "${ARGS[@]}" 2>&1 | tee "$RUN_DIR/lm-eval.log"
