#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
HERE="$ROOT/research/per-expert-quant"
PANEL_LOCK=${PANEL_LOCK:-$HERE/hourish-panel.lock.json}
SUITE_LOCK="$HERE/suite.lock.json"
RUNTIME_KIND=${RUNTIME_KIND:-bw24}

: "${ARM:?set ARM}"
: "${MODEL:?set MODEL}"
: "${ARTIFACT:?set ARTIFACT}"
: "${RUN_ID:?set one shared RUN_ID for every arm}"
: "${SERVER_BIN:?set SERVER_BIN}"
: "${SERVER_LOG:?set SERVER_LOG}"
if [[ "$RUNTIME_KIND" == bw24 ]]; then
  : "${BW24_SPILL_IO:?set BW24_SPILL_IO}"
  : "${BW24_SPILL_PREAD_DEPTH:?set BW24_SPILL_PREAD_DEPTH}"
  : "${BW24_SPILL_STATS:?set BW24_SPILL_STATS}"
  : "${BW24_SERVE_SPEC:?set BW24_SERVE_SPEC}"
elif [[ "$RUNTIME_KIND" == external_openai ]]; then
  : "${RUNTIME_IDENTITY:?set RUNTIME_IDENTITY}"
else
  echo "unknown RUNTIME_KIND=$RUNTIME_KIND (expected bw24 or external_openai)" >&2
  exit 2
fi

OUT_ROOT=${OUT_ROOT:-$HERE/results-hourish}
CACHE_DIR=${CACHE_DIR:-$HERE/.cache}
BASE_URL=${BASE_URL:-http://127.0.0.1:8080/v1/completions}
NUM_CONCURRENT=${NUM_CONCURRENT:-1}
HOURISH_SHARD_TIMEOUT_S=${HOURISH_SHARD_TIMEOUT_S:-43200}

[[ -f "$PANEL_LOCK" ]] || { echo "missing panel lock: $PANEL_LOCK" >&2; exit 2; }
PANEL_SHA256=$(python3 "$HERE/validate_capability_panel.py" "$PANEL_LOCK" \
  --suite-lock "$SUITE_LOCK" --print-sha)
[[ "$NUM_CONCURRENT" == 1 ]] || {
  echo "the matched hourish panel requires NUM_CONCURRENT=1" >&2
  exit 2
}
[[ "$HOURISH_SHARD_TIMEOUT_S" =~ ^[1-9][0-9]*$ ]] || {
  echo "HOURISH_SHARD_TIMEOUT_S must be a positive integer" >&2
  exit 2
}

mapfile -t TASK_ROWS < <(python3 "$HERE/validate_capability_panel.py" "$PANEL_LOCK" \
  --suite-lock "$SUITE_LOCK" --task-rows)

shard_complete() {
  local task=$1
  local dir="$OUT_ROOT/$ARM/$RUN_ID/shards/$task"
  [[ -d "$dir" ]] || return 1
  python3 - "$dir" "$task" "$PANEL_SHA256" "$BASE_URL" <<'PY'
import json, pathlib, sys

run_dir = pathlib.Path(sys.argv[1])
task, panel_sha, base_url = sys.argv[2:]
metadata_path = run_dir / "run-metadata.json"
if not metadata_path.is_file():
    raise SystemExit(1)
metadata = json.load(open(metadata_path))
if not (
    metadata.get("completed_successfully") is True
    and metadata.get("evaluator_exit_code") == 0
    and metadata.get("tee_exit_code") == 0
    and metadata.get("tasks") == [task]
    and metadata.get("panel_lock_sha256") == panel_sha
    and metadata.get("base_url") == base_url
    and metadata.get("samples")
    and list(metadata["samples"]) == [task]
    and list(run_dir.glob("**/results_*.json"))
    and list(run_dir.glob(f"**/samples_{task}_*.jsonl"))
):
    raise SystemExit(1)
PY
}

for row in "${TASK_ROWS[@]}"; do
  IFS=$'\t' read -r task samples_json max_gen_toks <<< "$row"
  if shard_complete "$task"; then
    echo "hourish shard already complete: $ARM/$task"
    continue
  fi
  suite=candidate
  predict_only=0
  unsafe=0
  timeout_s=$HOURISH_SHARD_TIMEOUT_S
  if [[ "$task" == humaneval_instruct ]]; then
    suite=code
    predict_only=1
    unsafe=1
  fi
  echo "starting hourish shard: arm=$ARM task=$task samples=$samples_json"
  ARM="$ARM" MODEL="$MODEL" ARTIFACT="$ARTIFACT" \
    SERVER_BIN="$SERVER_BIN" SERVER_LOG="$SERVER_LOG" \
    RUNTIME_KIND="$RUNTIME_KIND" RUNTIME_IDENTITY="${RUNTIME_IDENTITY:-}" \
    BW24_SPILL_IO="${BW24_SPILL_IO:-}" \
    BW24_SPILL_PREAD_DEPTH="${BW24_SPILL_PREAD_DEPTH:-}" \
    BW24_SPILL_STATS="${BW24_SPILL_STATS:-}" BW24_SERVE_SPEC="${BW24_SERVE_SPEC:-}" \
    BASE_URL="$BASE_URL" OUT_ROOT="$OUT_ROOT" CACHE_DIR="$CACHE_DIR" RUN_ID="$RUN_ID" \
    SUITE="$suite" TASKS_OVERRIDE="$task" SHARD_ID="$task" \
    SAMPLES_JSON="$samples_json" PANEL_LOCK="$PANEL_LOCK" \
    MAX_GEN_TOKS="$max_gen_toks" NUM_CONCURRENT=1 EVAL_TIMEOUT_S="$timeout_s" \
    PREDICT_ONLY="$predict_only" BW24_UNSAFE_EVALS="$unsafe" \
    "$HERE/run_public_evals.sh"
done

echo "all hourish generation shards complete: $ARM/$RUN_ID"
