#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
HERE="$ROOT/research/per-expert-quant"
PANEL_LOCK=${PANEL_LOCK:-$HERE/hourish-panel.lock.json}
PANEL_SHA256=52596541d81f9ff4039854fada54f57934c95c77028d7bbc7654a3fc9dca1ace

: "${ARM:?set ARM}"
: "${MODEL:?set MODEL}"
: "${ARTIFACT:?set ARTIFACT}"
: "${RUN_ID:?set one shared RUN_ID for every arm}"
: "${SERVER_BIN:?set SERVER_BIN}"
: "${SERVER_LOG:?set SERVER_LOG}"
: "${BW24_SPILL_IO:?set BW24_SPILL_IO}"
: "${BW24_SPILL_PREAD_DEPTH:?set BW24_SPILL_PREAD_DEPTH}"
: "${BW24_SPILL_STATS:?set BW24_SPILL_STATS}"
: "${BW24_SERVE_SPEC:?set BW24_SERVE_SPEC}"

OUT_ROOT=${OUT_ROOT:-$HERE/results-hourish}
CACHE_DIR=${CACHE_DIR:-$HERE/.cache}
BASE_URL=${BASE_URL:-http://127.0.0.1:8080/v1/completions}
NUM_CONCURRENT=${NUM_CONCURRENT:-1}

[[ -f "$PANEL_LOCK" ]] || { echo "missing panel lock: $PANEL_LOCK" >&2; exit 2; }
actual_panel_sha=$(sha256sum "$PANEL_LOCK" | cut -d' ' -f1)
[[ "$actual_panel_sha" == "$PANEL_SHA256" ]] || {
  echo "panel lock hash mismatch: expected $PANEL_SHA256, got $actual_panel_sha" >&2
  exit 2
}
[[ "$NUM_CONCURRENT" == 1 ]] || {
  echo "the matched hourish panel requires NUM_CONCURRENT=1" >&2
  exit 2
}

mapfile -t TASK_ROWS < <(python3 - "$PANEL_LOCK" <<'PY'
import json, sys

panel = json.load(open(sys.argv[1]))
expected = [
    "humaneval_instruct",
    "mbpp_instruct",
    "hendrycks_math500",
    "mmlu_pro_history",
    "mmlu_pro_other",
]
if panel.get("format") != "bw24-hourish-eval-panel-v1":
    raise SystemExit("unexpected panel format")
if set(panel.get("task_counts", {})) != set(expected):
    raise SystemExit("panel task set differs from the precommitted set")
for task in expected:
    indices = panel["samples"][task]
    if len(indices) != panel["task_counts"][task]:
        raise SystemExit(f"sample count mismatch for {task}")
    print("\t".join((
        task,
        json.dumps({task: indices}, sort_keys=True, separators=(",", ":")),
        str(panel["max_gen_toks"][task]),
    )))
PY
)

shard_complete() {
  local task=$1
  local dir="$OUT_ROOT/$ARM/$RUN_ID/shards/$task"
  [[ -d "$dir" ]] || return 1
  python3 - "$dir" "$task" "$PANEL_SHA256" <<'PY'
import json, pathlib, sys

run_dir, task, panel_sha = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
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
  timeout_s=2700
  if [[ "$task" == humaneval_instruct || "$task" == mbpp_instruct ]]; then
    suite=code
    predict_only=1
    unsafe=1
    timeout_s=3600
  fi
  echo "starting hourish shard: arm=$ARM task=$task samples=$samples_json"
  ARM="$ARM" MODEL="$MODEL" ARTIFACT="$ARTIFACT" \
    SERVER_BIN="$SERVER_BIN" SERVER_LOG="$SERVER_LOG" \
    BW24_SPILL_IO="$BW24_SPILL_IO" \
    BW24_SPILL_PREAD_DEPTH="$BW24_SPILL_PREAD_DEPTH" \
    BW24_SPILL_STATS="$BW24_SPILL_STATS" BW24_SERVE_SPEC="$BW24_SERVE_SPEC" \
    BASE_URL="$BASE_URL" OUT_ROOT="$OUT_ROOT" CACHE_DIR="$CACHE_DIR" RUN_ID="$RUN_ID" \
    SUITE="$suite" TASKS_OVERRIDE="$task" SHARD_ID="$task" \
    SAMPLES_JSON="$samples_json" PANEL_LOCK="$PANEL_LOCK" \
    MAX_GEN_TOKS="$max_gen_toks" NUM_CONCURRENT=1 EVAL_TIMEOUT_S="$timeout_s" \
    PREDICT_ONLY="$predict_only" BW24_UNSAFE_EVALS="$unsafe" \
    "$HERE/run_public_evals.sh"
done

echo "all hourish generation shards complete: $ARM/$RUN_ID"
