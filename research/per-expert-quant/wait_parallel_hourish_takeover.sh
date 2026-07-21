#!/usr/bin/env bash
set -euo pipefail

ROOT=${BW24_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
HERE="$ROOT/research/per-expert-quant"
OUT_ROOT=${OUT_ROOT:-$HERE/results-hourish}
WAIT_TASK=${WAIT_TASK:-humaneval_instruct}
PAUSED_PIDS=${PAUSED_PIDS:-}
LEGACY_SERVER_PID=${LEGACY_SERVER_PID:-}
POLL_SECONDS=${POLL_SECONDS:-5}
PARALLEL_RUNNER=${PARALLEL_RUNNER:-$HERE/run_parallel_hourish_remaining.sh}

: "${ARM:?set ARM}"
: "${RUN_ID:?set RUN_ID}"
: "${ARTIFACT:?set ARTIFACT}"
: "${SERVER_BIN:?set SERVER_BIN}"

die() { echo "error: $*" >&2; exit 2; }

[[ -x "$PARALLEL_RUNNER" ]] || die "missing parallel runner"
[[ "$PAUSED_PIDS" =~ ^[0-9]+(,[0-9]+)*$ ]] || die "set comma-separated PAUSED_PIDS"
[[ "$LEGACY_SERVER_PID" =~ ^[0-9]+$ ]] || die "set LEGACY_SERVER_PID"
[[ "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]] || die "invalid poll interval"

SHARD="$OUT_ROOT/$ARM/$RUN_ID/shards/$WAIT_TASK"
while ! python3 - "$SHARD" "$WAIT_TASK" <<'PY'
import json, pathlib, sys

root = pathlib.Path(sys.argv[1])
task = sys.argv[2]
receipt = root / "run-metadata.json"
if not receipt.is_file():
    raise SystemExit(1)
data = json.load(open(receipt))
if not (
    data.get("completed_successfully") is True
    and data.get("evaluator_exit_code") == 0
    and data.get("tee_exit_code") == 0
    and data.get("tasks") == [task]
    and list(root.glob("**/results_*.json"))
    and list(root.glob(f"**/samples_{task}_*.jsonl"))
):
    raise SystemExit(1)
PY
do
  sleep "$POLL_SECONDS"
done

IFS=, read -r -a pids <<< "$PAUSED_PIDS"
kill -TERM "${pids[@]}" 2>/dev/null || true
kill -CONT "${pids[@]}" 2>/dev/null || true
for _ in {1..200}; do
  alive=0
  for pid in "${pids[@]}" "$LEGACY_SERVER_PID"; do
    kill -0 "$pid" 2>/dev/null && alive=1
  done
  ((alive == 0)) && break
  sleep 0.1
done
for pid in "${pids[@]}" "$LEGACY_SERVER_PID"; do
  kill -KILL "$pid" 2>/dev/null || true
done

exec "$PARALLEL_RUNNER"
