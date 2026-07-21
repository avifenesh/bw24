#!/usr/bin/env bash
set -euo pipefail

ROOT=${BW24_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
HERE="$ROOT/research/per-expert-quant"
PANEL_LOCK=${PANEL_LOCK:-$HERE/expanded-capability-panel.lock.json}
OUT_ROOT=${OUT_ROOT:-$HERE/results-hourish}
CACHE_DIR=${CACHE_DIR:-$HERE/.cache}
HF_HOME_DIR=${HF_HOME_DIR:-${HF_HOME:-$HOME/.cache/huggingface}}
GPU_IDS=${GPU_IDS:-0,1,2,3,4,5,6,7}
PORT_BASE=${PORT_BASE:-8080}
SPILL_DEPTH=${BW24_SPILL_PREAD_DEPTH:-8}
HEALTH_TIMEOUT_S=${HEALTH_TIMEOUT_S:-900}
SHARD_TIMEOUT_S=${HOURISH_SHARD_TIMEOUT_S:-43200}
SCORER_LOCK=${HOURISH_SCORER_LOCK:-/tmp/bw24-hourish-scorer.lock}

: "${ARM:?set ARM}"
: "${ARTIFACT:?set ARTIFACT}"
: "${RUN_ID:?set RUN_ID}"
: "${SERVER_BIN:?set SERVER_BIN}"

die() { echo "error: $*" >&2; exit 2; }

[[ "$ARM" =~ ^[A-Za-z0-9._-]+$ ]] || die "invalid ARM"
[[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || die "invalid RUN_ID"
[[ -f "$ARTIFACT/manifest.json" && -x "$SERVER_BIN" && -f "$PANEL_LOCK" \
  && -d "$HF_HOME_DIR/hub" && -d "$HF_HOME_DIR/datasets" ]] \
  || die "missing artifact, server, panel lock, or Hugging Face cache"
[[ "$PORT_BASE" =~ ^[1-9][0-9]{0,4}$ ]] || die "invalid PORT_BASE"
[[ "$SPILL_DEPTH" =~ ^([1-9]|[1-5][0-9]|6[0-4])$ ]] || die "invalid spill depth"
[[ "$HEALTH_TIMEOUT_S" =~ ^[1-9][0-9]*$ && "$SHARD_TIMEOUT_S" =~ ^[1-9][0-9]*$ ]] \
  || die "invalid timeout"

PANEL_SHA=$(python3 "$HERE/validate_capability_panel.py" "$PANEL_LOCK" \
  --suite-lock "$HERE/suite.lock.json" --print-sha)
PANEL_FORMAT=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["format"])' \
  "$PANEL_LOCK")
[[ "$PANEL_FORMAT" == bw24-expanded-capability-panel-v1 ]] \
  || die "parallel ports require the expanded capability panel"

IFS=, read -r -a GPUS <<< "$GPU_IDS"
((${#GPUS[@]} > 0)) || die "GPU_IDS is empty"
declare -A SEEN_GPUS=()
for gpu in "${GPUS[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ && -z ${SEEN_GPUS[$gpu]+x} ]] || die "invalid or duplicate GPU id"
  SEEN_GPUS[$gpu]=1
done

RUN_DIR="$OUT_ROOT/$ARM/$RUN_ID"
CONTROL_DIR="$OUT_ROOT/_control/$RUN_ID/$ARM"
[[ ! -e "$CONTROL_DIR/complete" ]] || die "$ARM is already complete"
mkdir -p "$CONTROL_DIR" "$RUN_DIR"
ATTEMPT="$CONTROL_DIR/parallel-attempt-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir "$ATTEMPT"

shard_complete() {
  local task=$1 dir="$RUN_DIR/shards/$1"
  python3 - "$dir" "$task" "$PANEL_SHA" <<'PY'
import json, pathlib, re, sys

root = pathlib.Path(sys.argv[1])
task, panel_sha = sys.argv[2:]
path = root / "run-metadata.json"
if not path.is_file():
    raise SystemExit(1)
receipt = json.load(open(path))
base_url = receipt.get("base_url")
if not (
    receipt.get("completed_successfully") is True
    and receipt.get("evaluator_exit_code") == 0
    and receipt.get("tee_exit_code") == 0
    and receipt.get("tasks") == [task]
    and receipt.get("panel_lock_sha256") == panel_sha
    and isinstance(base_url, str)
    and re.fullmatch(r"http://127[.]0[.]0[.]1:[1-9][0-9]{0,4}/v1/completions", base_url)
    and list(root.glob("**/results_*.json"))
    and list(root.glob(f"**/samples_{task}_*.jsonl"))
):
    raise SystemExit(1)
PY
}

mapfile -t TASK_ROWS < <(python3 "$HERE/validate_capability_panel.py" "$PANEL_LOCK" \
  --suite-lock "$HERE/suite.lock.json" --task-rows)
declare -a REMAINING=()
for row in "${TASK_ROWS[@]}"; do
  IFS=$'\t' read -r task _ _ <<< "$row"
  shard_complete "$task" || REMAINING+=("$row")
done
((${#REMAINING[@]} <= ${#GPUS[@]})) \
  || die "${#REMAINING[@]} incomplete shards exceed ${#GPUS[@]} GPU lanes"

declare -a WORKER_PIDS=()
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  for pid in "${WORKER_PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  wait 2>/dev/null || true
  printf '%s\n' "$status" > "$ATTEMPT/exit-code"
  exit "$status"
}
trap cleanup EXIT INT TERM

run_shard() (
  local row=$1 gpu=$2 lane=$3 task samples max_tokens port cpu_start cpu_end suite unsafe
  IFS=$'\t' read -r task samples max_tokens <<< "$row"
  port=$((PORT_BASE + lane))
  cpu_start=$((gpu * 12))
  cpu_end=$((cpu_start + 11))
  local lane_dir="$ATTEMPT/$task" server_log="$ATTEMPT/$task/server.log" server_pid=""
  mkdir "$lane_dir"
  stop_server() {
    [[ -n "$server_pid" ]] || return 0
    kill "$server_pid" 2>/dev/null || true
    for _ in {1..100}; do kill -0 "$server_pid" 2>/dev/null || break; sleep 0.1; done
    kill -KILL "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  }
  trap stop_server EXIT INT TERM

  env -u BW24_API_KEY -u BW24_FULL_PREC -u BW24_MOE_GATE -u BW24_MOE_RESIDENT_GB \
    -u BW24_MOE_SLOTS -u BW24_MOE_STATS -u BW24_MOE_TRACE \
    CUDA_VISIBLE_DEVICES="$gpu" BW24_COMPAT=openai BW24_SERVE_SPEC=0 BW24_KV_REUSE=0 \
    BW24_CTX=8192 BW24_FAST=1 BW24_MMVQ=1 BW24_MOE_CACHE=1 BW24_MOE_GROUPED=1 \
    BW24_MOE_PREWARM=1 BW24_MOE_PREFETCH=1 BW24_MOE_PAGE_PREFETCH=1 \
    BW24_MOE_PAGE_PREFETCH_WINDOW=8 BW24_MOE_MMAP_ADVICE=normal BW24_MOE_RESIDENT=1 \
    BW24_MOE_VRAM_FRAC=0.85 BW24_SPILL_IO=worker BW24_SPILL_PREAD_DEPTH="$SPILL_DEPTH" \
    BW24_SPILL_STATS=1 BW24_MODELS="$ARM=$ARTIFACT" BW24_ADDR="127.0.0.1:$port" \
    taskset -c "$cpu_start-$cpu_end" "$SERVER_BIN" >"$server_log" 2>&1 &
  server_pid=$!
  printf '%s\n' "$server_pid" > "$lane_dir/server.pid"

  local deadline=$((SECONDS + HEALTH_TIMEOUT_S))
  while ((SECONDS < deadline)); do
    kill -0 "$server_pid" 2>/dev/null || die "$task server exited before health"
    if curl -fsS --max-time 5 "http://127.0.0.1:$port/health" >"$lane_dir/health.json.tmp" 2>/dev/null \
      && python3 "$HERE/validate_server_health.py" "$lane_dir/health.json.tmp" "$ARM" --exact
    then mv "$lane_dir/health.json.tmp" "$lane_dir/health.json"; break; fi
    sleep 1
  done
  [[ -f "$lane_dir/health.json" ]] || die "$task server health timeout"

  suite=candidate
  unsafe=0
  if [[ "$task" == humaneval_instruct ]]; then suite=code; unsafe=1; fi
  env HF_HOME="$HF_HOME_DIR" HF_HUB_CACHE="$HF_HOME_DIR/hub" \
    HF_DATASETS_CACHE="$HF_HOME_DIR/datasets" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    ARM="$ARM" MODEL="$ARM" ARTIFACT="$ARTIFACT" \
    SERVER_BIN="$SERVER_BIN" SERVER_LOG="$server_log" RUNTIME_KIND=bw24 \
    BW24_SPILL_IO=worker BW24_SPILL_PREAD_DEPTH="$SPILL_DEPTH" BW24_SPILL_STATS=1 \
    BW24_SERVE_SPEC=0 BASE_URL="http://127.0.0.1:$port/v1/completions" \
    OUT_ROOT="$OUT_ROOT" CACHE_DIR="$CACHE_DIR" RUN_ID="$RUN_ID" SUITE="$suite" \
    TASKS_OVERRIDE="$task" SHARD_ID="$task" SAMPLES_JSON="$samples" PANEL_LOCK="$PANEL_LOCK" \
    MAX_GEN_TOKS="$max_tokens" NUM_CONCURRENT=1 EVAL_TIMEOUT_S="$SHARD_TIMEOUT_S" \
    PREDICT_ONLY=$([[ "$task" == humaneval_instruct ]] && echo 1 || echo 0) \
    BW24_UNSAFE_EVALS="$unsafe" "$HERE/run_public_evals.sh"
)

for index in "${!REMAINING[@]}"; do
  run_shard "${REMAINING[$index]}" "${GPUS[$index]}" "$index" \
    >"$ATTEMPT/lane-$index.log" 2>&1 &
  WORKER_PIDS+=("$!")
done

failed=0
for pid in "${WORKER_PIDS[@]}"; do wait "$pid" || failed=1; done
((failed == 0)) || die "one or more capability shards failed"
WORKER_PIDS=()

(
  flock --exclusive --timeout 43200 9 || die "timed out waiting for scorer lock"
  if [[ ! -e "$RUN_DIR/code-score.json" && ! -e "$RUN_DIR/code-score.receipt.json" ]]; then
    PANEL_LOCK="$PANEL_LOCK" timeout --signal=TERM --kill-after=60s 7200s \
      "$HERE/score_hourish_code_container.sh" "$RUN_DIR"
  fi
  if [[ ! -e "$RUN_DIR/math-score.json" && ! -e "$RUN_DIR/math-score.receipt.json" ]]; then
    PANEL_LOCK="$PANEL_LOCK" timeout --signal=TERM --kill-after=60s 7200s \
      "$HERE/score_hourish_math_container.sh" "$RUN_DIR"
  fi
) 9>"$SCORER_LOCK"

[[ -f "$RUN_DIR/code-score.json" && -f "$RUN_DIR/code-score.receipt.json" \
  && -f "$RUN_DIR/math-score.json" && -f "$RUN_DIR/math-score.receipt.json" ]] \
  || die "score evidence is incomplete"

SUMMARY="$RUN_DIR/strict-summary.json"
if [[ ! -e "$SUMMARY" ]]; then
  python3 "$HERE/summarize_hourish_results.py" --out-root "$OUT_ROOT" --run-id "$RUN_ID" \
    --arms "$ARM" --baseline "$ARM" --panel-lock "$PANEL_LOCK" \
    --suite-lock "$HERE/suite.lock.json" --server-sha256 "$(sha256sum "$SERVER_BIN" | awk '{print $1}')" \
    --output "$SUMMARY"
fi
date -u +%FT%TZ > "$CONTROL_DIR/complete"
date -u +%FT%TZ > "$ATTEMPT/complete"
trap - EXIT INT TERM
printf '0\n' > "$ATTEMPT/exit-code"
echo "parallel hourish arm complete: $ARM/$RUN_ID"
