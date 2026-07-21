#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_REPO=${EXPERIMENT_REPO:-/data/src/bw24-hy3-110gb-86f80a0}
EVAL_REPO=${EVAL_REPO:-/data/src/bw24-eval-valid-c8689ec}
EXPECTED_EVAL_COMMIT=${EXPECTED_EVAL_COMMIT:-c8689ec937c899fbdbd399432ec175e7d48b53ae}
ROOT=${ROOT:-/data/experiments/hy3-110gb}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-$ROOT/artifacts/matched-repack}
REPACK_COMPLETE=${REPACK_COMPLETE:-$ROOT/logs/matched-repack/complete}
SERVER_RECEIPT=${SERVER_RECEIPT:-$ROOT/receipts/bw24-server-build.json}
PY=${PY:-/data/venvs/hy3-110gb/bin/python}
UV_BIN=${UV_BIN:-/data/venvs/hy3-110gb/bin/uv}
BUCKET=${BUCKET:?set BUCKET}
PARENT_RUN_ID=${PARENT_RUN_ID:?set PARENT_RUN_ID}
RUN_ID=${RUN_ID:-coverage110-screen-v1-$(date -u +%Y%m%dT%H%M%SZ)}
BASE_ARM=${BASE_ARM:-layer100-matched}
CANDIDATE_ARM=${CANDIDATE_ARM:-layer110-delta-restore}
EVAL_BUDGET_SECONDS=${EVAL_BUDGET_SECONDS:-3600}
SHARD_TIMEOUT_SECONDS=${SHARD_TIMEOUT_SECONDS:-3300}

OUT_ROOT=$ROOT/results/bounded-screen
RUN_ROOT=$ROOT/logs/bounded-screen/$RUN_ID
PRIVATE_ROOT=$ROOT/evidence/private-gate-$RUN_ID
PANEL_LOCK=$EVAL_REPO/research/per-expert-quant/expanded-capability-panel.lock.json
SUITE_LOCK=$EVAL_REPO/research/per-expert-quant/suite.lock.json
CACHE_DIR=$ROOT/cache/eval
mkdir -p "$RUN_ROOT" "$OUT_ROOT" "$CACHE_DIR"
exec 9>"$RUN_ROOT/transition.lock"
flock -n 9 || { echo "bounded screen already owns $RUN_ID" >&2; exit 1; }
exec > >(tee -a "$RUN_ROOT/transition.log") 2>&1

[[ $(git -C "$EVAL_REPO" rev-parse HEAD) == "$EXPECTED_EVAL_COMMIT" ]]
[[ -z $(git -C "$EVAL_REPO" status --porcelain) ]]
while [[ ! -f "$REPACK_COMPLETE" || ! -f "$SERVER_RECEIPT" ]]; do sleep 20; done
server_bin=$(jq -r .binary "$SERVER_RECEIPT")
server_sha=$(jq -r .binary_sha256 "$SERVER_RECEIPT")
[[ -x "$server_bin" && $(sha256sum "$server_bin" | awk '{print $1}') == "$server_sha" ]]
for arm in "$BASE_ARM" "$CANDIDATE_ARM"; do
  "$PY" "$EVAL_REPO/research/per-expert-quant/validate_artifact.py" \
    "$ARTIFACT_ROOT/$arm" --verify-sources
done
"$PY" "$EVAL_REPO/research/per-expert-quant/validate_capability_panel.py" \
  "$PANEL_LOCK" --suite-lock "$SUITE_LOCK" --print-sha

# The private gate is a health/routing prerequisite only.  It does not produce a capability score.
SERVER_BIN="$server_bin" ART_ROOT="$ARTIFACT_ROOT" \
REQUESTS="$ROOT/calibration/private/requests.jsonl" OUT_ROOT="$PRIVATE_ROOT" PY="$PY" \
CAPTURE_TOOL="$EXPERIMENT_REPO/research/per-expert-quant/capture_calibration.py" \
HEALTH_TOOL="$EVAL_REPO/research/per-expert-quant/validate_server_health.py" \
ROUTE_VALIDATOR="$EVAL_REPO/research/per-expert-quant/validate_pruned_route_trace.py" \
ARMS_CSV="$BASE_ARM,$CANDIDATE_ARM" GPU_BASE=0 PORT_BASE=8170 CPU_BASE=0 CPUS_PER_LANE=24 \
  "$EVAL_REPO/research/per-expert-quant/run_100gb_private_artifact_gate.sh"

# Prepare the exact frozen harness before the one-hour candidate clock starts.
harness_commit=$(jq -r .lm_eval_commit "$SUITE_LOCK")
harness_dir="$CACHE_DIR/lm-eval-${harness_commit:0:12}"
if [[ ! -d "$harness_dir/.git" ]]; then
  git init --quiet "$harness_dir"
  git -C "$harness_dir" remote add origin https://github.com/EleutherAI/lm-evaluation-harness.git
fi
if [[ $(git -C "$harness_dir" rev-parse HEAD 2>/dev/null || true) != "$harness_commit" ]]; then
  git -C "$harness_dir" fetch --quiet --depth=1 origin "$harness_commit"
  git -C "$harness_dir" checkout --quiet --detach FETCH_HEAD
fi
"$PY" "$EVAL_REPO/research/per-expert-quant/prepare_harness.py" "$harness_dir" --lock "$SUITE_LOCK"
if [[ ! -x "$harness_dir/.venv/bin/python" ]]; then
  "$UV_BIN" venv --python 3.12 "$harness_dir/.venv"
  "$UV_BIN" pip install --python "$harness_dir/.venv/bin/python" -e "$harness_dir[api,ifeval]"
fi

# Make the irreplaceable quant artifacts durable before starting the bounded
# candidate clock.  The pinned upstream source remains reproducible by revision.
aws s3 sync "$ARTIFACT_ROOT" \
  "s3://$BUCKET/runs/$PARENT_RUN_ID/artifacts/matched-repack" --only-show-errors

tasks=(mmlu_pro_history mmlu_pro_other hendrycks_math500 humaneval_instruct)
server_pids=()
eval_pids=()
cleanup() {
  for pid in "${eval_pids[@]:-}"; do kill -- "-$pid" 2>/dev/null || true; done
  for pid in "${server_pids[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  for pid in "${server_pids[@]:-}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT

for arm_index in 0 1; do
  arm=$BASE_ARM; ((arm_index == 0)) || arm=$CANDIDATE_ARM
  for task_index in "${!tasks[@]}"; do
    task=${tasks[$task_index]}
    gpu=$((arm_index * 4 + task_index))
    port=$((8200 + gpu))
    model="$arm-$task"
    log="$RUN_ROOT/$model.server.log"
    cpu_start=$((gpu * 24)); cpu_end=$((cpu_start + 23))
    numa=0; ((gpu >= 4)) && numa=1
    taskset -c "$cpu_start-$cpu_end" numactl --membind="$numa" env \
      -u BW24_API_KEY -u BW24_FULL_PREC \
      CUDA_VISIBLE_DEVICES="$gpu" BW24_COMPAT=openai BW24_SERVE_SPEC=0 BW24_KV_REUSE=0 \
      BW24_CTX=32768 BW24_FAST=1 BW24_MMVQ=1 BW24_MOE_CACHE=1 BW24_MOE_GROUPED=1 \
      BW24_MOE_PREWARM=1 BW24_MOE_PREFETCH=1 BW24_MOE_PAGE_PREFETCH=1 \
      BW24_MOE_PAGE_PREFETCH_WINDOW=8 BW24_MOE_MMAP_ADVICE=normal \
      BW24_MOE_RESIDENT=1 BW24_MOE_VRAM_FRAC=0.92 BW24_SPILL_IO=worker \
      BW24_SPILL_PREAD_DEPTH=8 BW24_SPILL_STATS=1 \
      BW24_MODELS="$model=$ARTIFACT_ROOT/$arm" BW24_ADDR="127.0.0.1:$port" \
      "$server_bin" >"$log" 2>&1 &
    server_pids+=("$!")
  done
done

for gpu in $(seq 0 7); do
  arm_index=$((gpu / 4)); task_index=$((gpu % 4))
  arm=$BASE_ARM; ((arm_index == 0)) || arm=$CANDIDATE_ARM
  task=${tasks[$task_index]}; model="$arm-$task"; port=$((8200 + gpu))
  for _ in $(seq 1 1200); do
    kill -0 "${server_pids[$gpu]}" 2>/dev/null || {
      tail -100 "$RUN_ROOT/$model.server.log"; exit 4;
    }
    if curl -fsS --max-time 5 "http://127.0.0.1:$port/health" >"$RUN_ROOT/$model.health.tmp" 2>/dev/null \
      && "$PY" "$EVAL_REPO/research/per-expert-quant/validate_server_health.py" \
        "$RUN_ROOT/$model.health.tmp" "$model" --exact; then
      mv "$RUN_ROOT/$model.health.tmp" "$RUN_ROOT/$model.health.json"
      break
    fi
    sleep 1
  done
  [[ -f "$RUN_ROOT/$model.health.json" ]]
done

screen_started=$(date +%s)
for arm_index in 0 1; do
  arm=$BASE_ARM; ((arm_index == 0)) || arm=$CANDIDATE_ARM
  for task_index in "${!tasks[@]}"; do
    task=${tasks[$task_index]}; gpu=$((arm_index * 4 + task_index)); port=$((8200 + gpu))
    model="$arm-$task"; suite=candidate; predict_only=0; unsafe=0
    if [[ "$task" == humaneval_instruct ]]; then
      # Generate only.  The frozen scorer executes completions later inside the
      # existing network-disabled, read-only, capability-dropped container.
      suite=code
      predict_only=1
      unsafe=1
    fi
    samples=$(jq -c --arg task "$task" '{($task): .samples[$task]}' "$PANEL_LOCK")
    max_gen=$(jq -r --arg task "$task" '.max_gen_toks[$task]' "$PANEL_LOCK")
    log="$RUN_ROOT/$model.eval.log"
    setsid env \
      ARM="$arm" MODEL="$model" ARTIFACT="$ARTIFACT_ROOT/$arm" \
      SERVER_BIN="$server_bin" SERVER_LOG="$RUN_ROOT/$model.server.log" \
      BASE_URL="http://127.0.0.1:$port/v1/completions" SUITE="$suite" \
      TASKS_OVERRIDE="$task" SHARD_ID="$task" SAMPLES_JSON="$samples" \
      MAX_GEN_TOKS="$max_gen" PANEL_LOCK="$PANEL_LOCK" OUT_ROOT="$OUT_ROOT" \
      CACHE_DIR="$CACHE_DIR" RUN_ID="$RUN_ID" EVAL_TIMEOUT_S="$SHARD_TIMEOUT_SECONDS" \
      NUM_CONCURRENT=1 BW24_SPILL_IO=worker BW24_SPILL_PREAD_DEPTH=8 \
      BW24_SPILL_STATS=1 BW24_SERVE_SPEC=0 PREDICT_ONLY="$predict_only" \
      BW24_UNSAFE_EVALS="$unsafe" \
      UV_BIN="$UV_BIN" \
      "$EVAL_REPO/research/per-expert-quant/run_public_evals.sh" >"$log" 2>&1 &
    eval_pids+=("$!")
  done
done

timed_out=0
while true; do
  alive=0
  for pid in "${eval_pids[@]}"; do kill -0 "$pid" 2>/dev/null && alive=1; done
  ((alive == 0)) && break
  if (( $(date +%s) - screen_started >= EVAL_BUDGET_SECONDS )); then
    timed_out=1
    for pid in "${eval_pids[@]}"; do kill -- "-$pid" 2>/dev/null || true; done
    break
  fi
  sleep 5
done
failure=$timed_out
for pid in "${eval_pids[@]}"; do wait "$pid" || failure=1; done
generation_elapsed=$(($(date +%s) - screen_started))
cleanup
trap - EXIT

((failure == 0)) || {
  aws s3 sync "$ROOT" "s3://$BUCKET/runs/$PARENT_RUN_ID" --only-show-errors \
    --exclude 'source/*' --exclude 'tmp/*' --exclude 'cache/*'
  echo "bounded screen incomplete failure=$failure timed_out=$timed_out elapsed=$generation_elapsed" >&2
  exit 5
}

for arm in "$BASE_ARM" "$CANDIDATE_ARM"; do
  for task in "${tasks[@]}"; do
    metadata="$OUT_ROOT/$arm/$RUN_ID/shards/$task/run-metadata.json"
    [[ -f "$metadata" ]]
    jq -e '.completed_successfully == true' "$metadata" >/dev/null
  done
done

score_arm() {
  local arm=$1 run_dir remaining code_pid math_pid failed=0
  run_dir="$OUT_ROOT/$arm/$RUN_ID"
  remaining=$((EVAL_BUDGET_SECONDS - ($(date +%s) - screen_started)))
  ((remaining > 0)) || { echo "no bounded-screen time remains for scoring" >&2; return 1; }
  PANEL_LOCK="$PANEL_LOCK" timeout --signal=TERM --kill-after=60s "${remaining}s" \
    "$EVAL_REPO/research/per-expert-quant/score_hourish_code_container.sh" "$run_dir" &
  code_pid=$!
  PANEL_LOCK="$PANEL_LOCK" timeout --signal=TERM --kill-after=60s "${remaining}s" \
    "$EVAL_REPO/research/per-expert-quant/score_hourish_math_container.sh" "$run_dir" &
  math_pid=$!
  wait "$code_pid" || failed=1
  wait "$math_pid" || failed=1
  ((failed == 0))
}
score_arm "$BASE_ARM"
score_arm "$CANDIDATE_ARM"
eval_elapsed=$(($(date +%s) - screen_started))
((eval_elapsed <= EVAL_BUDGET_SECONDS)) || {
  echo "bounded screen exceeded total budget: $eval_elapsed > $EVAL_BUDGET_SECONDS" >&2
  exit 5
}
"$PY" "$EVAL_REPO/research/per-expert-quant/summarize_hourish_results.py" \
  --out-root "$OUT_ROOT" --run-id "$RUN_ID" \
  --arms "$BASE_ARM,$CANDIDATE_ARM" --baseline "$BASE_ARM" \
  --panel-lock "$PANEL_LOCK" --suite-lock "$SUITE_LOCK" \
  --server-sha256 "$server_sha" --output "$RUN_ROOT/summary.json"
"$PY" - "$ROOT/receipts/bounded-screen-$RUN_ID.json" "$RUN_ID" "$eval_elapsed" \
  "$RUN_ROOT/summary.json" "$server_sha" "$EVAL_BUDGET_SECONDS" <<'PY'
import hashlib
import json
import pathlib
import sys
from datetime import datetime, timezone

output, run_id, elapsed, summary, server_sha, budget = sys.argv[1:]
summary_path = pathlib.Path(summary)
pathlib.Path(output).write_text(json.dumps({
    "format": "bw24-hy3-110gb-bounded-screen-v1",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "run_id": run_id,
    "candidate_wall_clock_budget_seconds": int(budget),
    "eval_elapsed_seconds": int(elapsed),
    "summary": str(summary_path),
    "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
    "server_binary_sha256": server_sha,
    "public_results_used_for_construction_or_healing": False,
}, indent=2, sort_keys=True) + "\n")
PY
date -u +%Y-%m-%dT%H:%M:%SZ | tee "$RUN_ROOT/complete"
aws s3 sync "$ROOT" "s3://$BUCKET/runs/$PARENT_RUN_ID" --only-show-errors \
  --exclude 'source/*' --exclude 'tmp/*' --exclude 'cache/*'
echo "bounded matched screen complete run=$RUN_ID elapsed=$eval_elapsed"
