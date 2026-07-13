#!/usr/bin/env bash
set -euo pipefail

ROOT=${BW24_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
HERE="$ROOT/research/per-expert-quant"
PRACTICAL_ROOT=${PRACTICAL_ROOT:-/data/results/per-expert-quant/practical-v1}
PRACTICAL_READY=${PRACTICAL_READY:-/data/logs/practical-v1/complete}
OUT_ROOT=${OUT_ROOT:-/data/results/per-expert-quant/trusted-full-v1}
LOG_ROOT=${LOG_ROOT:-/data/logs/trusted-full-v1}
SERVER_BIN=${SERVER_BIN:-/data/build/bw24-portable-ada-fix-target/release/bw24-server}
FULL_SUMMARIZER=${FULL_SUMMARIZER:-$HERE/summarize_promoted_results.py}
CACHE_DIR=${CACHE_DIR:-/data/cache/per-expert-evals}
HF_HOME=${HF_HOME:-/data/cache/huggingface}
DATASET_SNAPSHOTTER=${DATASET_SNAPSHOTTER:-$HERE/snapshot_trusted_dataset_cache.py}
SPILL_DEPTH=${SPILL_DEPTH:-8}
IQ4_ART_ROOT=${IQ4_ART_ROOT:-/scratch/bw24-artifacts-iq3-iq4-q4-99f3dc3}
CENTERED_ART_ROOT=${CENTERED_ART_ROOT:-/scratch/bw24-artifacts-iq3-iq4-q4-centered-0f98d7d}
PARETO_ART_ROOT=${PARETO_ART_ROOT:-/scratch/bw24-artifacts-iq3-iq4-q4-pareto-6c5c5ea}
VRAM_FRAC=${VRAM_FRAC:-0.75}
WAIT_INTERVAL_S=${WAIT_INTERVAL_S:-30}
SERVER_HEALTH_TIMEOUT_S=${SERVER_HEALTH_TIMEOUT_S:-1800}
EVAL_TIMEOUT_S=${EVAL_TIMEOUT_S:-604800}
TASK_ATTEMPTS=${TASK_ATTEMPTS:-3}
# MMLU-Pro's pinned task contract uses 2048; apply the same ceiling to GPQA/MATH so
# strict final-answer extraction cannot be turned into a 256-token truncation test.
TRUSTED_MAX_GEN_TOKS=2048

TASKS=(
  gpqa_diamond_cot_zeroshot
  hendrycks_math500
  mmlu_pro_history
  mmlu_pro_other
  mmlu_pro_economics
  mmlu_pro_law
  mmlu_pro_psychology
)

die() { echo "error: $*" >&2; exit 2; }

[[ -x "$SERVER_BIN" ]] || die "missing trusted-full server"
[[ -x "$HERE/run_public_evals.sh" ]] || die "missing public-eval runner"
[[ -x "$HERE/score_promoted_math_container.sh" ]] || die "missing trusted MATH scorer"
[[ -f "$FULL_SUMMARIZER" ]] || die "missing trusted-full summarizer"
[[ -f "$DATASET_SNAPSHOTTER" ]] || die "missing trusted dataset snapshotter"
[[ -f "$HERE/suite.lock.json" ]] || die "missing frozen suite lock"
[[ "$SPILL_DEPTH" =~ ^[1-9][0-9]*$ ]] || die "invalid spill depth"
python3 - "$VRAM_FRAC" <<'PY'
import sys
value = float(sys.argv[1])
if not 0.5 <= value < 0.9:
    raise SystemExit("VRAM_FRAC must be in [0.5, 0.9)")
PY
[[ "$EVAL_TIMEOUT_S" =~ ^[1-9][0-9]*$ ]] || die "invalid eval timeout"
[[ "$TASK_ATTEMPTS" =~ ^[1-9][0-9]*$ ]] || die "invalid task attempt count"
command -v sudo >/dev/null || die "sudo is required for isolated loopback namespaces"
command -v ip >/dev/null || die "iproute2 is required for isolated loopback namespaces"

mkdir -p "$LOG_ROOT" "$OUT_ROOT/run-configs"
exec 9>"$LOG_ROOT/transition.lock"
flock -n 9 || exit 0
exec >>"$LOG_ROOT/transition.log" 2>&1
echo "$(date -u +%FT%TZ) trusted-full transition started"

while [[ ! -f "$PRACTICAL_READY" || ! -s "$PRACTICAL_ROOT/_active-run-id" ]]; do
  sleep "$WAIT_INTERVAL_S"
done
PRACTICAL_RUN_ID=$(<"$PRACTICAL_ROOT/_active-run-id")
PRACTICAL_PROMOTION="$PRACTICAL_ROOT/practical-promotion-$PRACTICAL_RUN_ID.json"
while [[ ! -f "$PRACTICAL_PROMOTION" ]]; do sleep "$WAIT_INTERVAL_S"; done

mapfile -t ARMS < <(python3 - "$PRACTICAL_PROMOTION" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
if d.get("format") != "bw24-practical-promotion-v1":
    raise SystemExit("wrong practical promotion format")
arms = d.get("trusted_full_arms")
if not isinstance(arms, list) or not 2 <= len(arms) <= 3 or len(set(arms)) != len(arms):
    raise SystemExit("expected two or three unique trusted-full arms")
if arms[0] != "plain_quant":
    raise SystemExit("plain_quant must remain the trusted-full baseline")
for arm in arms:
    print(arm)
PY
)

artifact_for() {
  case "$1" in
    plain_quant) printf '%s\n' /scratch/bw24-artifacts/plain-quant ;;
    traffic_nvfp4_53_q2_139) printf '%s\n' /scratch/bw24-artifacts/traffic-nvfp4-53-q2-139 ;;
    prune100_unhealed|prune100_router_repair|prune100_joint_heal)
      printf '/scratch/bw24-artifacts-100gb-5f02c37/%s\n' "$1" ;;
    smart100_empirical|smart100_balanced|smart100_rescue)
      printf '/scratch/bw24-artifacts-smart100-2605fde/%s\n' "$1" ;;
    smart100_iq3_iq4_q4_empirical)
      printf '%s/%s\n' "$IQ4_ART_ROOT" "$1" ;;
    smart100_iq3_iq4_q4_centered)
      printf '%s/%s\n' "$CENTERED_ART_ROOT" "$1" ;;
    smart100_iq3_iq4_q4_pareto)
      printf '%s/%s\n' "$PARETO_ART_ROOT" "$1" ;;
    *) die "no frozen artifact mapping for $1" ;;
  esac
}

for arm in "${ARMS[@]}"; do
  artifact=$(artifact_for "$arm")
  [[ -f "$artifact/manifest.json" ]] || die "missing manifest for $arm"
done

RUN_ID="trusted-full-v1-$(date -u +%Y%m%dT%H%M%SZ)"
RUN_CONFIG="$OUT_ROOT/run-configs/$RUN_ID.json"
DATASET_RECEIPT="$OUT_ROOT/run-configs/$RUN_ID.datasets.json"
HARNESS_COMMIT=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["lm_eval_commit"])' "$HERE/suite.lock.json")
HARNESS_PYTHON="$CACHE_DIR/lm-eval-${HARNESS_COMMIT:0:12}/.venv/bin/python"
[[ -x "$HARNESS_PYTHON" ]] || die "missing pinned lm-eval Python for dataset snapshot"
HOME="$HF_HOME" HF_HOME="$HF_HOME" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 nice -n 19 "$HARNESS_PYTHON" "$DATASET_SNAPSHOTTER" \
  --lock "$HERE/suite.lock.json" --output "$DATASET_RECEIPT"
ARM_COUNT=${#ARMS[@]}
BASE_LANES=$((8 / ARM_COUNT))
EXTRA_LANES=$((8 % ARM_COUNT))
export RUN_ID PRACTICAL_PROMOTION SERVER_BIN ROOT HERE ARM_COUNT BASE_LANES EXTRA_LANES VRAM_FRAC DATASET_RECEIPT TRUSTED_MAX_GEN_TOKS \
  IQ4_ART_ROOT CENTERED_ART_ROOT PARETO_ART_ROOT
python3 - "$RUN_CONFIG" "${ARMS[@]}" <<'PY'
import hashlib, json, os, pathlib, subprocess, sys

def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()

arms = sys.argv[2:]
payload = {
    "format": "bw24-trusted-full-run-v1",
    "run_id": os.environ["RUN_ID"],
    "arms": arms,
    "baseline": "plain_quant",
    "tasks": [
        "gpqa_diamond_cot_zeroshot", "hendrycks_math500", "mmlu_pro_history",
        "mmlu_pro_other", "mmlu_pro_economics", "mmlu_pro_law", "mmlu_pro_psychology",
    ],
    "documents_per_arm": 4746,
    "max_gen_toks": int(os.environ["TRUSTED_MAX_GEN_TOKS"]),
    "vram_fraction": float(os.environ["VRAM_FRAC"]),
    "protocol": "all eight GPUs; task-family shards balanced across isolated per-arm server lanes; concurrency one per lane",
    "lane_allocation": {
        arm: int(os.environ["BASE_LANES"]) + (index < int(os.environ["EXTRA_LANES"]))
        for index, arm in enumerate(arms)
    },
    "practical_promotion": {
        "path": os.environ["PRACTICAL_PROMOTION"],
        "sha256": sha(os.environ["PRACTICAL_PROMOTION"]),
    },
    "suite_lock": {"path": str(pathlib.Path(os.environ["HERE"]) / "suite.lock.json"),
                   "sha256": sha(pathlib.Path(os.environ["HERE"]) / "suite.lock.json")},
    "dataset_cache_receipt": {"path": os.environ["DATASET_RECEIPT"],
                              "sha256": sha(os.environ["DATASET_RECEIPT"])},
    "server": {"path": os.environ["SERVER_BIN"], "sha256": sha(os.environ["SERVER_BIN"])},
    "bw24_commit": subprocess.check_output(
        ["git", "-C", os.environ["ROOT"], "rev-parse", "HEAD"], text=True
    ).strip(),
    "artifacts": {},
}
for arm in arms:
    if arm == "plain_quant":
        root = pathlib.Path("/scratch/bw24-artifacts/plain-quant")
    elif arm == "traffic_nvfp4_53_q2_139":
        root = pathlib.Path("/scratch/bw24-artifacts/traffic-nvfp4-53-q2-139")
    elif arm == "smart100_iq3_iq4_q4_empirical":
        root = pathlib.Path(os.environ["IQ4_ART_ROOT"]) / arm
    elif arm == "smart100_iq3_iq4_q4_centered":
        root = pathlib.Path(os.environ["CENTERED_ART_ROOT"]) / arm
    elif arm == "smart100_iq3_iq4_q4_pareto":
        root = pathlib.Path(os.environ["PARETO_ART_ROOT"]) / arm
    elif arm.startswith("smart100_"):
        root = pathlib.Path("/scratch/bw24-artifacts-smart100-2605fde") / arm
    else:
        root = pathlib.Path("/scratch/bw24-artifacts-100gb-5f02c37") / arm
    payload["artifacts"][arm] = {
        "path": str(root), "manifest_sha256": sha(root / "manifest.json")
    }
pathlib.Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
sha256sum "$RUN_CONFIG" > "$RUN_CONFIG.sha256"
printf '%s\n' "$RUN_ID" > "$OUT_ROOT/_active-run-id"

USER_NAME=$(id -un)
USER_PATH=$PATH
declare -a WORKER_PIDS=()
declare -a NAMESPACES=()

cleanup_all() {
  status=$?
  trap - EXIT INT TERM
  for pid in "${WORKER_PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  for ns in "${NAMESPACES[@]:-}"; do
    mapfile -t pids < <(sudo ip netns pids "$ns" 2>/dev/null || true)
    ((${#pids[@]} == 0)) || sudo kill "${pids[@]}" 2>/dev/null || true
    sudo ip netns del "$ns" 2>/dev/null || true
  done
  exit "$status"
}
trap cleanup_all EXIT INT TERM

run_lane() (
  local arm=$1 gpu=$2 cpus=$3 lane=$4 lane_count=$5 artifact ns arm_log server_log
  artifact=$(artifact_for "$arm")
  ns="bw24-full-${RUN_ID//[^A-Za-z0-9]/-}-$gpu"
  arm_log="$LOG_ROOT/$RUN_ID-$arm-lane$lane"
  server_log="$arm_log/server.log"
  mkdir -p "$arm_log"
  sudo ip netns del "$ns" 2>/dev/null || true
  sudo ip netns add "$ns"
  sudo ip -n "$ns" link set lo up
  echo "$ns" > "$arm_log/netns"

  stop_namespace() {
    mapfile -t pids < <(sudo ip netns pids "$ns" 2>/dev/null || true)
    if ((${#pids[@]})); then
      sudo kill "${pids[@]}" 2>/dev/null || true
      for _ in {1..100}; do
        mapfile -t pids < <(sudo ip netns pids "$ns" 2>/dev/null || true)
        ((${#pids[@]} == 0)) && break
        sleep 0.1
      done
      ((${#pids[@]} == 0)) || sudo kill -KILL "${pids[@]}" 2>/dev/null || true
    fi
    sudo ip netns del "$ns" 2>/dev/null || true
  }
  trap stop_namespace EXIT INT TERM

  sudo ip netns exec "$ns" sudo -u "$USER_NAME" env \
    PATH="$USER_PATH" CUDA_VISIBLE_DEVICES="$gpu" \
    BW24_COMPAT=openai BW24_SERVE_SPEC=0 BW24_KV_REUSE=0 BW24_CTX=8192 \
    BW24_FAST=1 BW24_MMVQ=1 BW24_MOE_CACHE=1 BW24_MOE_GROUPED=1 \
    BW24_MOE_PREWARM=1 BW24_MOE_PREFETCH=1 BW24_MOE_PAGE_PREFETCH=1 \
    BW24_MOE_PAGE_PREFETCH_WINDOW=8 BW24_MOE_MMAP_ADVICE=normal \
    BW24_MOE_RESIDENT=1 BW24_MOE_VRAM_FRAC="$VRAM_FRAC" BW24_SPILL_IO=worker \
    BW24_SPILL_PREAD_DEPTH="$SPILL_DEPTH" BW24_SPILL_STATS=1 \
    BW24_MODELS="$arm=$artifact" BW24_ADDR=127.0.0.1:8080 \
    taskset -c "$cpus" "$SERVER_BIN" >"$server_log" 2>&1 &

  local deadline=$((SECONDS + SERVER_HEALTH_TIMEOUT_S))
  while ((SECONDS < deadline)); do
    if sudo ip netns exec "$ns" curl -fsS --max-time 5 http://127.0.0.1:8080/health \
      >"$arm_log/health.json.tmp" 2>/dev/null \
      && python3 "$HERE/validate_server_health.py" "$arm_log/health.json.tmp" "$arm" --exact
    then
      mv "$arm_log/health.json.tmp" "$arm_log/health.json"
      break
    fi
    sleep 1
  done
  [[ -f "$arm_log/health.json" ]] || die "$arm trusted-full server health timeout"

  for task_index in "${!TASKS[@]}"; do
    (( task_index % lane_count == lane )) || continue
    task=${TASKS[$task_index]}
    shard="$OUT_ROOT/$arm/$RUN_ID/shards/$task"
    completed=0
    for task_attempt in $(seq 1 "$TASK_ATTEMPTS"); do
      if [[ -e "$shard" ]]; then
        failed_root="$OUT_ROOT/_failed/$arm/$RUN_ID"
        mkdir -p "$failed_root"
        mv "$shard" "$failed_root/$task-attempt$(date -u +%Y%m%dT%H%M%SZ)-$task_attempt"
      fi
      set +e
      sudo ip netns exec "$ns" sudo -u "$USER_NAME" env \
        HOME="$HF_HOME" HF_HOME="$HF_HOME" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
        TRANSFORMERS_OFFLINE=1 PATH="$USER_PATH" ARM="$arm" MODEL="$arm" \
        ARTIFACT="$artifact" SERVER_BIN="$SERVER_BIN" SERVER_LOG="$server_log" \
        BW24_SPILL_IO=worker BW24_SPILL_PREAD_DEPTH="$SPILL_DEPTH" BW24_SPILL_STATS=1 \
        BW24_SERVE_SPEC=0 BASE_URL=http://127.0.0.1:8080/v1/completions \
        OUT_ROOT="$OUT_ROOT" CACHE_DIR="$CACHE_DIR" RUN_ID="$RUN_ID" \
        SUITE=candidate TASKS_OVERRIDE="$task" SHARD_ID="$task" LIMIT=all \
        MAX_GEN_TOKS="$TRUSTED_MAX_GEN_TOKS" NUM_CONCURRENT=1 EVAL_TIMEOUT_S="$EVAL_TIMEOUT_S" \
        "$HERE/run_public_evals.sh" 2>&1 | tee -a "$arm_log/$task.log"
      statuses=("${PIPESTATUS[@]}")
      set -e
      if (( statuses[0] == 0 && statuses[1] == 0 )); then
        completed=1
        break
      fi
      echo "task process attempt $task_attempt/$TASK_ATTEMPTS failed: arm=$arm task=$task" \
        | tee -a "$arm_log/$task.log"
    done
    (( completed == 1 )) || die "$arm/$task exhausted process attempts"
    if [[ "$task" == hendrycks_math500 ]]; then
      SUITE_LOCK="$HERE/suite.lock.json" \
        "$HERE/score_promoted_math_container.sh" "$OUT_ROOT/$arm/$RUN_ID" \
        | tee "$arm_log/math-score.log"
    fi
  done
  date -u +%FT%TZ > "$arm_log/complete"
)

gpu=0
for arm_index in "${!ARMS[@]}"; do
  lane_count=$BASE_LANES
  (( arm_index < EXTRA_LANES )) && lane_count=$((lane_count + 1))
  for ((lane=0; lane<lane_count; lane++)); do
    ns="bw24-full-${RUN_ID//[^A-Za-z0-9]/-}-$gpu"
    NAMESPACES+=("$ns")
    cpu_start=$((gpu * 12))
    cpu_end=$((cpu_start + 11))
    run_lane "${ARMS[$arm_index]}" "$gpu" "$cpu_start-$cpu_end" "$lane" "$lane_count" &
    WORKER_PIDS+=("$!")
    gpu=$((gpu + 1))
  done
done

failed=0
for pid in "${WORKER_PIDS[@]}"; do wait "$pid" || failed=1; done
((failed == 0)) || die "one or more trusted-full workers failed"

ARMS_CSV=$(IFS=,; echo "${ARMS[*]}")
PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}" python3 "$FULL_SUMMARIZER" \
  --out-root "$OUT_ROOT" --run-id "$RUN_ID" --arms "$ARMS_CSV" \
  --baseline plain_quant --expected-n all \
  --lock "$HERE/suite.lock.json" \
  --out "$OUT_ROOT/_runs/$RUN_ID/trusted-full-results.md"
sha256sum "$RUN_CONFIG" "$PRACTICAL_PROMOTION" "$HERE/suite.lock.json" "$DATASET_RECEIPT" \
  "$OUT_ROOT/_runs/$RUN_ID/trusted-full-results.json" > "$LOG_ROOT/$RUN_ID-evidence.sha256"
date -u +%FT%TZ | tee "$LOG_ROOT/complete"
trap - EXIT INT TERM
