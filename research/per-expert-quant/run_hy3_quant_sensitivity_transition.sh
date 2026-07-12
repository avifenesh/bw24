#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
PY=${PY:-python3}
PRACTICAL_PID_FILE=${PRACTICAL_PID_FILE:-/tmp/bw24-practical-transition.pid}
PRACTICAL_COMPLETE=${PRACTICAL_COMPLETE:-/data/logs/practical-v1/complete}
PRACTICAL_ACTIVE_RUN=${PRACTICAL_ACTIVE_RUN:-/data/results/per-expert-quant/practical-v1/_active-run-id}
PRACTICAL_OUT=${PRACTICAL_OUT:-/data/results/per-expert-quant/practical-v1}
CALIBRATION=${CALIBRATION:-/data/calibration/hy3-100gb-5f02c37}
REQUESTS=${REQUESTS:-/data/calibration/hy3-confidence-v1/requests.jsonl}
SOURCE=${SOURCE:-/opt/dlami/nvme/models/hy3-source}
OUT_ROOT=${OUT_ROOT:-/data/calibration/hy3-quant-sensitivity-53de6ca}
LOG_ROOT=${LOG_ROOT:-/data/logs/hy3-quant-sensitivity-53de6ca}
MAX_TOKENS_PER_EXPERT=${MAX_TOKENS_PER_EXPERT:-16}
EXPECTED_COMMIT=${EXPECTED_COMMIT:-53de6ca88d77da435a067e6b844e029e545d06ef}
SCORER="$ROOT/tools/build_hy3_quant_sensitivity.py"

die() { echo "quant-sensitivity transition: $*" >&2; exit 1; }
mkdir -p "$OUT_ROOT" "$LOG_ROOT"
exec 9>"$LOG_ROOT/transition.lock"
flock -n 9 || die "another transition owns $LOG_ROOT/transition.lock"
echo "$(date -u +%FT%TZ) quant-sensitivity transition started" | tee -a "$LOG_ROOT/transition.log"

[[ -x "$SCORER" || -f "$SCORER" ]] || die "missing scorer $SCORER"
[[ $(git -C "$ROOT" rev-parse HEAD) == "$EXPECTED_COMMIT" ]] \
  || die "source commit is not $EXPECTED_COMMIT"

while [[ -f "$PRACTICAL_PID_FILE" ]] && kill -0 "$(cat "$PRACTICAL_PID_FILE")" 2>/dev/null; do
  sleep 30
done
[[ -f "$PRACTICAL_COMPLETE" ]] || die "practical transition exited without complete marker"
[[ -f "$PRACTICAL_ACTIVE_RUN" ]] || die "missing practical active-run marker"
run_id=$(cat "$PRACTICAL_ACTIVE_RUN")
[[ -f "$PRACTICAL_OUT/practical-promotion-$run_id.json" ]] \
  || die "practical run $run_id has no promotion evidence"

if pgrep -x bw24-server >/dev/null || pgrep -af '/harbor run ' >/dev/null; then
  die "model server or Harbor evaluator still active after practical completion"
fi
if [[ -n $(docker ps -q) ]]; then
  die "task containers still active after practical completion"
fi

for path in \
  "$CALIBRATION/moe-inputs.lock.json" \
  "$CALIBRATION/routes-weighted.trace" \
  "$REQUESTS" \
  "$SOURCE/config.json" \
  "$SOURCE/model.safetensors.index.json"; do
  [[ -f "$path" ]] || die "missing input $path"
done

"$PY" "$SCORER" --self-test | tee "$LOG_ROOT/self-test.log"

layers=(1-10 11-20 21-30 31-40 41-50 51-60 61-70 71-79)
cpus=(0-11 12-23 24-35 36-47 48-59 60-71 72-83 84-95)
pids=()
for gpu in $(seq 0 7); do
  out="$OUT_ROOT/lane-$gpu.json"
  log="$LOG_ROOT/lane-$gpu.log"
  [[ ! -e "$out" ]] || die "refusing existing output $out"
  CUDA_VISIBLE_DEVICES=$gpu taskset -c "${cpus[$gpu]}" nice -n 19 \
    "$PY" "$SCORER" \
      --trace-lock "$CALIBRATION/moe-inputs.lock.json" \
      --weight-trace "$CALIBRATION/routes-weighted.trace" \
      --requests "$REQUESTS" --source-dir "$SOURCE" \
      --layers "${layers[$gpu]}" --device cuda:0 \
      --max-tokens-per-expert "$MAX_TOKENS_PER_EXPERT" \
      --out "$out" >"$log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
((failed == 0)) || die "one or more sensitivity lanes failed"
[[ $(find "$OUT_ROOT" -maxdepth 1 -name 'lane-*.json' | wc -l) -eq 8 ]] \
  || die "expected eight completed lane outputs"
sha256sum "$OUT_ROOT"/lane-*.json "$LOG_ROOT"/lane-*.log > "$OUT_ROOT/evidence.sha256"
date -u +%FT%TZ | tee "$LOG_ROOT/complete"
