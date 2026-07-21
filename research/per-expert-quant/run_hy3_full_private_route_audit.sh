#!/usr/bin/env bash
set -euo pipefail

SERVER_BIN=${SERVER_BIN:?}
ART_ROOT=${ART_ROOT:?}
REQUESTS=${REQUESTS:?}
OUT_ROOT=${OUT_ROOT:?}
PY=${PY:-python3}
CAPTURE_TOOL=${CAPTURE_TOOL:?}
HEALTH_TOOL=${HEALTH_TOOL:?}
ROUTE_VALIDATOR=${ROUTE_VALIDATOR:?}
MERGE_TOOL=${MERGE_TOOL:?}
ANALYSIS_TOOL=${ANALYSIS_TOOL:?}
SELECTION_RECEIPT=${SELECTION_RECEIPT:?}
ARMS_CSV=${ARMS_CSV:-layer100-matched,layer110-delta-restore}
LANES_PER_ARM=${LANES_PER_ARM:-4}
PORT_BASE=${PORT_BASE:-8300}
CPUS_PER_LANE=${CPUS_PER_LANE:-24}
IFS=, read -r -a ARMS <<<"$ARMS_CSV"
(( ${#ARMS[@]} == 2 && LANES_PER_ARM == 4 )) \
  || { echo "full private audit requires two arms and four lanes per arm" >&2; exit 2; }

mkdir -p "$OUT_ROOT"
exec 9>"$OUT_ROOT/audit.lock"
flock -n 9 || exit 0
[[ ! -f "$OUT_ROOT/complete" ]] || exit 0
if find "$OUT_ROOT" -mindepth 1 -maxdepth 1 ! -name audit.lock -print -quit | grep -q .; then
  echo "refusing partial full-private-audit directory: $OUT_ROOT" >&2
  exit 3
fi

"$PY" - "$REQUESTS" "$OUT_ROOT" "$LANES_PER_ARM" <<'PY'
import json, pathlib, sys
source, root, lanes = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), int(sys.argv[3])
rows = [json.loads(line) for line in source.read_text().splitlines() if line]
assert len(rows) >= lanes
for arm in ("layer100-matched", "layer110-delta-restore"):
    for lane in range(lanes):
        lane_root = root / arm / f"lane{lane}"
        lane_root.mkdir(parents=True, exist_ok=True)
        selected = rows[lane::lanes]
        (lane_root / "requests.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected)
        )
PY

server_pids=()
capture_pids=()
cleanup() {
  for pid in "${server_pids[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  for pid in "${server_pids[@]:-}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT

for arm_index in "${!ARMS[@]}"; do
  arm=${ARMS[$arm_index]}
  artifact="$ART_ROOT/$arm"
  [[ -f "$artifact/manifest.json" ]]
  for lane in $(seq 0 $((LANES_PER_ARM - 1))); do
    global_lane=$((arm_index * LANES_PER_ARM + lane))
    gpu=$global_lane
    port=$((PORT_BASE + global_lane))
    start=$((global_lane * CPUS_PER_LANE)); end=$((start + CPUS_PER_LANE - 1))
    numa=0; (( gpu >= 4 )) && numa=1
    lane_root="$OUT_ROOT/$arm/lane$lane"
    model="$arm-lane$lane"
    taskset -c "$start-$end" numactl --membind="$numa" env \
      -u BW24_API_KEY -u BW24_FULL_PREC \
      CUDA_VISIBLE_DEVICES="$gpu" \
      BW24_COMPAT=openai BW24_SERVE_SPEC=0 BW24_KV_REUSE=0 BW24_CTX=1032 \
      BW24_FAST=1 BW24_MMVQ=1 BW24_MOE_CACHE=1 BW24_MOE_GROUPED=1 \
      BW24_MOE_PREWARM=1 BW24_MOE_PREFETCH=1 BW24_MOE_PAGE_PREFETCH=1 \
      BW24_MOE_PAGE_PREFETCH_WINDOW=8 BW24_MOE_MMAP_ADVICE=normal \
      BW24_MOE_RESIDENT=1 BW24_MOE_VRAM_FRAC=0.85 \
      BW24_SPILL_IO=worker BW24_SPILL_PREAD_DEPTH=8 BW24_SPILL_STATS=1 \
      BW24_MOE_WEIGHT_TRACE="$lane_root/routes.trace" BW24_MODELS="$model=$artifact" \
      BW24_ADDR="127.0.0.1:$port" "$SERVER_BIN" >"$lane_root/server.log" 2>&1 &
    server_pids+=("$!")
  done
done

for arm_index in "${!ARMS[@]}"; do
  arm=${ARMS[$arm_index]}
  for lane in $(seq 0 $((LANES_PER_ARM - 1))); do
    global_lane=$((arm_index * LANES_PER_ARM + lane))
    port=$((PORT_BASE + global_lane))
    pid=${server_pids[$global_lane]}
    lane_root="$OUT_ROOT/$arm/lane$lane"
    model="$arm-lane$lane"
    for _ in {1..900}; do
      kill -0 "$pid" 2>/dev/null || { tail -n 100 "$lane_root/server.log"; exit 4; }
      if curl -fsS --max-time 5 "http://127.0.0.1:$port/health" >"$lane_root/health.json.tmp" 2>/dev/null \
        && "$PY" "$HEALTH_TOOL" "$lane_root/health.json.tmp" "$model" --exact; then
        mv "$lane_root/health.json.tmp" "$lane_root/health.json"
        break
      fi
      sleep 1
    done
    [[ -f "$lane_root/health.json" ]]
    "$PY" "$CAPTURE_TOOL" --requests "$lane_root/requests.jsonl" \
      --endpoint "http://127.0.0.1:$port/v1/completions" --model "$model" \
      --out "$lane_root/results.jsonl" --timeout 1800 --retries 0 \
      >"$lane_root/capture.log" 2>&1 &
    capture_pids+=("$!")
  done
done

failure=0
for pid in "${capture_pids[@]}"; do wait "$pid" || failure=1; done
(( failure == 0 )) || { echo "full private prompt capture failed" >&2; exit 5; }
cleanup
trap - EXIT

expected_total=$("$PY" - "$REQUESTS" <<'PY'
import json, pathlib, sys
print(sum(int(json.loads(line)["prompt_tokens"]) for line in pathlib.Path(sys.argv[1]).read_text().splitlines() if line))
PY
)

for arm in "${ARMS[@]}"; do
  for lane in $(seq 0 $((LANES_PER_ARM - 1))); do
    lane_root="$OUT_ROOT/$arm/lane$lane"
    expected=$("$PY" - "$lane_root/requests.jsonl" <<'PY'
import json, pathlib, sys
print(sum(int(json.loads(line)["prompt_tokens"]) for line in pathlib.Path(sys.argv[1]).read_text().splitlines() if line))
PY
)
    "$PY" "$ROUTE_VALIDATOR" --manifest "$ART_ROOT/$arm/manifest.json" \
      --trace "$lane_root/routes.trace" --expected-tokens "$expected" \
      --layers 1-79 --top-k 8 --output "$lane_root/route-gate.json"
    if grep -E -n -i 'CUDA_ERROR|illegal address|errors=[1-9][0-9]*|short_reads=[1-9][0-9]*' \
        "$lane_root/server.log"; then
      echo "$arm lane $lane contains a CUDA or spill correctness error" >&2
      exit 6
    fi
  done
  "$PY" "$MERGE_TOOL" --requests "$REQUESTS" --lane-root "$OUT_ROOT/$arm" \
    --lanes "$LANES_PER_ARM" --output-trace "$OUT_ROOT/$arm.routes.trace" \
    --output-results "$OUT_ROOT/$arm.results.jsonl" --receipt "$OUT_ROOT/$arm.merge.json"
  "$PY" "$ROUTE_VALIDATOR" --manifest "$ART_ROOT/$arm/manifest.json" \
    --trace "$OUT_ROOT/$arm.routes.trace" --expected-tokens "$expected_total" \
    --layers 1-79 --top-k 8 --output "$OUT_ROOT/$arm.route-gate.json"
done

"$PY" "$ANALYSIS_TOOL" --requests "$REQUESTS" \
  --base-trace "$OUT_ROOT/layer100-matched.routes.trace" \
  --restored-trace "$OUT_ROOT/layer110-delta-restore.routes.trace" \
  --selection-receipt "$SELECTION_RECEIPT" \
  --output "$OUT_ROOT/private-route-displacement.json"

sha256sum "$SERVER_BIN" "$REQUESTS" "$MERGE_TOOL" "$ANALYSIS_TOOL" \
  "$SELECTION_RECEIPT" "$ART_ROOT"/*/manifest.json \
  "$OUT_ROOT"/*.routes.trace "$OUT_ROOT"/*.results.jsonl "$OUT_ROOT"/*.json \
  "$OUT_ROOT"/*/lane*/*.json "$OUT_ROOT"/*/lane*/*.log \
  >"$OUT_ROOT/evidence.sha256"
date -u +%FT%TZ >"$OUT_ROOT/complete"
echo "full 24-prompt private route audit complete"
