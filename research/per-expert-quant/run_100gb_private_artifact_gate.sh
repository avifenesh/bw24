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
ARMS_CSV=${ARMS_CSV:-prune100_unhealed,prune100_router_repair,prune100_joint_heal}
GPU_BASE=${GPU_BASE:-0}
PORT_BASE=${PORT_BASE:-8070}
CPU_BASE=${CPU_BASE:-0}
CPUS_PER_LANE=${CPUS_PER_LANE:-12}
NUMA_SPLIT_GPU=${NUMA_SPLIT_GPU:-4}
IFS=, read -r -a ARMS <<<"$ARMS_CSV"
(( ${#ARMS[@]} >= 1 && ${#ARMS[@]} <= 8 )) || { echo "ARMS_CSV must contain 1-8 arms" >&2; exit 2; }
[[ "$GPU_BASE" =~ ^[0-9]+$ && "$PORT_BASE" =~ ^[1-9][0-9]{0,4}$ \
  && "$CPU_BASE" =~ ^[0-9]+$ && "$CPUS_PER_LANE" =~ ^[1-9][0-9]*$ \
  && "$NUMA_SPLIT_GPU" =~ ^[1-9][0-9]*$ && "$PORT_BASE" -le 65535 \
  && $((PORT_BASE + ${#ARMS[@]} - 1)) -le 65535 ]] \
  || { echo "invalid private-gate lane configuration" >&2; exit 2; }

mkdir -p "$OUT_ROOT"
exec 9>"$OUT_ROOT/gate.lock"
flock -n 9 || exit 0
[[ ! -f "$OUT_ROOT/complete" ]] || exit 0
if find "$OUT_ROOT" -mindepth 1 -maxdepth 1 ! -name gate.lock -print -quit | grep -q .; then
  echo "refusing partial private artifact gate directory: $OUT_ROOT" >&2
  exit 3
fi

PILOT_REQUESTS="$OUT_ROOT/requests.jsonl"
"$PY" - "$REQUESTS" "$PILOT_REQUESTS" <<'PY'
import hashlib, json, pathlib, sys
source, output = map(pathlib.Path, sys.argv[1:])
rows = [json.loads(line) for line in source.read_text().splitlines() if line]
short = min(rows, key=lambda row: (int(row["prompt_tokens"]), int(row["ordinal"])))
long = max(rows, key=lambda row: (int(row["prompt_tokens"]), -int(row["ordinal"])))
assert int(short["prompt_tokens"]) < int(long["prompt_tokens"])
output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in (short, long)))
print(json.dumps({
    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "pilot_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    "ordinals": [short["ordinal"], long["ordinal"]],
    "prompt_tokens": [short["prompt_tokens"], long["prompt_tokens"]],
    "total_prompt_tokens": int(short["prompt_tokens"]) + int(long["prompt_tokens"]),
}, sort_keys=True))
PY

EXPECTED_TOKENS=$("$PY" - "$PILOT_REQUESTS" <<'PY'
import json, pathlib, sys
print(sum(int(json.loads(line)["prompt_tokens"]) for line in pathlib.Path(sys.argv[1]).read_text().splitlines() if line))
PY
)

server_pids=()
capture_pids=()
cleanup() {
  for pid in "${server_pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  for pid in "${server_pids[@]:-}"; do
    wait "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT

for lane in "${!ARMS[@]}"; do
  arm=${ARMS[$lane]}
  artifact="$ART_ROOT/$arm"
  gpu=$((GPU_BASE + lane))
  port=$((PORT_BASE + lane))
  start=$((CPU_BASE + lane * CPUS_PER_LANE)); end=$((start + CPUS_PER_LANE - 1))
  trace="$OUT_ROOT/$arm.routes.trace"
  log="$OUT_ROOT/$arm.server.log"
  [[ -f "$artifact/manifest.json" && ! -e "$trace" ]]
  numa=0; (( gpu >= NUMA_SPLIT_GPU )) && numa=1
  taskset -c "$start-$end" numactl --membind="$numa" env -u BW24_API_KEY -u BW24_FULL_PREC \
    CUDA_VISIBLE_DEVICES="$gpu" \
    BW24_COMPAT=openai BW24_SERVE_SPEC=0 BW24_KV_REUSE=0 BW24_CTX=1032 \
    BW24_FAST=1 BW24_MMVQ=1 BW24_MOE_CACHE=1 BW24_MOE_GROUPED=1 \
    BW24_MOE_PREWARM=1 BW24_MOE_PREFETCH=1 BW24_MOE_PAGE_PREFETCH=1 \
    BW24_MOE_PAGE_PREFETCH_WINDOW=8 BW24_MOE_MMAP_ADVICE=normal \
    BW24_MOE_RESIDENT=1 BW24_MOE_VRAM_FRAC=0.85 \
    BW24_SPILL_IO=worker BW24_SPILL_PREAD_DEPTH=8 BW24_SPILL_STATS=1 \
    BW24_MOE_WEIGHT_TRACE="$trace" BW24_MODELS="$arm=$artifact" \
    BW24_ADDR="127.0.0.1:$port" "$SERVER_BIN" >"$log" 2>&1 &
  server_pids+=("$!")
done

for lane in "${!ARMS[@]}"; do
  arm=${ARMS[$lane]}; port=$((PORT_BASE + lane)); pid=${server_pids[$lane]}
  health="$OUT_ROOT/$arm.health.json"
  for _ in {1..900}; do
    kill -0 "$pid" 2>/dev/null || { tail -n 100 "$OUT_ROOT/$arm.server.log"; exit 4; }
    if curl -fsS --max-time 5 "http://127.0.0.1:$port/health" >"$health.tmp" 2>/dev/null \
      && "$PY" "$HEALTH_TOOL" "$health.tmp" "$arm" --exact; then
      mv "$health.tmp" "$health"
      break
    fi
    sleep 1
  done
  [[ -f "$health" ]]
  "$PY" "$CAPTURE_TOOL" --requests "$PILOT_REQUESTS" \
    --endpoint "http://127.0.0.1:$port/v1/completions" --model "$arm" \
    --out "$OUT_ROOT/$arm.results.jsonl" --timeout 1800 --retries 0 \
    >"$OUT_ROOT/$arm.capture.log" 2>&1 &
  capture_pids+=("$!")
done

failure=0
for pid in "${capture_pids[@]}"; do wait "$pid" || failure=1; done
(( failure == 0 )) || { echo "private artifact prompt gate failed" >&2; exit 5; }
cleanup
trap - EXIT

for arm in "${ARMS[@]}"; do
  "$PY" - "$PILOT_REQUESTS" "$OUT_ROOT/$arm.results.jsonl" <<'PY'
import json, pathlib, sys
requests = [json.loads(line) for line in pathlib.Path(sys.argv[1]).read_text().splitlines() if line]
results = [json.loads(line) for line in pathlib.Path(sys.argv[2]).read_text().splitlines() if line]
assert len(results) == len(requests) == 2
assert all(row.get("ok") for row in results)
assert [row["ordinal"] for row in results] == [row["ordinal"] for row in requests]
PY
  "$PY" "$ROUTE_VALIDATOR" --manifest "$ART_ROOT/$arm/manifest.json" \
    --trace "$OUT_ROOT/$arm.routes.trace" --expected-tokens "$EXPECTED_TOKENS" \
    --layers 1-79 --top-k 8 --output "$OUT_ROOT/$arm.route-gate.json"
  if rg -n -i 'CUDA_ERROR|illegal address|errors=[1-9][0-9]*|short_reads=[1-9][0-9]*' \
      "$OUT_ROOT/$arm.server.log"; then
    echo "$arm server log contains a CUDA or spill correctness error" >&2
    exit 6
  fi
done

sha256sum "$SERVER_BIN" "$REQUESTS" "$PILOT_REQUESTS" "$ROUTE_VALIDATOR" \
  "$ART_ROOT"/*/manifest.json "$OUT_ROOT"/*.health.json "$OUT_ROOT"/*.results.jsonl \
  "$OUT_ROOT"/*.routes.trace "$OUT_ROOT"/*.route-gate.json "$OUT_ROOT"/*.server.log \
  >"$OUT_ROOT/evidence.sha256"
date -u +%FT%TZ >"$OUT_ROOT/complete"
echo "private short/long artifact and pruned-route gates complete"
