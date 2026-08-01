#!/bin/bash
# batched-tick increment 2 (GPU 1, 8xH100 block box): serving A/B — single replica,
# 9B Q8_0, fresh server per point, arms interleaved within each (rep, concurrency) cell:
#   base = ~/memra-int prebuilt v0.59.0 binary (increment 1 merged) — the denominator
#   fa   = arc1 binary, MEMRA_BATCH_APPEND=0 (component 1 only: z-batched fa_decode)
#   faap = arc1 binary, naked (components 1+2: z-batched fa_decode + z-batched KV append)
# Load: load-serve.py (temp 0.7, ~200-tok prompt, 128 gen). tick p50 from /metrics captured
# right after each load point, before server shutdown. Kill by exact PID only.
set -u
cd "$HOME/arc1" || exit 1
OUT="$HOME/arc1/research/batched-tick-inc2-20260801"
LOADPY="$HOME/arc1/research/batched-tick-20260801/load-serve.py"
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=1
PORT=8094
MODEL="$HOME/models/Qwen3.5-9B-Q8_0.gguf"

run_point() { # label binary extra_env c rep
  local label=$1 bin=$2 envv=$3 c=$4 rep=$5
  env $envv MEMRA_MODELS="qwen=$MODEL" MEMRA_ADDR=127.0.0.1:$PORT \
    "$bin" >"$OUT/server-$label-c$c-rep$rep.log" 2>&1 &
  local srv=$!
  local up=0
  for _ in $(seq 1 120); do
    if curl -s "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then up=1; break; fi
    sleep 1
  done
  if [ "$up" != 1 ]; then echo "$label c$c rep$rep: SERVER FAILED TO START"; kill $srv 2>/dev/null; return; fi
  python3 "$LOADPY" --base "http://127.0.0.1:$PORT" --model qwen \
    --concurrency "$c" --max-tokens 128 --label "$label-c$c-rep$rep" \
    --out "$OUT/load-points.jsonl" --per-request "$OUT/per-request.jsonl" \
    >"$OUT/load-$label-c$c-rep$rep.log" 2>&1
  curl -s "http://127.0.0.1:$PORT/metrics" >"$OUT/metrics-$label-c$c-rep$rep.json"
  kill $srv 2>/dev/null; wait $srv 2>/dev/null
  local agg p50
  agg=$(grep -oE '"agg_tok_s": [0-9.]+' "$OUT/load-$label-c$c-rep$rep.log" | tail -1 | grep -oE '[0-9.]+')
  [ -z "$agg" ] && agg=$(python3 -c "import json,sys; print(json.loads(open('$OUT/load-points.jsonl').readlines()[-1]).get('agg_tok_s','?'))" 2>/dev/null)
  p50=$(python3 -c "import json; print(json.load(open('$OUT/metrics-$label-c$c-rep$rep.json')).get('step_p50_ms','?'))" 2>/dev/null)
  echo "$label c=$c rep$rep: agg=$agg tok/s tick_p50=${p50}ms"
}

BASE_BIN="$HOME/memra-int/target/release/memra-server"
NEW_BIN="$HOME/arc1/target/release/memra-server"

for rep in 1 2 3; do
  for c in 8 16 32; do
    run_point base "$BASE_BIN" "MEMRA_NOOP=0" "$c" "$rep"
    run_point fa   "$NEW_BIN"  "MEMRA_BATCH_APPEND=0" "$c" "$rep"
    run_point faap "$NEW_BIN"  "MEMRA_NOOP=0" "$c" "$rep"
  done
done
echo "SERVE-AB-INC2 DONE"
