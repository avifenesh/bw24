#!/bin/bash
# Lane 3 (GPU 3): batched-tick device-sampling A/B — single replica, 9B Q8_0, fresh server
# per point, arms interleaved within each (rep, concurrency) cell. Arms:
#   dev  = naked binary (device-side batched sampling, the new default)
#   host = MEMRA_SERVE_DEVSAMPLE=0 (rollback seam == baseline tick semantics: all rows
#          host-sample from last_logits; decode_step_batch_sampled with all-None metas is
#          structurally the old decode_step_batch)
# Load: tools/load-serve.py (temp 0.7, ~200-tok prompt, 128 gen, requests = max(8, 4c)).
# tick p50 from /metrics captured right after each load point, before server shutdown.
set -u
cd "$HOME/lane3" || exit 1
BW="$HOME/lane3/target/release"
OUT="$HOME/lane3/research/batched-tick-20260801"
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=3
PORT=8093
MODEL="$HOME/models/Qwen3.5-9B-Q8_0.gguf"

run_point() { # label extra_env c rep
  local label=$1 envv=$2 c=$3 rep=$4
  env $envv MEMRA_MODELS="qwen=$MODEL" MEMRA_ADDR=127.0.0.1:$PORT \
    "$BW/memra-server" >"$OUT/server-$label-c$c-rep$rep.log" 2>&1 &
  local srv=$!
  local up=0
  for _ in $(seq 1 120); do
    if curl -s "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then up=1; break; fi
    sleep 1
  done
  if [ "$up" != 1 ]; then echo "$label c$c rep$rep: SERVER FAILED TO START"; kill $srv 2>/dev/null; return; fi
  python3 tools/load-serve.py --base "http://127.0.0.1:$PORT" --model qwen \
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

for rep in 1 2 3; do
  for c in 8 16 32; do
    run_point dev  "MEMRA_SERVE_DEVSAMPLE=1" "$c" "$rep"
    run_point host "MEMRA_SERVE_DEVSAMPLE=0" "$c" "$rep"
  done
done
echo "SERVE-AB DONE"
