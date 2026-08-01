#!/bin/bash
# batched-tick inc2 ROUND 2 (GPU 1): component-3 serving A/B + interference forensics.
# Arms (fresh server per point, interleaved within each (rep, c) cell):
#   base = ~/memra-int prebuilt v0.59.0 (increment 1) — denominator
#   faap = arc1 binary, MEMRA_SERVE_LEANLOGITS=0 (components 1+2 only)
#   lean = arc1 binary, naked (components 1+2+3)
# Each point also snapshots nvidia-smi compute-apps before/after (neighbor forensics for
# the round-1 bimodality: bad points should correlate with concurrent neighbor load).
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
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader \
    > "$OUT/neigh-$label-c$c-rep$rep-pre.txt" 2>&1
  env $envv MEMRA_MODELS="qwen=$MODEL" MEMRA_ADDR=127.0.0.1:$PORT \
    "$bin" >"$OUT/r2-server-$label-c$c-rep$rep.log" 2>&1 &
  local srv=$!
  local up=0
  for _ in $(seq 1 120); do
    if curl -s "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then up=1; break; fi
    sleep 1
  done
  if [ "$up" != 1 ]; then echo "$label c$c rep$rep: SERVER FAILED TO START"; kill $srv 2>/dev/null; return; fi
  python3 "$LOADPY" --base "http://127.0.0.1:$PORT" --model qwen \
    --concurrency "$c" --max-tokens 128 --label "r2-$label-c$c-rep$rep" \
    --out "$OUT/load-points.jsonl" --per-request "$OUT/per-request.jsonl" \
    >"$OUT/r2-load-$label-c$c-rep$rep.log" 2>&1
  curl -s "http://127.0.0.1:$PORT/metrics" >"$OUT/r2-metrics-$label-c$c-rep$rep.json"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader \
    > "$OUT/neigh-$label-c$c-rep$rep-post.txt" 2>&1
  kill $srv 2>/dev/null; wait $srv 2>/dev/null
  local agg p50
  agg=$(grep -oE '"agg_tok_s": [0-9.]+' "$OUT/r2-load-$label-c$c-rep$rep.log" | tail -1 | grep -oE '[0-9.]+')
  p50=$(python3 -c "import json; print(json.load(open('$OUT/r2-metrics-$label-c$c-rep$rep.json')).get('step_p50_ms','?'))" 2>/dev/null)
  echo "$label c=$c rep$rep: agg=$agg tok/s tick_p50=${p50}ms"
}

BASE_BIN="$HOME/memra-int/target/release/memra-server"
NEW_BIN="$HOME/arc1/target/release/memra-server"

for rep in 1 2 3 4; do
  for c in 8 16 32; do
    run_point base "$BASE_BIN" "MEMRA_NOOP=0" "$c" "$rep"
    run_point faap "$NEW_BIN"  "MEMRA_SERVE_LEANLOGITS=0" "$c" "$rep"
    run_point lean "$NEW_BIN"  "MEMRA_NOOP=0" "$c" "$rep"
  done
done
echo "SERVE-AB2 DONE"
