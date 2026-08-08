#!/usr/bin/env bash
# pp2spec-crash STEP 7 — the FENCE FIX under the repro. Build 7450928b: sentinel traps
# (kept — correctness armor) + PpNRt::fence_stages_behind at verify-ppn entry.
# VERDICT RULE: if the fence is the root cause, ZERO trap lines and ZERO errors across
# the exact-A sequence x3 reps. Any trap line = the race has another entry point (the
# trap quotes it). Then the crash gate: c=4 and c=8 over >=200 requests total.
set -uo pipefail
cd ~/memra
export PATH=$HOME/.cargo/bin:$PATH
OUT=~/receipts/pp2crash
mkdir -p "$OUT"
Q9=/scratch-models/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
BIN=target/release
ADDR=127.0.0.1:8123
BASE=http://$ADDR

exec 9>/tmp/memra-gpu.lock
flock -w 14400 9 || { echo "FAIL: gpu lock timeout"; exit 1; }
echo "gpu lock acquired $(date -u +%FT%TZ)"

wait_up() {
  for _ in $(seq 1 "$1"); do
    curl -sf "$BASE/v1/models" >/dev/null 2>&1 && return 0
    sleep 2
  done
  return 1
}

if curl -sf "$BASE/v1/models" >/dev/null 2>&1; then
  echo "FAIL: something already serving $ADDR"; exit 1
fi

for rep in 1 2 3; do
  echo "=== FENCE rep $rep: exact A sequence ==="
  env MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=1,0 MEMRA_PP2SPEC_UNQUARANTINE=1 \
    MEMRA_SPEC_GATE=0 MEMRA_MODELS="q9=$Q9" MEMRA_ADDR=$ADDR \
    $BIN/memra-server > "$OUT/X${rep}-server.log" 2>&1 &
  PID=$!
  if ! wait_up 180; then echo "FAIL: rep $rep server never came up"; tail -20 "$OUT/X${rep}-server.log"; kill $PID 2>/dev/null; exit 1; fi
  python3 tools/load-serve.py --base "$BASE" --model q9 --concurrency 2 \
    --requests 8 --max-tokens 96 --greedy --warmup 1 --label X${rep}-c2 \
    --out "$OUT/X-points.jsonl" > "$OUT/X${rep}-c2.log" 2>&1
  python3 tools/load-serve.py --base "$BASE" --model q9 --concurrency 4 \
    --requests 16 --max-tokens 96 --greedy --warmup 0 --label X${rep}-c4 \
    --out "$OUT/X-points.jsonl" > "$OUT/X${rep}-c4.log" 2>&1
  kill $PID 2>/dev/null; wait $PID 2>/dev/null; sleep 4
  echo "--- rep $rep trap/illegal lines ---"
  grep -n -i "sentinel\|#87 trap\|illegal\|abort" "$OUT/X${rep}-server.log" | grep -v "OVERRIDDEN" | head -6 || echo "(rep $rep CLEAN)"
done

echo "=== CRASH GATE: c=4 x 100 + c=8 x 104 on one server (fresh) ==="
env MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=1,0 MEMRA_PP2SPEC_UNQUARANTINE=1 \
  MEMRA_SPEC_GATE=0 MEMRA_MODELS="q9=$Q9" MEMRA_ADDR=$ADDR \
  $BIN/memra-server > "$OUT/XG-server.log" 2>&1 &
PID=$!
if ! wait_up 180; then echo "FAIL: gate server never came up"; kill $PID 2>/dev/null; exit 1; fi
python3 tools/load-serve.py --base "$BASE" --model q9 --concurrency 4 \
  --requests 100 --max-tokens 96 --greedy --warmup 1 --label XG-c4x100 \
  --out "$OUT/X-points.jsonl" > "$OUT/XG-c4.log" 2>&1
python3 tools/load-serve.py --base "$BASE" --model q9 --concurrency 8 \
  --requests 104 --max-tokens 96 --greedy --warmup 0 --label XG-c8x104 \
  --out "$OUT/X-points.jsonl" > "$OUT/XG-c8.log" 2>&1
kill $PID 2>/dev/null; wait $PID 2>/dev/null; sleep 4
echo "--- gate trap/illegal lines ---"
grep -n -i "sentinel\|#87 trap\|illegal\|abort" "$OUT/XG-server.log" | grep -v "OVERRIDDEN" | head -8 || echo "(gate CLEAN)"
nvidia-smi --query-gpu=index,memory.used --format=csv > "$OUT/X-gpu-post.csv"
echo PP2CRASH_ROUND7_DONE
