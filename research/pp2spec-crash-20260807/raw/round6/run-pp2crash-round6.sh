#!/usr/bin/env bash
# pp2spec-crash STEP 6 — run the SENTINEL-TRAP build (32d86e21). The traps convert the
# MMU fault into a quoted error naming the first-NaN buffer:
#   "draft(graph) ... round-seed NaN N/4096"      -> h_seed handoff (verify-side producer)
#   "draft(eager) ... head-logits NaN / step-seed" -> discriminates head vs seed
#   "verify argmax sentinel ... col NaN"           -> stage-split verify trunk poisoned
# Exact A sequence, x3 reps for the pattern; server survives (error, not context death) —
# so also check whether later requests on the SAME server recover (stickiness should be GONE
# if the trap fires before the MMU fault poisons the context).
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
  echo "=== TRAP rep $rep: exact A sequence + recovery probe ==="
  env MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=1,0 MEMRA_PP2SPEC_UNQUARANTINE=1 \
    MEMRA_SPEC_GATE=0 MEMRA_MODELS="q9=$Q9" MEMRA_ADDR=$ADDR \
    $BIN/memra-server > "$OUT/T${rep}-server.log" 2>&1 &
  PID=$!
  if ! wait_up 180; then echo "FAIL: rep $rep server never came up"; tail -20 "$OUT/T${rep}-server.log"; kill $PID 2>/dev/null; exit 1; fi
  python3 tools/load-serve.py --base "$BASE" --model q9 --concurrency 2 \
    --requests 8 --max-tokens 96 --greedy --warmup 1 --label T${rep}-c2 \
    --out "$OUT/T-points.jsonl" > "$OUT/T${rep}-c2.log" 2>&1
  python3 tools/load-serve.py --base "$BASE" --model q9 --concurrency 4 \
    --requests 16 --max-tokens 96 --greedy --warmup 0 --label T${rep}-c4 \
    --out "$OUT/T-points.jsonl" > "$OUT/T${rep}-c4.log" 2>&1
  # RECOVERY PROBE: after the burst storm, does a solo request still serve?
  python3 tools/load-serve.py --base "$BASE" --model q9 --concurrency 1 \
    --requests 4 --max-tokens 48 --greedy --warmup 0 --label T${rep}-recover \
    --out "$OUT/T-points.jsonl" > "$OUT/T${rep}-recover.log" 2>&1
  kill $PID 2>/dev/null; wait $PID 2>/dev/null; sleep 4
  echo "--- rep $rep trap lines ---"
  grep -n "sentinel\|#87 trap" "$OUT/T${rep}-server.log" | head -6
  echo "--- rep $rep illegal/fatal lines ---"
  grep -n -i "illegal\|abort" "$OUT/T${rep}-server.log" | head -4 || echo "(none)"
done
nvidia-smi --query-gpu=index,memory.used --format=csv > "$OUT/T-gpu-post.csv"
echo PP2CRASH_ROUND6_DONE
