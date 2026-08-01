#!/bin/bash
# batched-tick inc2: 20-min worker-panic soak (increments.md incident follow-up).
# ONE long-lived server (final binary, naked config = fa+append+lean), continuous c=16
# temp-0.7 load for 20 minutes (consecutive load points against the SAME process — also
# exercises retire/park/reuse-pool churn). Verdict: server log scanned for panics/errors;
# every request of every point must be ok.
set -u
cd "$HOME/arc1" || exit 1
OUT="$HOME/arc1/research/batched-tick-inc2-20260801"
LOADPY="$HOME/arc1/research/batched-tick-20260801/load-serve.py"
export CUDA_VISIBLE_DEVICES=1
PORT=8095
MODEL="$HOME/models/Qwen3.5-9B-Q8_0.gguf"

MEMRA_MODELS="qwen=$MODEL" MEMRA_ADDR=127.0.0.1:$PORT \
  "$HOME/arc1/target/release/memra-server" >"$OUT/soak-server.log" 2>&1 &
SRV=$!
echo "soak server pid=$SRV"
up=0
for _ in $(seq 1 120); do
  curl -s "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { up=1; break; }
  sleep 1
done
if [ "$up" != 1 ]; then echo "SOAK SERVER FAILED"; kill $SRV 2>/dev/null; exit 1; fi

T_END=$(( $(date +%s) + 1200 ))
i=0
while [ "$(date +%s)" -lt "$T_END" ]; do
  i=$((i+1))
  python3 "$LOADPY" --base "http://127.0.0.1:$PORT" --model qwen \
    --concurrency 16 --max-tokens 128 --label "soak-$i" \
    --out "$OUT/soak-points.jsonl" --per-request "$OUT/soak-per-request.jsonl" \
    >>"$OUT/soak-load.log" 2>&1
  # server still alive?
  if ! kill -0 $SRV 2>/dev/null; then echo "SOAK: SERVER DIED at point $i"; break; fi
done
echo "soak points completed: $i"
alive=0; kill -0 $SRV 2>/dev/null && alive=1
echo "server alive at end: $alive"
kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
echo "--- error scan ---"
grep -cE "panic|out of range|error|Error" "$OUT/soak-server.log" || true
n_err=$(python3 -c "
import json
ne=sum(json.loads(l).get('n_err',0) for l in open('$OUT/soak-points.jsonl'))
print(ne)")
echo "total request errors: $n_err"
echo "SOAK DONE"
