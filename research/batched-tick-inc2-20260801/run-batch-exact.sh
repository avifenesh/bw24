#!/bin/bash
# batched-tick inc2: serving-level batched-vs-isolated exactness (check-batch-exact).
# Usage: run-batch-exact.sh <label> [extra_env ...]
set -u
cd "$HOME/arc1" || exit 1
R=research/batched-tick-inc2-20260801
LABEL=${1:-inc2}
shift || true
mkdir -p "$R"
env "$@" CUDA_VISIBLE_DEVICES=1 MEMRA_MODELS="qwen=$HOME/models/Qwen3.5-9B-Q8_0.gguf" \
  MEMRA_ADDR=127.0.0.1:8094 ./target/release/memra-server >"$R/server-exact-$LABEL.log" 2>&1 &
SRV=$!
echo "server pid=$SRV"
up=0
for _ in $(seq 1 120); do
  curl -s http://127.0.0.1:8094/health >/dev/null 2>&1 && { up=1; break; }
  sleep 1
done
if [ "$up" != 1 ]; then echo "SERVER FAILED"; tail -5 "$R/server-exact-$LABEL.log"; kill $SRV 2>/dev/null; exit 1; fi
python3 tools/check-batch-exact.py --base http://127.0.0.1:8094 --model qwen --n 16 \
  --max-tokens 96 --label "$LABEL" --out "$R/batch-exact-$LABEL.jsonl" 2>&1 \
  | tee "$R/batch-exact-$LABEL.log" | tail -6
kill $SRV 2>/dev/null
wait $SRV 2>/dev/null
echo EXACT-DONE
