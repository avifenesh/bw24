#!/bin/bash
# inc3 serving-level exactness receipts:
#  1. check-batch-exact (16 greedy prompts, batched-vs-isolated byte identity) on:
#     defer (naked, 3c), base (TOKDEFER=0) — shared isolated refs from the base arm,
#     c16m (mirror + cap 16: worker chunks at 16 through the exact-16 tier).
#  2. greedy-hash: one fixed greedy prompt across all three arms — same sha256 required
#     (the fleet-v060 cross-replica agreement pattern, single-replica form).
set -u
W=/home/avifenesh/projects/wt-batched-tick-3
R=$W/research/batched-tick-inc3-20260801
M=/home/avifenesh/models/qwen3.5-9b-judge-q8_0.gguf
PORT=8094
BASE=http://127.0.0.1:$PORT
REF=$R/exact-refs.json
run_arm() { # $1 arm  $2 env
  local arm=$1 envv=$2
  (
    flock 9
    env $envv MEMRA_MODELS="qwen=$M" MEMRA_ADDR=127.0.0.1:$PORT \
      "$W/target/release/memra-server" >"$R/server-exact-$arm.log" 2>&1 &
    SRV=$!
    up=0
    for _ in $(seq 1 180); do curl -s $BASE/health >/dev/null 2>&1 && { up=1; break; }; sleep 1; done
    if [ "$up" != 1 ]; then echo "SERVER FAILED $arm"; kill $SRV 2>/dev/null; exit 1; fi
    python3 "$W/tools/check-batch-exact.py" --base $BASE --model qwen --n 16 \
      --max-tokens 96 --label "$arm" --out "$R/batch-exact-$arm.jsonl" --ref "$REF" 2>&1 \
      | tee "$R/batch-exact-$arm.log" | tail -4
    # greedy-hash (fleet pattern): fixed greedy prompt, sha256 of the completion text.
    H=$(curl -s $BASE/v1/chat/completions -H 'Content-Type: application/json' -d '{
      "model":"qwen","messages":[{"role":"user","content":"List the first eight prime numbers, comma-separated, nothing else."}],
      "max_tokens":200,"temperature":0,"seed":0,"stream":false}' \
      | python3 -c 'import json,sys,hashlib; d=json.load(sys.stdin); print(hashlib.sha256(d["choices"][0]["message"]["content"].encode()).hexdigest()[:16])')
    echo "GREEDY_HASH inc3 $arm $H" | tee -a "$R/greedy-hash-inc3.log"
    kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
  ) 9>/tmp/gpu5090.lock
}
run_arm base "MEMRA_SERVE_TOKDEFER=0"
run_arm defer ""
run_arm c16m "MEMRA_Q8RP=1"   # auto exact-16 policy — server log carries the cap-16 line
echo SERVE-EXACT-DONE
