#!/usr/bin/env bash
# darklanes serving R3 — two-replicas-per-GPU pair sweep (2026-08-01), GPU 5.
# Usage: run-pair-sweep.sh <arm-label>   (e.g. pair-timeslice | pair-mps)
# Direct 2-harness (no proxy confound): pair c_total {8,16,24,32} = per-replica {4,8,12,16}.
set -u
ARM=${1:?arm label required}
D=/home/ubuntu/darklane-serving-20260801
OUT=$D/pair-sweep.jsonl
PERREQ=$D/pair-sweep-per-request.jsonl
VRAM=$D/vram-gpu5-r3.csv

( while true; do
    echo "$(date +%s),$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 5)"
    sleep 1
  done >> "$VRAM" ) &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

ghash() {
  curl -sf -m 120 http://127.0.0.1:$1/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"qwen","messages":[{"role":"user","content":"List the first eight prime numbers, comma-separated, nothing else."}],"max_tokens":64,"temperature":0,"seed":0}' \
    | python3 -c 'import json,sys,hashlib; r=json.load(sys.stdin); print(hashlib.sha256(r["choices"][0]["message"]["content"].encode()).hexdigest()[:16])'
}
echo "GREEDY_HASH $ARM 8085 $(ghash 8085)"
echo "GREEDY_HASH $ARM 8088 $(ghash 8088)"
# co-resident SIMULTANEOUS greedy (both replicas decoding at once must not change outputs)
H1=$( (ghash 8085) & (ghash 8088) & wait )
echo "GREEDY_HASH_CONCURRENT $ARM $H1"

for CT in 8 16 24 32; do
  C=$((CT / 2))
  echo "PHASE $ARM-c$CT START $(date +%s)"
  ( python3 $D/load-serve.py --base http://127.0.0.1:8085 --concurrency $C --requests $((4*C)) \
      --model qwen --out $OUT --per-request $PERREQ --label $ARM-r8085 &
    python3 $D/load-serve.py --base http://127.0.0.1:8088 --concurrency $C --requests $((4*C)) \
      --model qwen --out $OUT --per-request $PERREQ --label $ARM-r8088 &
    wait )
  echo "PHASE $ARM-c$CT END $(date +%s) VRAM $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 5)"
done
echo "PAIR SWEEP $ARM DONE"
