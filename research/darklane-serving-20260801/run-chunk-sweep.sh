#!/usr/bin/env bash
# darklanes serving — decode batch-width tier probe (2026-08-01), GPU 5 only.
# chunk in {8,16,32} via MEMRA_DECODE_BATCH_CAP; replicas on GPU 6/7 untouched.
set -u
D=/home/ubuntu/darklane-serving-20260801
BIN=/home/ubuntu/memra/target/release/memra-server
MODEL=/home/ubuntu/models/Qwen3.5-9B-Q8_0.gguf
OUT=$D/chunk-sweep.jsonl
PERREQ=$D/chunk-sweep-per-request.jsonl
EXACT=$D/chunk-exact.jsonl
REFS=$D/isolated-refs.json
VRAM=$D/vram-gpu5.csv

# background VRAM sampler (GPU 5, 1 Hz)
( while true; do
    echo "$(date +%s),$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 5)"
    sleep 1
  done >> "$VRAM" ) &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

restart_8085() { # $1 = chunk cap
  pkill -f 'MEMRA_ADDR=127.0.0.1:8085' 2>/dev/null
  # kill by port owner instead (env not in cmdline for exec'd children)
  P=$(lsof -t -i :8085 2>/dev/null); [ -n "$P" ] && kill $P
  sleep 3
  CUDA_VISIBLE_DEVICES=5 MEMRA_COMPAT=openai MEMRA_DECODE_BATCH_CAP=$1 \
    MEMRA_MODELS="qwen=$MODEL" MEMRA_ADDR=127.0.0.1:8085 \
    nohup $BIN > $D/logs/replica-8085-chunk$1.log 2>&1 < /dev/null &
  for _ in $(seq 90); do
    curl -sf -m 2 http://127.0.0.1:8085/health >/dev/null 2>&1 && return 0
    sleep 2
  done
  echo "FATAL: 8085 did not come up at chunk $1"; tail -3 $D/logs/replica-8085-chunk$1.log
  return 1
}

greedy_hash() {
  curl -sf -m 120 http://127.0.0.1:8085/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"qwen","messages":[{"role":"user","content":"List the first eight prime numbers, comma-separated, nothing else."}],"max_tokens":64,"temperature":0,"seed":0}' \
    | python3 -c 'import json,sys,hashlib; r=json.load(sys.stdin); print(hashlib.sha256(r["choices"][0]["message"]["content"].encode()).hexdigest()[:16])'
}

for CHUNK in 8 16 32; do
  echo "########## chunk=$CHUNK $(date +%s) ##########"
  restart_8085 $CHUNK || exit 1
  echo "BASELINE_VRAM chunk=$CHUNK $(date +%s) $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 5)"
  echo "GREEDY_HASH chunk=$CHUNK $(greedy_hash)"
  # batched exactness vs isolated refs (refs collected once, at chunk 8, sequentially)
  python3 $D/check-batch-exact.py --base http://127.0.0.1:8085 --model qwen \
    --n 24 --max-tokens 64 --label chunk$CHUNK --out $EXACT --ref $REFS || true
  for C in 8 16 32 64; do
    echo "PHASE chunk$CHUNK-c$C START $(date +%s)"
    python3 $D/load-serve.py --base http://127.0.0.1:8085 --concurrency $C \
      --model qwen --out $OUT --per-request $PERREQ --label chunk$CHUNK
    echo "PHASE chunk$CHUNK-c$C END $(date +%s) VRAM $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 5)"
  done
done
echo "CHUNK SWEEP DONE"
