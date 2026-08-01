#!/usr/bin/env bash
# darklanes serving v1 load sweep — 2026-08-01
# single replica (8085, GPU 5) vs 3-replica proxy (8080, GPUs 5/6/7)
set -u
cd /home/ubuntu/darklane-serving-20260801
OUT=load-points.jsonl
PERREQ=per-request.jsonl

for C in 1 4 8 16 32 64; do
  echo "=== single c=$C ==="
  python3 load-serve.py --base http://127.0.0.1:8085 --concurrency $C \
    --model qwen --out $OUT --per-request $PERREQ --label single-gpu5
done

for C in 1 4 8 16 32 64; do
  echo "=== proxy c=$C ==="
  python3 load-serve.py --base http://127.0.0.1:8080 --concurrency $C \
    --model qwen --out $OUT --per-request $PERREQ --label proxy-3rep
done
echo "SWEEP DONE"
