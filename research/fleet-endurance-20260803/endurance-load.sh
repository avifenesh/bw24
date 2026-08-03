#!/usr/bin/env bash
set -u
OUT=/home/ubuntu/receipts/fleet-endurance-20260803
END=$(date -u -d '2026-08-03 10:45:00 UTC' +%s)
N=0
echo "LOAD START $(date -u +%FT%TZ) end-epoch $END" >> "$OUT/load-driver.log"
while [ "$(date +%s)" -lt "$END" ]; do
  N=$((N+1))
  python3 /home/ubuntu/memra/tools/load-serve.py --base http://127.0.0.1:9080 \
    --concurrency 96 --duration 120 --model qwen --max-tokens 128 \
    --out "$OUT/load-windows.jsonl" \
    --per-request "$OUT/perreq.jsonl" \
    --label endur-w$(printf %03d $N) \
    >> "$OUT/load-driver.log" 2>&1
done
echo "LOAD DONE $(date -u +%FT%TZ) windows=$N" >> "$OUT/load-driver.log"
