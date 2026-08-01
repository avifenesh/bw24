#!/bin/bash
# spec-serving burst-size sensitivity: MEMRA_SPEC_BURST in {8, 32, 128} at c=4 and c=8.
# Interleaved x2 rounds (8 -> 32 -> 128 per round). All params baked.
set -u
cd /home/ubuntu/arc5
OUT=/home/ubuntu/arc5/research/spec-serving-20260801
MODEL=/opt/dlami/nvme/models/Qwen3.6-27B-Q4_K_M.gguf
BIN=/home/ubuntu/arc5/target/release/memra-server
LS=/home/ubuntu/arc5/tools/load-serve.py

wait_ready() {
  for i in $(seq 1 120); do
    sleep 2
    if curl -s -m 2 "http://127.0.0.1:8186/health" | grep -q q27; then return 0; fi
  done
  return 1
}
wait_drain() {
  for i in $(seq 1 30); do
    sleep 2
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4)
    [ "$u" -lt 1000 ] && return 0
  done
  return 1
}

for r in 1 2; do
  for b in 8 32 128; do
    echo "--- burst=$b round $r $(date -u +%FT%TZ) ---" >> "$OUT/burst-driver.log"
    CUDA_VISIBLE_DEVICES=4 MEMRA_MODELS="q27=$MODEL" MEMRA_ADDR=127.0.0.1:8186 \
      MEMRA_SPEC_K=3 MEMRA_SPEC_HPOST=1 MEMRA_SPEC_PMIN=0.3 MEMRA_SPEC_BURST=$b \
      nohup "$BIN" >> "$OUT/server-spec-burst$b-r$r.log" 2>&1 &
    pid=$!
    if wait_ready; then
      for c in 4 8; do
        python3 "$LS" --base http://127.0.0.1:8186 --model q27 --concurrency "$c" \
          --max-tokens 128 --greedy --label "specburst$b-r$r-c$c" \
          --out "$OUT/burst-points.jsonl" --per-request "$OUT/burst-per-request.jsonl" \
          >> "$OUT/burst-driver.log" 2>&1
      done
    else
      echo "burst=$b r$r: server failed to start" >> "$OUT/burst-driver.log"
    fi
    kill "$pid" 2>/dev/null
    wait_drain
  done
done
touch "$OUT/.burst-done"
