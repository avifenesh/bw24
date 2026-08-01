#!/bin/bash
# spec-serving-20260801: plain vs spec serving matrix on GPU 4 (H100), q27 = Qwen3.6-27B-Q4_K_M.
# One server at a time (resident = ~40.7GB with q8rp mirrors; two do not fit 80GB — see
# server-plain-8085.log/server-spec-8186.log OOM receipts). Arms alternate per round
# (A/B interleave; restart per arm, warm-cache load ~5s). All params baked as literals.
set -u
cd /home/ubuntu/arc5
OUT=/home/ubuntu/arc5/research/spec-serving-20260801
MODEL=/opt/dlami/nvme/models/Qwen3.6-27B-Q4_K_M.gguf
BIN=/home/ubuntu/arc5/target/release/memra-server
LS=/home/ubuntu/arc5/tools/load-serve.py

wait_ready() { # $1=port
  for i in $(seq 1 120); do
    sleep 2
    if curl -s -m 2 "http://127.0.0.1:$1/health" | grep -q q27; then return 0; fi
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

start_plain() { # port 8185, spec OFF -> batched decode path
  CUDA_VISIBLE_DEVICES=4 MEMRA_MODELS="q27=$MODEL" MEMRA_ADDR=127.0.0.1:8185 \
    MEMRA_SERVE_SPEC=0 \
    nohup "$BIN" >> "$OUT/server-plain-r$1.log" 2>&1 &
  echo $! > "$OUT/.pid"
  wait_ready 8185
}
start_spec() { # port 8186, spec ON (default) + round-47 config
  CUDA_VISIBLE_DEVICES=4 MEMRA_MODELS="q27=$MODEL" MEMRA_ADDR=127.0.0.1:8186 \
    MEMRA_SPEC_K=3 MEMRA_SPEC_HPOST=1 MEMRA_SPEC_PMIN=0.3 \
    nohup "$BIN" >> "$OUT/server-spec-r$1.log" 2>&1 &
  echo $! > "$OUT/.pid"
  wait_ready 8186
}
stop_server() {
  kill "$(cat "$OUT/.pid")" 2>/dev/null
  wait_drain
}

run_points() { # $1=port $2=arm $3=round
  for c in 1 2 4 8; do
    python3 "$LS" --base "http://127.0.0.1:$1" --model q27 --concurrency "$c" \
      --max-tokens 128 --greedy --label "$2-r$3-c$c" \
      --out "$OUT/points.jsonl" --per-request "$OUT/per-request.jsonl" \
      >> "$OUT/driver.log" 2>&1
  done
}

# correctness capture: one fixed greedy chat request, full text saved. $1=port $2=arm
capture_text() {
  curl -s -m 300 "http://127.0.0.1:$1/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"q27","messages":[{"role":"user","content":"Explain in three short paragraphs why speculative decoding preserves the target model distribution under greedy acceptance."}],"max_tokens":160,"temperature":0.0,"stream":false}' \
    > "$OUT/correctness-$2-p1.json"
  curl -s -m 300 "http://127.0.0.1:$1/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"q27","messages":[{"role":"user","content":"List the first 12 prime numbers, then state the sum of the first 6."}],"max_tokens":120,"temperature":0.0,"stream":false}' \
    > "$OUT/correctness-$2-p2.json"
}

echo "=== matrix start $(date -u +%FT%TZ) ===" >> "$OUT/driver.log"
for r in 1 2 3; do
  echo "--- round $r plain $(date -u +%FT%TZ) ---" >> "$OUT/driver.log"
  if start_plain "$r"; then
    [ "$r" = 1 ] && capture_text 8185 plain
    run_points 8185 plain "$r"
  else
    echo "round $r: plain server failed to start" >> "$OUT/driver.log"
  fi
  stop_server
  echo "--- round $r spec $(date -u +%FT%TZ) ---" >> "$OUT/driver.log"
  if start_spec "$r"; then
    [ "$r" = 1 ] && capture_text 8186 spec
    run_points 8186 spec "$r"
  else
    echo "round $r: spec server failed to start" >> "$OUT/driver.log"
  fi
  stop_server
done
echo "=== matrix done $(date -u +%FT%TZ) ===" >> "$OUT/driver.log"
touch "$OUT/.matrix-done"
