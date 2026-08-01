#!/bin/bash
# spec-serving postfix matrix: draft-graph persistence fix A/B.
# Arms per round (interleaved x3): spec-pre (old binary), spec-post (new), plain-post (new).
# Then burst sweep on post x2 (8/32/128 at c=4,8). All params baked as literals.
set -u
cd /home/ubuntu/arc5
OUT=/home/ubuntu/arc5/research/spec-serving-20260801
MODEL=/opt/dlami/nvme/models/Qwen3.6-27B-Q4_K_M.gguf
BIN_POST=/home/ubuntu/arc5/target/release/memra-server
BIN_PRE=/home/ubuntu/arc5/target/release/memra-server-pre
LS=/home/ubuntu/arc5/tools/load-serve.py
PTS=$OUT/postfix-points.jsonl
PR=$OUT/postfix-per-request.jsonl
DRV=$OUT/postfix-driver.log

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
stop_server() { kill "$(cat "$OUT/.pid2")" 2>/dev/null; wait_drain; }

run_points() { # $1=port $2=label-prefix $3=round $4=concurrency-list
  for c in $4; do
    python3 "$LS" --base "http://127.0.0.1:$1" --model q27 --concurrency "$c" \
      --max-tokens 128 --greedy --label "$2-r$3-c$c" \
      --out "$PTS" --per-request "$PR" >> "$DRV" 2>&1
  done
}

capture_text() { # $1=port $2=tag
  curl -s -m 300 "http://127.0.0.1:$1/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"q27","messages":[{"role":"user","content":"Explain in three short paragraphs why speculative decoding preserves the target model distribution under greedy acceptance."}],"max_tokens":160,"temperature":0.0,"stream":false}' \
    > "$OUT/correctness-$2-p1.json"
  curl -s -m 300 "http://127.0.0.1:$1/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"q27","messages":[{"role":"user","content":"List the first 12 prime numbers, then state the sum of the first 6."}],"max_tokens":120,"temperature":0.0,"stream":false}' \
    > "$OUT/correctness-$2-p2.json"
}

echo "=== postfix matrix start $(date -u +%FT%TZ) ===" >> "$DRV"
for r in 1 2 3; do
  # arm 1: spec-pre (old binary)
  echo "--- r$r spec-pre $(date -u +%FT%TZ) ---" >> "$DRV"
  CUDA_VISIBLE_DEVICES=4 MEMRA_MODELS="q27=$MODEL" MEMRA_ADDR=127.0.0.1:8186 \
    MEMRA_SPEC_K=3 MEMRA_SPEC_HPOST=1 MEMRA_SPEC_PMIN=0.3 \
    nohup "$BIN_PRE" >> "$OUT/server-specpre-r$r.log" 2>&1 &
  echo $! > "$OUT/.pid2"
  if wait_ready 8186; then run_points 8186 specpre "$r" "1 4 8"; fi
  stop_server
  # arm 2: spec-post (new binary)
  echo "--- r$r spec-post $(date -u +%FT%TZ) ---" >> "$DRV"
  CUDA_VISIBLE_DEVICES=4 MEMRA_MODELS="q27=$MODEL" MEMRA_ADDR=127.0.0.1:8186 \
    MEMRA_SPEC_K=3 MEMRA_SPEC_HPOST=1 MEMRA_SPEC_PMIN=0.3 \
    nohup "$BIN_POST" >> "$OUT/server-specpost-r$r.log" 2>&1 &
  echo $! > "$OUT/.pid2"
  if wait_ready 8186; then
    [ "$r" = 1 ] && capture_text 8186 specpost
    run_points 8186 specpost "$r" "1 2 4 8"
  fi
  stop_server
  # arm 3: plain-post (new binary, spec off)
  echo "--- r$r plain-post $(date -u +%FT%TZ) ---" >> "$DRV"
  CUDA_VISIBLE_DEVICES=4 MEMRA_MODELS="q27=$MODEL" MEMRA_ADDR=127.0.0.1:8185 \
    MEMRA_SERVE_SPEC=0 \
    nohup "$BIN_POST" >> "$OUT/server-plainpost-r$r.log" 2>&1 &
  echo $! > "$OUT/.pid2"
  if wait_ready 8185; then
    [ "$r" = 1 ] && capture_text 8185 plainpost
    run_points 8185 plainpost "$r" "1 2 4 8"
  fi
  stop_server
done
# burst sweep on post x2 (interleaved 8 -> 32 -> 128 per round)
for r in 1 2; do
  for b in 8 32 128; do
    echo "--- burstpost=$b r$r $(date -u +%FT%TZ) ---" >> "$DRV"
    CUDA_VISIBLE_DEVICES=4 MEMRA_MODELS="q27=$MODEL" MEMRA_ADDR=127.0.0.1:8186 \
      MEMRA_SPEC_K=3 MEMRA_SPEC_HPOST=1 MEMRA_SPEC_PMIN=0.3 MEMRA_SPEC_BURST=$b \
      nohup "$BIN_POST" >> "$OUT/server-burstpost$b-r$r.log" 2>&1 &
    echo $! > "$OUT/.pid2"
    if wait_ready 8186; then run_points 8186 "burstpost$b" "$r" "4 8"; fi
    stop_server
  done
done
echo "=== postfix matrix done $(date -u +%FT%TZ) ===" >> "$DRV"
touch "$OUT/.postfix-done"
