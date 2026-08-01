#!/bin/bash
# spec-serving carry matrix: pending-carry fix A/B. Arms per round (interleaved x3):
# spec-carry (new binary), spec-pre (original binary), plain-carry (new binary, spec off).
# Then burst sweep on carry x2 (8/32/128 at c=4,8). All params baked as literals.
set -u
cd /home/ubuntu/arc5
OUT=/home/ubuntu/arc5/research/spec-serving-20260801
MODEL=/opt/dlami/nvme/models/Qwen3.6-27B-Q4_K_M.gguf
BIN_NEW=/home/ubuntu/arc5/target/release/memra-server
BIN_PRE=/home/ubuntu/arc5/target/release/memra-server-pre
LS=/home/ubuntu/arc5/tools/load-serve.py
PTS=$OUT/carry-points.jsonl
PR=$OUT/carry-per-request.jsonl
DRV=$OUT/carry-driver.log

wait_ready() {
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
stop_server() { kill "$(cat "$OUT/.pid3")" 2>/dev/null; wait_drain; }
run_points() { # $1=port $2=label $3=round $4=c-list
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

echo "=== carry matrix start $(date -u +%FT%TZ) ===" >> "$DRV"
for r in 1 2 3; do
  echo "--- r$r spec-carry $(date -u +%FT%TZ) ---" >> "$DRV"
  CUDA_VISIBLE_DEVICES=4 MEMRA_MODELS="q27=$MODEL" MEMRA_ADDR=127.0.0.1:8186 \
    MEMRA_SPEC_K=3 MEMRA_SPEC_HPOST=1 MEMRA_SPEC_PMIN=0.3 \
    nohup "$BIN_NEW" >> "$OUT/server-speccarry-r$r.log" 2>&1 &
  echo $! > "$OUT/.pid3"
  if wait_ready 8186; then
    [ "$r" = 1 ] && capture_text 8186 speccarry
    run_points 8186 speccarry "$r" "1 2 4 8"
  fi
  stop_server
  echo "--- r$r spec-pre2 $(date -u +%FT%TZ) ---" >> "$DRV"
  CUDA_VISIBLE_DEVICES=4 MEMRA_MODELS="q27=$MODEL" MEMRA_ADDR=127.0.0.1:8186 \
    MEMRA_SPEC_K=3 MEMRA_SPEC_HPOST=1 MEMRA_SPEC_PMIN=0.3 \
    nohup "$BIN_PRE" >> "$OUT/server-specpre2-r$r.log" 2>&1 &
  echo $! > "$OUT/.pid3"
  if wait_ready 8186; then run_points 8186 specpre2 "$r" "1 8"; fi
  stop_server
  echo "--- r$r plain-carry $(date -u +%FT%TZ) ---" >> "$DRV"
  CUDA_VISIBLE_DEVICES=4 MEMRA_MODELS="q27=$MODEL" MEMRA_ADDR=127.0.0.1:8185 \
    MEMRA_SERVE_SPEC=0 \
    nohup "$BIN_NEW" >> "$OUT/server-plaincarry-r$r.log" 2>&1 &
  echo $! > "$OUT/.pid3"
  if wait_ready 8185; then run_points 8185 plaincarry "$r" "1 2 4"; fi
  stop_server
done
for r in 1 2; do
  for b in 8 32 128; do
    echo "--- carryburst=$b r$r $(date -u +%FT%TZ) ---" >> "$DRV"
    CUDA_VISIBLE_DEVICES=4 MEMRA_MODELS="q27=$MODEL" MEMRA_ADDR=127.0.0.1:8186 \
      MEMRA_SPEC_K=3 MEMRA_SPEC_HPOST=1 MEMRA_SPEC_PMIN=0.3 MEMRA_SPEC_BURST=$b \
      nohup "$BIN_NEW" >> "$OUT/server-carryburst$b-r$r.log" 2>&1 &
    echo $! > "$OUT/.pid3"
    if wait_ready 8186; then run_points 8186 "carryburst$b" "$r" "4 8"; fi
    stop_server
  done
done
# setup-trace confirmation: one c=1 run with the trace on
CUDA_VISIBLE_DEVICES=4 MEMRA_MODELS="q27=$MODEL" MEMRA_ADDR=127.0.0.1:8186 \
  MEMRA_SPEC_K=3 MEMRA_SPEC_HPOST=1 MEMRA_SPEC_PMIN=0.3 MEMRA_SPEC_SETUP_TRACE=1 \
  nohup "$BIN_NEW" >> "$OUT/server-carrytrace.log" 2>&1 &
echo $! > "$OUT/.pid3"
if wait_ready 8186; then
  python3 "$LS" --base http://127.0.0.1:8186 --model q27 --concurrency 1 --requests 2 \
    --max-tokens 128 --greedy --label carrytrace >> "$DRV" 2>&1
fi
stop_server
echo "=== carry matrix done $(date -u +%FT%TZ) ===" >> "$DRV"
touch "$OUT/.carry-done"
