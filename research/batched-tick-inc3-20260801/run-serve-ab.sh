#!/bin/bash
# inc3 serve A/B on the 5090 (single replica, fresh server per point, arms INTERLEAVED
# per (rep, c) cell). Arms:
#   base  — MEMRA_SERVE_TOKDEFER=0 (the exact pre-3c tick, same binary)
#   defer — naked (3c deferred per-tick token readback)
# Load: tools/load-serve.py, temp 0.7, ~200-tok prompt, 128-tok gens, requests=4c.
# The gpu5090 lock is held per POINT (server spawn -> load -> kill) so co-resident
# lanes can interleave between points.
set -u
W=/home/avifenesh/projects/wt-batched-tick-3
R=$W/research/batched-tick-inc3-20260801
M=/home/avifenesh/models/qwen3.5-9b-judge-q8_0.gguf
PORT=8093
BASE=http://127.0.0.1:$PORT
point() { # $1 arm-name $2 env(k=v space-separated, may be empty) $3 concurrency $4 rep
  local arm=$1 envv=$2 c=$3 rep=$4
  local log=$R/sv-$arm-c$c-rep$rep.log srv=$R/server-$arm-c$c-rep$rep.log
  (
    flock 9
    env $envv MEMRA_MODELS="qwen=$M" MEMRA_ADDR=127.0.0.1:$PORT \
      "$W/target/release/memra-server" >"$srv" 2>&1 &
    SRV=$!
    up=0
    for _ in $(seq 1 180); do
      curl -s $BASE/health >/dev/null 2>&1 && { up=1; break; }
      sleep 1
    done
    if [ "$up" != 1 ]; then echo "SERVER FAILED $arm c$c rep$rep"; tail -3 "$srv"; kill $SRV 2>/dev/null; exit 1; fi
    python3 "$W/tools/load-serve.py" --base $BASE --model qwen --concurrency "$c" \
      --requests $((4 * c)) --max-tokens 128 --label "$arm-c$c-r$rep" \
      --out "$R/serve-points.jsonl" --per-request "$R/serve-per-request.jsonl" \
      2>&1 | tee "$log"
    curl -s $BASE/metrics > "$R/metrics-$arm-c$c-rep$rep.json" 2>/dev/null
    kill $SRV 2>/dev/null
    wait $SRV 2>/dev/null
  ) 9>/tmp/gpu5090.lock
}
for rep in 1 2 3 4; do
  for c in 8 16 32; do
    point base  "MEMRA_SERVE_TOKDEFER=0" "$c" "$rep"
    point defer ""                       "$c" "$rep"
  done
done
# Phase 2 — the chunk-16 serve pair (both arms carry the q8rp mirror so chunk width is
# the only variable; c8m pins the width with the door, c16m rides the auto exact-16
# policy — the server log line "decode chunk cap 16 (exact-16 tier)" is the receipt).
for rep in 1 2 3 4; do
  for c in 16 32; do
    point c8m  "MEMRA_Q8RP=1 MEMRA_DECODE_BATCH_CAP=8" "$c" "$rep"
    point c16m "MEMRA_Q8RP=1"                          "$c" "$rep"
  done
done
echo SERVE-AB-DONE
