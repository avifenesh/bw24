#!/usr/bin/env bash
# pro6000-prod: Battery E — the 96GB-only lever: Q8_0 split-plane mirror (MEMRA_Q8RP=1)
# -> exact-16 decode chunk tier, vs the naked Q8_0 baseline. On a 24GB 5090 the 27B
# trunk+mirror (~57GB) cannot fit; on this 96GB card it fits with room. 5090 9B receipts
# said +18.8% at c=16 (chunk 16 + mirror). Question: does it transfer to the 27B here?
#   E1 serve aggregate c=8/16/32, mirror ON vs OFF, N=3 passes interleaved per c
#   E2 c=1 single-stream mirror ON vs OFF, N=5 (does the mirror tax or help solo?)
set -u
cd /root/bw24
R=/root/receipts/q8rp
mkdir -p "$R"
Q8=/root/models/Qwen3.6-27B-Q8_0.gguf
ADDR=127.0.0.1:8199
BASE=http://$ADDR

log() { echo "[$(date -u +%H:%M:%SZ)] $*" >> "$R/driver.log"; }
gpustate() { nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw,memory.used --format=csv,noheader; }

nvidia-smi --query-gpu=timestamp,power.draw,clocks.sm,temperature.gpu,utilization.gpu,memory.used --format=csv -l 1 > "$R/gpu-1hz.csv" 2>&1 &
SMPID=$!
trap 'kill $SMPID 2>/dev/null' EXIT

start_server() { # $1 extra env, $2 logfile
  env $1 MEMRA_SERVE_SPEC=0 MEMRA_MODELS="q27=$Q8" MEMRA_ADDR=$ADDR target/release/memra-server > "$2" 2>&1 &
  SPID=$!
  for _ in $(seq 300); do curl -sf $BASE/health >/dev/null 2>&1 && return 0; sleep 2; done
  echo "server did not come up: $2"; tail -5 "$2"; return 1
}
stop_server() { kill "${SPID:-0}" 2>/dev/null; wait "${SPID:-0}" 2>/dev/null || true; sleep 3; }

# Interleave at the pass level: (off pass, on pass) x3 — server restart per arm per pass.
for pass in 1 2 3; do
  for armenv in "MEMRA_Q8RP=0" "MEMRA_Q8RP=1"; do
    arm=off; [ "$armenv" = "MEMRA_Q8RP=1" ] && arm=on
    log "E pass$pass mirror-$arm: starting server"
    start_server "$armenv" "$R/server-$arm-p$pass.log" || continue
    log "E pass$pass mirror-$arm up | $(gpustate)"
    if [ "$pass" = 1 ]; then
      for r in 1 2 3 4 5; do
        python3 tools/load-serve.py --base $BASE --model q27 --concurrency 1 --requests 4 \
          --max-tokens 128 --out "$R/points-c1.jsonl" --label "c1-$arm-r$r" >> "$R/load.log" 2>&1
      done
      log "E c1 $arm done: $(tail -1 "$R/points-c1.jsonl" | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print('agg=%.1f' % d['agg_tok_s'])" 2>/dev/null || echo parse-fail)"
    fi
    for c in 8 16 32; do
      python3 tools/load-serve.py --base $BASE --model q27 --concurrency $c --requests $((c*8)) \
        --max-tokens 128 --out "$R/points-batch.jsonl" --label "c$c-$arm-p$pass" >> "$R/load.log" 2>&1
      log "E c$c $arm p$pass: $(tail -1 "$R/points-batch.jsonl" | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print('agg=%.1f p50=%.2f err=%d' % (d['agg_tok_s'], d['lat_p50_s'], d['n_err']))" 2>/dev/null || echo parse-fail) | vram $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
    done
    stop_server
  done
done
log "BATTERY-E DONE"
echo "BATTERY-E DONE"
