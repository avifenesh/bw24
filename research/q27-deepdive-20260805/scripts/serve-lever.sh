#!/bin/bash
# PHASE-1 serve-level confirmation of the two landed decode levers, through memra-server
# (the goal shape: serving consolidated on one card). Arms:
#   base   = tree default (FFN fuse2 ON, gen-graph budget-keyed)
#   nofuse = MEMRA_Q8_FFN_FUSE2=0        (isolates lever 1 at serve level)
# Server restarted per arm (dispatch env reads once at load). c=1 and c=8, N=3 passes,
# arm order alternated per pass so thermal drift cancels in the pair mean.
set -u
cd /root/bw24
R=/root/receipts-dd
Q8=/root/models/Qwen3.6-27B-Q8_0.gguf
ADDR=127.0.0.1:8199
BASE=http://$ADDR
export LD_LIBRARY_PATH=/usr/local/cuda-13.1/compat
export PATH=$HOME/.cargo/bin:/usr/local/cuda-13.1/bin:$PATH

start_server() { # $1 extra-env  $2 logfile
  env $1 MEMRA_MODELS="q27=$Q8" MEMRA_ADDR=$ADDR target/release/memra-server > "$2" 2>&1 &
  SPID=$!
  for _ in $(seq 300); do curl -sf $BASE/health >/dev/null 2>&1 && return 0; sleep 2; done
  echo "SERVER-DID-NOT-COME-UP $2"; tail -20 "$2"; return 1
}
stop_server() { kill "${SPID:-0}" 2>/dev/null; wait "${SPID:-0}" 2>/dev/null || true; sleep 3; }

for pass in 1 2 3; do
  if [ $((pass % 2)) -eq 1 ]; then ARMS="base nofuse"; else ARMS="nofuse base"; fi
  for arm in $ARMS; do
    ENVX=""; [ "$arm" = nofuse ] && ENVX="MEMRA_Q8_FFN_FUSE2=0"
    start_server "$ENVX" "$R/logs/serve-$arm-p$pass.server.log" || exit 1
    for c in 1 8; do
      python3 tools/load-serve.py --base $BASE --model q27 --concurrency $c \
        --requests $((c*6)) --max-tokens 128 \
        --out "$R/logs/serve-points.jsonl" --label "$arm-c$c-p$pass" \
        >> "$R/logs/serve-load-$arm-p$pass.log" 2>&1
      echo "serve $arm c=$c p$pass: $(tail -1 $R/logs/serve-points.jsonl)"
    done
    stop_server
    nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm --format=csv,noheader
  done
done
echo SERVE-LEVER-DONE
