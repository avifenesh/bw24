#!/bin/bash
# Stage 6+7 driver: boot memra-server on the pilot artifact (+ drafter when built),
# marker check + regression rows; then same for the base artifact; diff.
set -uo pipefail
export LD_LIBRARY_PATH=/usr/local/cuda-13.1/compat
export PATH=$HOME/.cargo/bin:/usr/local/cuda-13.1/bin:$PATH
cd /root/bw24
PORT=8177
PILOT=/root/pilot/q9pilot-Q8_0.gguf
BASE=/root/pilot/q9base-Q8_0.gguf
DRAFT=${DRAFT:-/root/pilot/draft-q9pilot-owntrim.gguf}

boot() { # model [draft]
  local spec="pilot=$1"
  [ -n "${2:-}" ] && [ -f "${2:-}" ] && spec="pilot=$1+$2"
  MEMRA_COMPAT=openai MEMRA_MODELS="$spec" MEMRA_ADDR=127.0.0.1:$PORT \
    target/release/memra-server > /root/pilot/logs/server-$3.log 2>&1 &
  SPID=$!
  for _ in $(seq 150); do curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && return 0; sleep 2; done
  echo "server $3 did not come up"; tail -5 /root/pilot/logs/server-$3.log; return 1
}
stop() { kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null; sleep 2; }

case "$1" in
  pilot)
    boot "$PILOT" "$DRAFT" pilot || exit 1
    python3 /root/pilot/regression.py /root/pilot/regression-pilot.jsonl pilot $PORT
    RC=$?
    stop; exit $RC
    ;;
  base)
    boot "$BASE" "" base || exit 1
    python3 /root/pilot/regression.py /root/pilot/regression-base.jsonl base $PORT
    RC=$?
    stop; exit $RC
    ;;
esac
