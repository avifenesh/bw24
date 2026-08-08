#!/usr/bin/env bash
# G5 supervision probe: prove worker death is VISIBLE and acted on, against a real CUDA worker.
#
# The unit tests pin this ladder against a FAKE worker. This script pins it against the real
# one, using MEMRA_PANIC_AFTER (worker.rs fault-injection door) to panic the GPU worker thread
# after N served requests. Two arms:
#
#   ARM A — respawn (MEMRA_WORKER_RESPAWN=1, the default): panic -> /health and /readyz flip to
#           503 with the QUOTED panic payload in `detail` -> the supervisor reloads the weights
#           -> health returns to 200 with generation bumped -> the box serves again.
#   ARM B — no respawn (MEMRA_WORKER_RESPAWN=0): panic -> the PROCESS exits 70 (EX_SOFTWARE), so
#           systemd Restart=on-failure restarts the unit whole. This is the case that used to be
#           a permanently-green health check in front of a box answering nothing.
#
# Usage: research/serve-hardening-20260806/probe-worker-death.sh [port]
# Run under: flock /tmp/memra-5090.lock  (needs the GPU)
set -u
cd "$(git rev-parse --show-toplevel)"

PORT=${1:-8098}
MODEL=${MODEL:-/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf}
BIN=${BIN:-target/release/memra-server}
BASE=http://127.0.0.1:$PORT
LOGDIR=research/serve-hardening-20260806/logs

if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
  echo "FATAL: port $PORT is already listening"; exit 2
fi
[ -f "$MODEL" ] || { echo "FATAL: no model at $MODEL"; exit 2; }

echo "== G5 worker-death probe =="
echo "commit $(git rev-parse --short HEAD) ($(git rev-parse --abbrev-ref HEAD)); rig RTX 5090 Laptop, driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"

hit() { # one real completion; prints the status only
  curl -s -o /dev/null -w '%{http_code}' -m 120 "$BASE/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"probe","messages":[{"role":"user","content":"Say OK."}],"max_tokens":8,"temperature":0}'
}
show() { # $1=path
  printf '  %-8s ' "$1"
  : > /tmp/wd-body
  curl -s -o /tmp/wd-body -w '%{http_code} ' -m 5 "$BASE$1" || printf '(refused) '
  if [ -s /tmp/wd-body ]; then cat /tmp/wd-body; else printf '(empty)'; fi
  echo
}
boot() { # $1=respawn budget, $2=logfile
  MEMRA_COMPAT=openai MEMRA_MODELS="probe=$MODEL" MEMRA_ADDR=127.0.0.1:$PORT \
    MEMRA_CTX=1024 MEMRA_PANIC_AFTER=1 MEMRA_WORKER_RESPAWN=$1 \
    "$BIN" > "$2" 2>&1 &
  SPID=$!
  for _ in $(seq 240); do curl -sf -m 2 -o /dev/null "$BASE/readyz" && return 0; sleep 1
    kill -0 $SPID 2>/dev/null || { echo "  server died during load"; return 1; }
  done
  echo "  FATAL: never became ready"; return 1
}

# ---------------------------------------------------------------- ARM A: respawn
echo
echo "--- ARM A: MEMRA_WORKER_RESPAWN=1 (default) — panic must flip health, then recover ---"
boot 1 "$LOGDIR/worker-death-respawn-server.log" || exit 1
trap 'kill -9 $SPID 2>/dev/null' EXIT
echo "  before (healthy):"; show /health; show /readyz
echo "  request that trips MEMRA_PANIC_AFTER=1 -> HTTP $(hit)"
sleep 0.5
echo "  immediately after the panic (must be 503 on BOTH, with the quoted payload in detail):"
show /health; show /livez; show /readyz
# A request arriving while the worker is dead must not HANG. It has two legitimate outcomes,
# both fine: a typed 503 (the handler's send fails / the stream closes), or a 200 served by the
# respawned worker — because the supervisor OWNS the command Receiver across a respawn, so a
# request queued during the dead window survives and the reloaded worker drains it. The second
# outcome is what this rig shows, and it is the stronger one.
echo "  a request while the worker is dead (must not hang: typed 503, or served by the respawn):"
curl -s -D /tmp/wd-hdr -o /tmp/wd-body -m 20 "$BASE/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"probe","messages":[{"role":"user","content":"hi"}],"max_tokens":4}'
grep -iE '^HTTP/|^retry-after' /tmp/wd-hdr; printf '  '; head -c 250 /tmp/wd-body; echo
echo "  waiting for the respawn to reload weights (backoff 2s + load)..."
t0=$(date +%s)
for _ in $(seq 240); do curl -sf -m 2 -o /dev/null "$BASE/readyz" && break; sleep 1; done
echo "  recovered after $(( $(date +%s) - t0 ))s (generation must be 1, not 0):"
show /health; show /readyz
echo "  the respawned worker serves: HTTP $(hit)"
echo "  supervisor lines:"
grep -aE '\[worker\] (PANIC|FATAL|respawn)' "$LOGDIR/worker-death-respawn-server.log" | sed 's/^/    /'
kill -TERM $SPID 2>/dev/null; wait $SPID 2>/dev/null; echo "  exit code after a clean SIGTERM: $?"
trap - EXIT
sleep 3

# ------------------------------------------------------- ARM B: loud process exit
echo
echo "--- ARM B: MEMRA_WORKER_RESPAWN=0 — panic must exit the PROCESS 70 (EX_SOFTWARE) ---"
boot 0 "$LOGDIR/worker-death-exit70-server.log" || exit 1
trap 'kill -9 $SPID 2>/dev/null' EXIT
echo "  before (healthy):"; show /health
echo "  request that trips the panic -> HTTP $(hit)"
for _ in $(seq 30); do kill -0 $SPID 2>/dev/null || break; sleep 0.5; done
if kill -0 $SPID 2>/dev/null; then
  echo "  FAIL: the process is STILL ALIVE after an unrecoverable worker panic"
  show /health
  kill -9 $SPID
else
  wait $SPID 2>/dev/null; RC=$?
  echo "  process exit code: $RC   (expected 70 = EX_SOFTWARE 'the engine died', vs 1 = bad config)"
  echo "  port after exit (must be refused — no listener without a worker):"; show /health
fi
echo "  supervisor lines:"
grep -aE '\[worker\] (PANIC|FATAL)' "$LOGDIR/worker-death-exit70-server.log" | sed 's/^/    /'
trap - EXIT
echo
echo "PROBE DONE"
