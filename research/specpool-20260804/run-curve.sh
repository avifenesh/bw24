#!/usr/bin/env bash
# F5 curve runner: owner's exact serve env (serve-qwen36-27b-memra half=128k),
# worktree binary, port 8102, one long driven session. Caller wraps in flock.
# Usage: run-curve.sh <tag> [turns] [--rewrite]
set -uo pipefail
cd "$(dirname "$0")"
TAG="$1"; TURNS="${2:-30}"; EXTRA="${3:-}"
DIR=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp
BIN=/home/avifenesh/projects/wt-specpool/target/release/memra-server
PORT=8102
LOG="server-$TAG.log"

MEMRA_MODELS="qwen36-27b=$DIR/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf+$DIR/draft-daily-owntrim-nvfp4head-q4blk.gguf" \
MEMRA_ADDR="127.0.0.1:$PORT" \
MEMRA_API_KEY=aviary-local \
MEMRA_CTX=131072 \
MEMRA_MAX_SESSIONS=1 \
MEMRA_REUSE_POOL=1 \
MEMRA_PRIME_CHUNK=2048 \
"$BIN" > "$LOG" 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null; wait $PID 2>/dev/null' EXIT

for i in $(seq 1 120); do
  curl -sf -H 'Authorization: Bearer aviary-local' "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 $PID 2>/dev/null || { echo "server died during load:"; tail -5 "$LOG"; exit 1; }
  sleep 2
done

rm -f "curve-$TAG.jsonl"
python3 drive-session.py $PORT "curve-$TAG.jsonl" "$TURNS" $EXTRA
RC=$?
kill $PID 2>/dev/null; wait $PID 2>/dev/null
trap - EXIT
echo "== evict lines: $(grep -c 'spec pool evicted' "$LOG" || true), spec-reuse resumes: $(grep -c 'spec-reuse' "$LOG" || true)"
exit $RC
