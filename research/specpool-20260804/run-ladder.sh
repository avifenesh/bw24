#!/usr/bin/env bash
# F5 ladder-path test: boot the server FIRST (full VRAM for weights), then attach
# a ballast allocation so the NEXT spec-session ask genuinely cannot fit even
# post-evict — exercising the right-size ladder + learned_ctx memo.
# Usage: run-ladder.sh <tag> <ballast_mib> [turns]   (caller wraps in flock)
set -uo pipefail
cd "$(dirname "$0")"
TAG="$1"; BALLAST="$2"; TURNS="${3:-8}"
DIR=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp
BIN="${MEMRA_BIN:-/home/avifenesh/projects/wt-specpool/target/release/memra-server}"
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
BAL=""
trap 'kill $PID $BAL 2>/dev/null; wait $PID $BAL 2>/dev/null' EXIT

for i in $(seq 1 120); do
  curl -sf -H 'Authorization: Bearer aviary-local' "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 $PID 2>/dev/null || { echo "server died during load:"; tail -5 "$LOG"; exit 1; }
  sleep 2
done

# NOW steal VRAM: the server is loaded, no session yet — the first spec ask
# (~4.4GB at 128k) + ballast will not fit, forcing the ladder.
python3 vram-ballast.py "$BALLAST" & BAL=$!
sleep 3

rm -f "curve-$TAG.jsonl"
python3 drive-session.py $PORT "curve-$TAG.jsonl" "$TURNS" --rewrite
RC=$?
kill $PID $BAL 2>/dev/null; wait $PID $BAL 2>/dev/null
trap - EXIT
echo "== evicts: $(grep -c 'spec pool evicted' "$LOG" || true), right-sized: $(grep -c 'right-sized' "$LOG" || true), tokenwise fallbacks: $(grep -c 'tokenwise path' "$LOG" || true)"
grep 'right-sized' "$LOG" | head -3
exit $RC
