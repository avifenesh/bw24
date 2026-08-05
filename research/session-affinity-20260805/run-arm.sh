#!/usr/bin/env bash
# One arm of the affinity gate: boot the owner's daily-driver serve config against THIS
# lane's binary, drive the rewrite-pattern conversation, tear down.
#
# The serve config is the owner regime verbatim (tools/serve-examples/serve-qwen36-27b-memra:
# 27B NVFP4 + regime draft, ctx 128k, MEMRA_MAX_SESSIONS=1, MEMRA_REUSE_POOL=1,
# MEMRA_PRIME_CHUNK=2048) because the number under test is that regime's TTFT. The only
# thing an arm varies is MEMRA_AFFINITY.
#
# Usage: run-arm.sh <arm-name> <MEMRA_AFFINITY 0|1> [turns] [extra driver args...]
#        run-arm.sh <arm-name> <MEMRA_AFFINITY 0|1> --replay <transcript> [driver args...]
# Writes: <arm>.jsonl (per-turn rows), <arm>.server.log (raw server log — the receipt).
# GPU: caller holds flock /tmp/gpu5090.lock for timing arms.
set -uo pipefail
cd "$(dirname "$0")"

ARM="${1:?arm name}"
AFF="${2:?MEMRA_AFFINITY 0|1}"
if [ "${3:-}" = "--replay" ]; then
  MODE_ARGS=(--replay "$4")   # positional: --replay <transcript> <port> <out>
  shift 4
else
  MODE_ARGS=()
  TURNS="${3:-25}"
  shift 3 2>/dev/null || shift 2
fi

DIR=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp
MODEL="$DIR/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf"
DRAFT="$DIR/draft-daily-owntrim-nvfp4head-q4blk.gguf"
BIN="${MEMRA_BIN:-$(cd ../.. && pwd)/target/release/memra-server}"
PORT="${PORT:-8179}"
LOG="$PWD/$ARM.server.log"
OUT="$PWD/$ARM.jsonl"
rm -f "$OUT"

[ -f "$MODEL" ] || { echo "run-arm: SKIP (no model at $MODEL)"; exit 0; }
[ -x "$BIN" ] || { echo "run-arm: no binary at $BIN"; exit 1; }

echo "== arm $ARM: MEMRA_AFFINITY=$AFF, ${MODE_ARGS[*]:-$TURNS turns}, bin $BIN"
MEMRA_MODELS="qwen36-27b=$MODEL+$DRAFT" \
MEMRA_ADDR="127.0.0.1:$PORT" \
MEMRA_API_KEY=aviary-local \
MEMRA_CTX="${CTX:-131072}" \
MEMRA_MAX_SESSIONS=1 \
MEMRA_REUSE_POOL="${REUSE_POOL:-1}" \
MEMRA_PRIME_CHUNK=2048 \
MEMRA_AFFINITY="$AFF" \
"$BIN" > "$LOG" 2>&1 &
SPID=$!
cleanup() { kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null; }
trap cleanup EXIT

for _ in $(seq 180); do
  curl -sf -H 'Authorization: Bearer aviary-local' "http://127.0.0.1:$PORT/health" \
    >/dev/null 2>&1 && break
  kill -0 "$SPID" 2>/dev/null || { echo "server died during load; log tail:"; tail -20 "$LOG"; exit 1; }
  sleep 2
done
curl -sf -H 'Authorization: Bearer aviary-local' "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 \
  || { echo "server never came up; log tail:"; tail -20 "$LOG"; exit 1; }

if [ ${#MODE_ARGS[@]} -gt 0 ]; then
  python3 drive-affinity.py "${MODE_ARGS[@]}" "$PORT" "$OUT" "$@"
else
  python3 drive-affinity.py "$PORT" "$OUT" "$TURNS" "$@"
fi
RC=$?
# The resume decisions are in the server log, not the HTTP responses — keep the count next to
# the rows so a summary can never drift from its receipt.
echo "# affinity rewinds: $(grep -c 'spec-affinity: rewound' "$LOG" 2>/dev/null || echo 0)"
echo "# prefix resumes:   $(grep -c 'spec-reuse:' "$LOG" 2>/dev/null || echo 0)"
exit $RC
