#!/usr/bin/env bash
# W3: the cache-hit arm at a machine-config prefix-cache budget that can hold the 4k entry.
# W1 finding: a 4107-token step35 entry is 343.0MB and the DEFAULT MEMRA_PREFIX_CACHE_MB=256
# refuses the seed insert ("skip seed insert: entry 343.0MB > budget 268MB"), so 4k repeats
# can never hit at the default. MEMRA_PREFIX_CACHE_MB is the documented machine-specific
# config seam; 2048MB here (~90GiB VRAM headroom on the pair, entries are host-side).
# Also reruns the SHORT cache-hit arm (fits the default; this is the ms-class receipt).
set -uo pipefail
export PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
ROOT=$HOME/serve-receipt
REPO=$ROOT/memra
BIN=$REPO/target/release/memra-server
MODEL=$HOME/step37/models/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf
DRAFT=$HOME/step37/models/step-3.7-flash/Step3.7-flash-mtp-Q8_0.gguf
PROMPT4K=$ROOT/prompt-pp4096.txt
RAW=$ROOT/raw
KEY=receipt-trial-20260808
PORT=18097
BASE=http://127.0.0.1:$PORT
LABEL=serve-ready-receipt-w3
mkdir -p "$RAW"
cd "$REPO" || exit 1
SERVER_PID=
stop_server() { [[ -n ${SERVER_PID:-} ]] && { kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; SERVER_PID=; }; }
TS=$(date -u +%Y%m%dT%H%M%SZ)
LOG=$RAW/w3-cachehit-$TS.log
{
echo "=== W3 cache-hit arm at MEMRA_PREFIX_CACHE_MB=2048  $TS"
echo "commit=$(cat $ROOT/COMMIT.txt)"
(
  flock -w 7200 9 || { echo "LOCK TIMEOUT W3"; exit 75; }
  trap stop_server EXIT
  echo "lock acquired $(date -u +%FT%TZ)"
  env \
    -u MEMRA_PRIME_PIPE -u MEMRA_PRIME_CHUNK -u MEMRA_PREFILL_TICK \
    -u MEMRA_SERVE_BATCH -u MEMRA_PRIME_BATCH_HOLD_MS -u MEMRA_SERVE_SPEC \
    -u MEMRA_SPEC_GATE -u MEMRA_SPEC_GATE_LOW -u MEMRA_SPEC_GATE_HIGH \
    MEMRA_MODELS="step35=${MODEL}+${DRAFT}" \
    MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 \
    MEMRA_MOE_GROUPED=1 \
    MEMRA_PREFIX_CACHE_MB=2048 \
    MEMRA_API_KEY="$KEY" \
    MEMRA_ADDR="127.0.0.1:$PORT" \
    "$BIN" >"$RAW/server-w3-$TS.log" 2>&1 &
  SERVER_PID=$!
  for attempt in $(seq 1 180); do
    sleep 5
    curl -sf "$BASE/readyz" >/dev/null 2>&1 && { echo "server ready after ~$((attempt*5))s"; break; }
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "SERVER DIED"; tail -60 "$RAW/server-w3-$TS.log"; exit 1; }
  done

  echo; echo "########## 4k cache-hit repeat N=3, budget 2048MB ##########"
  MEMRA_API_KEY=$KEY python3 $ROOT/hitprobe.py --base $BASE --model step35 \
    --prompt-file "$PROMPT4K" --requests 3 --max-tokens 8 \
    --label "$LABEL-4k" --out $RAW/ttft-cachehit-4k-2048-$TS.jsonl --timeout 600 || exit 1

  echo; echo "########## SHORT cache-hit repeat N=3 (fits any budget) ##########"
  SHORTP=$RAW/short-prompt.txt
  python3 - "$SHORTP" <<'PY'
import sys
FILLER = ("The quick brown fox jumps over the lazy dog while the seasoned engineer "
          "measures throughput, latency, and saturation across every replica. ")
open(sys.argv[1], "w").write(
    "Summarize the operational state of a GPU serving cluster in exactly three "
    "sentences, then list four risks. Context follows. " + FILLER * 8)
PY
  MEMRA_API_KEY=$KEY python3 $ROOT/hitprobe.py --base $BASE --model step35 \
    --prompt-file "$SHORTP" --requests 3 --max-tokens 8 \
    --label "$LABEL-short" --out $RAW/ttft-cachehit-short-$TS.jsonl --timeout 600 || exit 1

  grep -E "prefix-cache" "$RAW/server-w3-$TS.log" | tail -12
  stop_server
  sleep 3
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
  echo "lock released $(date -u +%FT%TZ)"
) 9>/tmp/memra-gpu.lock
echo "=== W3 rc=$?"
echo "=== done $(date -u +%FT%TZ)"
} > "$LOG" 2>&1
echo "log: $LOG"
tail -12 "$LOG"
