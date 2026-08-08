#!/usr/bin/env bash
# 4k streaming TTFT: pipelined vs serial PP-2 prime, N=5 + one warmup per arm.
set -uo pipefail

REPO=${REPO:-"$HOME/memra"}
MODEL=${MODEL:-"/data/models/step37/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf"}
DRAFT=${DRAFT:-"/data/models/step37/step-3.7-flash/Step3.7-flash-mtp-Q8_0.gguf"}
PROMPT=${PROMPT:-"/tmp/pipeprime-prompt-pp4096.txt"}
RAW=${RAW:-"/tmp/pipeprime-ttft"}
TS=${TS:-$(date -u +%Y%m%dT%H%M%SZ)}
SUMMARY="$RAW/ttft4k-summary-$TS.log"
PORT=${PORT:-18095}
BASE="http://127.0.0.1:$PORT"

mkdir -p "$RAW"
cd "$REPO"

thermal() {
  nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm,memory.used \
    --format=csv,noheader
}

ttft_probe() {
  python3 - "$1" "$2" "$PROMPT" "$BASE" <<'PYEOF'
import json
import sys
import time
import urllib.request

label, n, prompt_path, base = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
prompt = open(prompt_path).read()
ttfts = []
for i in range(n):
    body = {
        "model": "step35",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 8,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    ttft = None
    with urllib.request.urlopen(req, timeout=600) as response:
        for raw_line in response:
            line = raw_line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            try:
                obj = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            choices = obj.get("choices") or []
            delta = (choices[0].get("delta") or {}) if choices else {}
            if delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content"):
                if ttft is None:
                    ttft = time.monotonic() - t0
    if ttft is None:
        raise RuntimeError(f"{label} request {i}: no streamed token delta")
    ttfts.append(ttft)
    print(f"  {label} req {i}: ttft={ttft:.3f}s")

ttfts.sort()
p50 = ttfts[len(ttfts) // 2]
p95 = ttfts[max(0, int(len(ttfts) * 0.95) - 1)]
print(
    f"{label}: N={n} ttft p50={p50:.3f}s p95={p95:.3f}s "
    f"min={ttfts[0]:.3f} max={ttfts[-1]:.3f}"
)
PYEOF
}

boot() {
  local label=$1
  shift
  SERVER_LOG="$RAW/ttft4k-server-$label-$TS.log"
  env MEMRA_MODELS="step35=${MODEL}+${DRAFT}" MEMRA_SERVE_SPEC=0 MEMRA_SERVE_BATCH=0 \
    MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_PREFILL_TICK=1024 \
    MEMRA_ADDR="127.0.0.1:$PORT" MEMRA_TAG="$label" "$@" \
    ./target/release/memra-server >"$SERVER_LOG" 2>&1 &
  SRV=$!
  for i in $(seq 1 120); do
    sleep 5
    if curl -sf "$BASE/readyz" >/dev/null 2>&1; then
      echo "$label ready ~$((i * 5))s server_log=$SERVER_LOG"
      return 0
    fi
    if ! kill -0 "$SRV" 2>/dev/null; then
      echo "$label SERVER DIED"
      cat "$SERVER_LOG"
      return 1
    fi
  done
  echo "$label readiness timeout"
  return 1
}

stop_server() {
  if [[ -n ${SRV:-} ]]; then
    kill "$SRV" 2>/dev/null || true
    wait "$SRV" 2>/dev/null || true
    SRV=
  fi
}

{
  echo "=== pipeprime TTFT $TS commit=$(git rev-parse HEAD)"
  echo "prompt=$PROMPT bytes=$(wc -c <"$PROMPT") sha256=$(sha256sum "$PROMPT" | awk '{print $1}')"
  echo "geometry: MEMRA_PREFILL_TICK=1024, naked PP-2 auto chunk"
  (
    flock -w 14400 9 || {
      echo "LOCK TIMEOUT"
      exit 75
    }
    trap stop_server EXIT
    echo "lock acquired $(date -u +%FT%TZ)"
    thermal

    echo "########## PIPE ##########"
    boot pipe || exit 1
    ttft_probe warmup-pipe 1 || exit 1
    ttft_probe PIPE 5 || exit 1
    stop_server
    sleep 3
    thermal

    echo "########## SERIAL ##########"
    boot serial MEMRA_PRIME_PIPE=0 || exit 1
    ttft_probe warmup-serial 1 || exit 1
    ttft_probe SERIAL 5 || exit 1
    stop_server
    sleep 2
    thermal

    echo "lock released $(date -u +%FT%TZ)"
  ) 9>/tmp/memra-gpu.lock
  ttft_rc=$?
  echo "=== ttft rc=$ttft_rc"
  echo "=== done $(date -u +%FT%TZ)"
  exit "$ttft_rc"
} >"$SUMMARY" 2>&1
