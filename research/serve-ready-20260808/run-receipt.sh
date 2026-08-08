#!/usr/bin/env bash
# SERVE-READY CAPACITY RECEIPT — box1 (2x RTX PRO 6000 Server 96GB), Step-3.7-Flash IQ4_XS PP-2.
#
# THE TRIAL CONFIG (everything measured through the SERVE surface):
#   MEMRA_MODELS="step35=<trunk>+<mtp-draft>"   Step trunk + MTP drafter attached
#   MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1      PP-2 across both cards
#   MEMRA_MOE_GROUPED=1                         Lever C explicit (grouped expert prefill)
#   spec gate at its placement-aware DEFAULT     = plain decode on PP-2 (specplace policy:
#                                                 LOW=0/HIGH=1, never admit spec)
#   MEMRA_API_KEY set                            admission + keys on (also defaults
#                                                 MEMRA_COMPAT=openai)
#   metering                                     [meter] admit lines are unconditional at
#                                                 admission; /metrics counters on
#
# Windows (each its own flock hold, cards verified back to 0 MiB before release):
#   W1  TTFT (short N=8, 4k N=5, cache-hit N=3) + decode ladder c=1/2/4/8 x3
#   W2  10-minute fleet-replay sustained load on a FRESH server (metrics from zero)
# Gates (serve-smoke full, run-gen argmax) run from run-gates.sh — separate window.
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
LABEL=serve-ready-receipt
mkdir -p "$RAW"
cd "$REPO" || exit 1

thermal() {
  nvidia-smi --query-gpu=index,name,temperature.gpu,clocks.sm,memory.used,utilization.gpu \
    --format=csv,noheader
}

SERVER_PID=
stop_server() {
  if [[ -n ${SERVER_PID:-} ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=
  fi
}

boot_server() {  # $1 = server log path
  env \
    -u MEMRA_PRIME_PIPE -u MEMRA_PRIME_CHUNK -u MEMRA_PREFILL_TICK \
    -u MEMRA_SERVE_BATCH -u MEMRA_PRIME_BATCH_HOLD_MS -u MEMRA_SERVE_SPEC \
    -u MEMRA_SPEC_GATE -u MEMRA_SPEC_GATE_LOW -u MEMRA_SPEC_GATE_HIGH \
    MEMRA_MODELS="step35=${MODEL}+${DRAFT}" \
    MEMRA_PP_STAGES=2 \
    MEMRA_PP_DEVICES=0,1 \
    MEMRA_MOE_GROUPED=1 \
    MEMRA_API_KEY="$KEY" \
    MEMRA_ADDR="127.0.0.1:$PORT" \
    "$BIN" >"$1" 2>&1 &
  SERVER_PID=$!
  for attempt in $(seq 1 180); do
    sleep 5
    if curl -sf "$BASE/readyz" >/dev/null 2>&1; then
      echo "server ready after ~$((attempt * 5))s"
      return 0
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "SERVER DIED"; tail -60 "$1"; return 1
    fi
  done
  echo "server readiness timeout"; tail -60 "$1"; return 1
}

metrics_snap() {  # $1 = out file
  curl -sf -H "Authorization: Bearer $KEY" "$BASE/metrics" > "$1" 2>/dev/null \
    || curl -sf "$BASE/metrics" > "$1"
}

TS=$(date -u +%Y%m%dT%H%M%SZ)
MAIN=$RAW/receipt-$TS.log
{
echo "=== SERVE-READY CAPACITY RECEIPT  $TS  label=$LABEL"
echo "commit=$(cat $ROOT/COMMIT.txt 2>/dev/null)"
echo "binary=$BIN sha256=$(sha256sum "$BIN" | awk '{print $1}')"
echo "model=$MODEL bytes=$(stat -c %s "$MODEL")"
echo "draft=$DRAFT bytes=$(stat -c %s "$DRAFT")"
echo "prompt4k sha256=$(sha256sum "$PROMPT4K" | awk '{print $1}')"
echo "trial config: PP-2 0,1 + MEMRA_MOE_GROUPED=1 + drafter attached + specplace default (plain decode on PP-2) + MEMRA_API_KEY on + metering on"

##############################################################################
echo; echo "################ WINDOW 1: TTFT + decode ladder ################"
(
  flock -w 7200 9 || { echo "LOCK TIMEOUT W1"; exit 75; }
  trap stop_server EXIT
  echo "lock acquired $(date -u +%FT%TZ)"
  thermal
  boot_server "$RAW/server-w1-$TS.log" || exit 1

  echo; echo "########## TTFT short (228 tok) N=8, warmup 1 ##########"
  MEMRA_API_KEY=$KEY python3 $ROOT/probe.py --base $BASE --model step35 --shape short \
    --requests 8 --warmup 1 --max-tokens 8 --expect-prompt-tokens 228 \
    --label $LABEL --out $RAW/ttft-short-$TS.jsonl --timeout 600 || exit 1
  thermal

  echo; echo "########## TTFT 4k prompt N=5, warmup 1 ##########"
  MEMRA_API_KEY=$KEY python3 $ROOT/probe.py --base $BASE --model step35 --shape 4k \
    --prompt-file "$PROMPT4K" --requests 5 --warmup 1 --max-tokens 8 \
    --label $LABEL --out $RAW/ttft-4k-$TS.jsonl --timeout 600 || exit 1
  thermal

  echo; echo "########## TTFT cache-hit repeat (same prompt+salt) N=3 ##########"
  MEMRA_API_KEY=$KEY python3 $ROOT/hitprobe.py --base $BASE --model step35 \
    --prompt-file "$PROMPT4K" --requests 3 --max-tokens 8 \
    --label $LABEL --out $RAW/ttft-cachehit-$TS.jsonl --timeout 600 || exit 1
  thermal

  echo; echo "########## decode ladder c=1/2/4/8, N=3 points each, streamed ##########"
  for c in 1 2 4 8; do
    for rep in 1 2 3; do
      python3 $REPO/tools/load-serve.py --base $BASE --model step35 \
        --concurrency $c --stream --api-key $KEY --max-tokens 128 --warmup 1 \
        --label "decode-c${c}-rep${rep}" \
        --out $RAW/decode-ladder-$TS.jsonl \
        --per-request $RAW/decode-ladder-req-$TS.jsonl || exit 1
    done
    thermal
  done

  metrics_snap "$RAW/metrics-w1-final-$TS.json"
  stop_server
  sleep 3
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
  echo "lock released $(date -u +%FT%TZ)"
) 9>/tmp/memra-gpu.lock
rc1=$?
echo "=== WINDOW 1 rc=$rc1"
[[ $rc1 -ne 0 ]] && exit $rc1

##############################################################################
echo; echo "################ WINDOW 2: 10-min fleet-replay sustained load ################"
(
  flock -w 7200 9 || { echo "LOCK TIMEOUT W2"; exit 75; }
  trap stop_server EXIT
  echo "lock acquired $(date -u +%FT%TZ)"
  thermal
  # FRESH server: /metrics counters start from zero for the replay hit-ratio receipt.
  boot_server "$RAW/server-w2-$TS.log" || exit 1

  metrics_snap "$RAW/metrics-w2-t0-$TS.json"
  echo; echo "########## fleet-replay: 600s, 12 req/min Poisson, 12 sessions, 4 tenants ##########"
  MEMRA_API_KEY=$KEY stdbuf -oL -eL python3 $REPO/tools/fleet-replay.py \
    --base $BASE --model step35 --duration 600 --requests-per-minute 12 \
    --sessions 12 --tenants 4 --seed 20260808 --timeout 300 \
    > $RAW/replay-summary-$TS.json \
    2> >(awk '{ print strftime("%Y-%m-%dT%H:%M:%SZ", systime(), 1), $0; fflush() }' \
         > $RAW/replay-events-$TS.log)
  replay_rc=$?
  echo "fleet-replay exit=$replay_rc"
  cat $RAW/replay-summary-$TS.json
  metrics_snap "$RAW/metrics-w2-final-$TS.json"
  thermal
  echo "--- meter/admission lines in server log (counts) ---"
  grep -c '\[meter\] admit' "$RAW/server-w2-$TS.log" || true
  grep -ciE 'shed|429|defer|park' "$RAW/server-w2-$TS.log" || true
  stop_server
  sleep 3
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
  echo "lock released $(date -u +%FT%TZ)"
  exit $replay_rc
) 9>/tmp/memra-gpu.lock
rc2=$?
echo "=== WINDOW 2 rc=$rc2"
echo "=== RECEIPT MEASUREMENT DONE $(date -u +%FT%TZ) rc1=$rc1 rc2=$rc2"
} > "$MAIN" 2>&1
echo "log: $MAIN"
tail -30 "$MAIN"
