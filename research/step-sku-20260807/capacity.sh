#!/usr/bin/env bash
# step-sku item 4: capacity/perf receipts for the Step-3.7-Flash listing ($/day shape).
# Research evidence only — goes in research/, NOT the public board (deployment bar: no
# listing before head-to-head).
#
# Cells (one flock window, thermal regime sampled per cell):
#   P1  pp rate, 4k prefill: concat-prime-probe ppprime, prompt-pp4096 (exactly 4096 ids,
#       cut with the parity-proven HF tokenizer), N=5 reps after 1 warmup, median stated.
#   D0  batched-mode probe at c=2: EXPECTED REFUSAL receipt — decode_step_batch has no
#       step35 arm at B>1 (decode.rs:2833 full_attn_decode_batched refuses; per-layer
#       n_head / partial rope / SWA offset view). The quoted error is the receipt that
#       c>1 rides the legacy path below, not an assumption.
#   D1  decode tok/s at c=1/2/4/8: MEMRA_SERVE_BATCH=0 (legacy round-robin eager — the
#       only multi-session path with a step35 arm today; MAX_ACTIVE=4 caps concurrent
#       actives, so c=8 = 4 active + 4 queued, which is the honest served shape). Spec
#       OFF per #87 (MEMRA_SERVE_SPEC=0), drafter attached (the served config per
#       lane/step-draft arm G). N=3 load points per c, medians reported.
#   T1  TTFT short-turn: load-serve --stream at c=1, greedy, N=8 requests, p50/p95.
#
# Run ON THE BOX: bash capacity.sh   (takes the flock itself)
set -uo pipefail
export PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
cd "$HOME/tokparity-memra"
M=$HOME/step37/models/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf
D=$HOME/step37/models/step-3.7-flash/Step3.7-flash-mtp-Q8_0.gguf
P4096=$HOME/step37/prompt-pp4096.txt
RAW=$HOME/step37/raw; mkdir -p "$RAW"
TS=$(date -u +%Y%m%dT%H%M%SZ); LOG=$RAW/capacity-$TS.log
PORT=8092
BASE=http://127.0.0.1:$PORT

thermal() { nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm,memory.used --format=csv,noheader; }

boot_server() { # boot_server <extra-env...>
  env MEMRA_MODELS="step35=${M}+${D}" MEMRA_SERVE_SPEC=0 \
      MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_ADDR=127.0.0.1:$PORT "$@" \
      ./target/release/memra-server > "$RAW/capacity-server-$TS-$1.log" 2>&1 &
  SRV=$!
  for i in $(seq 1 120); do
    sleep 5
    curl -sf "$BASE/readyz" >/dev/null 2>&1 && { echo "server ready after ~$((i*5))s"; return 0; }
    kill -0 $SRV 2>/dev/null || { echo "SERVER DIED"; tail -20 "$RAW"/capacity-server-$TS-*.log; return 1; }
  done
  echo "server never became ready"; return 1
}
stop_server() { kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; sleep 2; }

{
echo "=== step-sku capacity $TS ==="
(
  flock -w 7200 9 || { echo "LOCK TIMEOUT"; exit 75; }
  echo "lock acquired $(date -u +%FT%TZ)"
  thermal

  echo; echo "########## P1: prefill rate, pp4096, ppprime N=5 (median of 5 timed reps, 1 warmup) ##########"
  MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 timeout 3600 \
    ./target/release/concat-prime-probe "$M" ppprime --prompt-a "@$P4096" --reps 5 --warmup 1
  echo "ppprime exit=$?"
  thermal

  echo; echo "########## D0: batched-mode c=2 probe (EXPECTED step35 B>1 refusal receipt) ##########"
  boot_server MEMRA_TAG=batched || exit 1
  python3 tools/load-serve.py --base "$BASE" --model step35 --concurrency 2 --requests 4 \
    --max-tokens 32 --greedy --warmup 0 --label cap-d0-batched-c2 --timeout 300
  echo "d0 exit=$?"
  grep -m2 "batch step\|no batched" "$RAW/capacity-server-$TS-MEMRA_TAG=batched.log" || echo "(no refusal line captured)"
  stop_server

  echo; echo "########## D1: decode c-sweep, MEMRA_SERVE_BATCH=0 legacy eager, spec OFF, N=3 per c ##########"
  boot_server MEMRA_SERVE_BATCH=0 || exit 1
  for c in 1 2 4 8; do
    for rep in 1 2 3; do
      echo "--- D1 c=$c rep=$rep ---"
      python3 tools/load-serve.py --base "$BASE" --model step35 --concurrency "$c" \
        --requests $((4 * c)) --max-tokens 128 --warmup 1 \
        --label "cap-d1-c${c}-r${rep}" --out "$RAW/capacity-points-$TS.jsonl" --timeout 1200
      thermal
    done
  done

  echo; echo "########## T1: TTFT short-turn, streaming, c=1 greedy, N=8 ##########"
  python3 tools/load-serve.py --base "$BASE" --model step35 --concurrency 1 --requests 8 \
    --max-tokens 32 --greedy --stream --warmup 1 \
    --label cap-t1-ttft --out "$RAW/capacity-points-$TS.jsonl" \
    --per-request "$RAW/capacity-ttft-$TS.jsonl" --timeout 600
  echo "t1 exit=$?"
  stop_server

  thermal
  echo "lock released $(date -u +%FT%TZ)"
) 9>/tmp/memra-gpu.lock
echo "=== capacity rc=$?"
echo "=== done $(date -u +%FT%TZ)"
} > "$LOG" 2>&1
echo "LOG=$LOG"
