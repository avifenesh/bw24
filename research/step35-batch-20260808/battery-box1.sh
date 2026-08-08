#!/usr/bin/env bash
# lane/step35-batched-decode: the full gate battery + c-scaling perf on box1 (2x PRO 6000,
# PP-2, flock discipline — one window, release promptly).
#
# Order (correctness first, perf last, one lock hold per phase):
#   G1 kernel-check ALL GREEN (no kernel touched — proven, not assumed)
#   G2 decode-batch-gate --mode pp on step35, B=1,2,4,8 x reps — the bit-identity battery:
#      split vs unsplit batched walks over the sharded placement (the new arm on BOTH sides)
#   G3 run-gen argmax MATCH (prefill/decode + batched-prime/tokenwise)
#   G4 run-spec K=1..8 self-consistency + acceptance == baseline (drafter attached)
#   G5 chunkinv35 + tickinv35 no-regress (prefill segmentation — untouched, prove it)
#   G6 b2geo35 naked (must be GREEN now) + b2geo35c canary (must have teeth)
#   P1 decode aggregate c=1/2/4/8, N=3 per c, DEFAULT batched serve — vs the 34-flat baseline
#
# Run ON BOX1: bash research/step35-batch-20260808/battery-box1.sh
set -uo pipefail
export PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
cd "$HOME/stepbatch-memra"
M=$HOME/step37/models/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf
D=$HOME/step37/models/step-3.7-flash/Step3.7-flash-mtp-Q8_0.gguf
RAW=$HOME/step37/raw-stepbatch; mkdir -p "$RAW"
TS=$(date -u +%Y%m%dT%H%M%SZ); LOG=$RAW/battery-$TS.log
PORT=8095
BASE=http://127.0.0.1:$PORT

thermal() { nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm,memory.used --format=csv,noheader; }

{
echo "=== step35-batch battery $TS tree=$(git rev-parse --short HEAD 2>/dev/null || echo rsync)"
(
  flock -w 7200 9 || { echo "LOCK TIMEOUT"; exit 75; }
  echo "lock acquired $(date -u +%FT%TZ)"; thermal

  echo; echo "########## G1: kernel-check ##########"
  timeout 1800 ./target/release/kernel-check > "$RAW/kernel-check-$TS.log" 2>&1
  RC=$?
  grep -c FAIL "$RAW/kernel-check-$TS.log" | xargs -I{} echo "kernel-check FAIL-lines={} exit=$RC"
  tail -2 "$RAW/kernel-check-$TS.log"

  echo; echo "########## G2: decode-batch-gate --mode pp, step35, B=1,2,4,8 ##########"
  MEMRA_PP_DEVICES=0,1 timeout 5400 ./target/release/decode-batch-gate "$M" \
    --mode pp --batch 1,2,4,8 --steps 24 --reps 2 --stages 2 \
    > "$RAW/dbg-pp-step35-$TS.log" 2>&1
  echo "decode-batch-gate exit=$?"
  grep -E "pp mode verdict|BIT-IDENTICAL|FAIL|differing" "$RAW/dbg-pp-step35-$TS.log" | tail -20
  thermal

  echo; echo "########## G3: run-gen argmax ##########"
  MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_NGEN=64 \
    MEMRA_PROMPT="Write a short paragraph explaining how a CPU pipeline improves instruction throughput, and mention one hazard." \
    timeout 1800 ./target/release/run-gen "$M" > "$RAW/rungen-$TS.log" 2>&1
  echo "run-gen exit=$?"
  grep -E "argmax|MATCH|MISMATCH" "$RAW/rungen-$TS.log"

  echo; echo "########## G4: run-spec K=1..8 with drafter (full sweep in one process) ##########"
  MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_MTP_DRAFT="$D" MEMRA_NGEN=32 \
    MEMRA_PROMPT="Write a short paragraph explaining how a CPU pipeline improves instruction throughput, and mention one hazard." \
    timeout 3600 ./target/release/run-spec "$M" > "$RAW/runspec-$TS.log" 2>&1
  echo "run-spec exit=$?"
  grep -E "acceptance|self-consistency|SELF-CONSISTENCY" "$RAW/runspec-$TS.log"

  echo; echo "########## G5: chunkinv35 + tickinv35 ##########"
  MEMRA_STEP37_GGUF="$M" MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 timeout 3600 \
    bash tools/chunk-invariance-gate.sh --label step35-swa \
      --prompts research/chunk-invariance-20260805/prompt-pp6257.txt \
      --chunks 4096,513,512,256,64 --seam MEMRA_STEP35_SWA_TKV --steps 24 \
      > "$RAW/chunkinv35-$TS.log" 2>&1
  echo "chunkinv35 exit=$? ($(grep -m1 -oE 'CHUNK-(IN)?VARIANT' "$RAW/chunkinv35-$TS.log" | head -1))"
  MEMRA_STEP37_GGUF="$M" MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 timeout 3600 \
    bash tools/tick-invariance-gate.sh --label step35-tick \
      --prompts research/chunk-invariance-20260805/prompt-pp6257.txt \
      --budgets 0,1024,513,512,256,64 --splits 64,256,512 \
      --seam MEMRA_PRIME_CALLLOCAL --steps 24 \
      > "$RAW/tickinv35-$TS.log" 2>&1
  echo "tickinv35 exit=$? ($(grep -m1 -oE 'TICK-(IN)?VARIANT|INVARIANT' "$RAW/tickinv35-$TS.log" | head -1))"
  thermal
  echo "lock released $(date -u +%FT%TZ)"
) 9>/tmp/memra-gpu.lock
echo "=== gates rc=$?"

# G6 + P1 take the lock themselves (the b2geo gate scripts hold their own window)
echo; echo "########## G6: b2geo35 naked (GREEN expected) ##########"
MEMRA_STEP37_GGUF="$M" bash tools/step35-b2-geometry-gate.sh --port $PORT
echo "b2geo35 exit=$?"
echo; echo "########## G6c: b2geo35 canary (teeth) ##########"
MEMRA_STEP37_GGUF="$M" bash tools/step35-b2-geometry-gate.sh --canary --port $PORT
echo "b2geo35c exit=$?"

echo; echo "########## P1: decode aggregate c-sweep, DEFAULT batched serve, N=3 ##########"
(
  flock -w 7200 9 || { echo "LOCK TIMEOUT"; exit 75; }
  echo "lock acquired $(date -u +%FT%TZ)"; thermal
  env MEMRA_MODELS="step35=${M}+${D}" MEMRA_SERVE_SPEC=0 \
      MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_ADDR=127.0.0.1:$PORT \
      ./target/release/memra-server > "$RAW/perf-server-$TS.log" 2>&1 &
  SRV=$!
  trap 'kill $SRV 2>/dev/null; wait $SRV 2>/dev/null' EXIT
  for i in $(seq 1 120); do
    sleep 5; curl -sf "$BASE/readyz" >/dev/null 2>&1 && break
    kill -0 $SRV 2>/dev/null || { echo SERVER DIED; exit 1; }
  done
  grep -m1 "decode chunk cap" "$RAW/perf-server-$TS.log" || true
  for c in 1 2 4 8; do
    for rep in 1 2 3; do
      echo "--- P1 c=$c rep=$rep ---"
      python3 tools/load-serve.py --base "$BASE" --model step35 --concurrency "$c" \
        --requests $((4 * c)) --max-tokens 128 --warmup 1 \
        --label "sb-c${c}-r${rep}" --out "$RAW/perf-points-$TS.jsonl" --timeout 1800
      thermal
    done
  done
  kill $SRV; wait $SRV 2>/dev/null; trap - EXIT
  thermal
  echo "lock released $(date -u +%FT%TZ)"
) 9>/tmp/memra-gpu.lock
echo "=== perf rc=$?"
echo "=== done $(date -u +%FT%TZ)"
} > "$LOG" 2>&1
echo "LOG=$LOG"
