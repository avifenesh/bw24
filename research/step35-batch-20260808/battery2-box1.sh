#!/usr/bin/env bash
# Battery round 2 on box1 — only what round 1 could not deliver:
#   G1 kernel-check (round-1 binary carried a stale fatbin: `fa_prefill_qw_db_w_hd128 not in
#      any fatbin` — the target dir was seeded from another lane's cache and the .cu files'
#      mtimes predated it, so nvcc never reran. Fixed by touch + rebuild; fresh binary
#      verified to carry the kernel string.)
#   G2 decode-batch-gate --mode pp --plen 520 (same stale-fatbin panic)
#   G6/G6c b2geo35 naked + canary (round-1 run hung on a bare `wait` that included the
#      server job — fully-correct c=2 responses were already on disk; script fixed)
#   P1 c-sweep perf (never reached)
# Round-1 keepers (battery-20260808T034540Z.log): run-gen MATCH (both argmax gates),
# run-spec 8/8 PASS acceptance digit-identical to baseline, chunkinv35 CHUNK-INVARIANT.
# tickinv35 FAIL is PRE-EXISTING: the fix lives in lane/tick-seg (f01710ca), NOT an
# ancestor of this lane's base a131e8c7 — verified by git ancestry, not re-derived.
set -uo pipefail
export PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
cd "$HOME/stepbatch-memra"
M=$HOME/step37/models/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf
D=$HOME/step37/models/step-3.7-flash/Step3.7-flash-mtp-Q8_0.gguf
RAW=$HOME/step37/raw-stepbatch; mkdir -p "$RAW"
TS=$(date -u +%Y%m%dT%H%M%SZ); LOG=$RAW/battery2-$TS.log
PORT=8095
BASE=http://127.0.0.1:$PORT

thermal() { nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm,memory.used --format=csv,noheader; }

{
echo "=== step35-batch battery2 $TS"
(
  flock -w 28800 9 || { echo "LOCK TIMEOUT"; exit 75; }
  echo "lock acquired $(date -u +%FT%TZ)"; thermal

  echo; echo "########## G1: kernel-check (fresh fatbin) ##########"
  timeout 2400 ./target/release/kernel-check > "$RAW/kernel-check-$TS.log" 2>&1
  RC=$?
  echo "kernel-check exit=$RC FAIL-lines=$(grep -c FAIL "$RAW/kernel-check-$TS.log")"
  tail -2 "$RAW/kernel-check-$TS.log"

  echo; echo "########## G2: decode-batch-gate --mode pp, step35, B=1,2,4,8, plen 520 ##########"
  MEMRA_PP_DEVICES=0,1 timeout 10800 ./target/release/decode-batch-gate "$M" \
    --mode pp --batch 1,2,4,8 --steps 24 --reps 2 --stages 2 --plen 520 \
    > "$RAW/dbg-pp-step35-$TS.log" 2>&1
  echo "decode-batch-gate exit=$?"
  grep -E "pp mode verdict|BIT-IDENTICAL|FAIL|differing bits" "$RAW/dbg-pp-step35-$TS.log" | tail -24
  thermal
  echo "lock released $(date -u +%FT%TZ)"
) 9>/tmp/memra-gpu.lock
echo "=== gates2 rc=$?"

echo; echo "########## G6: b2geo35 naked (GREEN expected) ##########"
MEMRA_STEP37_GGUF="$M" bash tools/step35-b2-geometry-gate.sh --port $PORT
echo "b2geo35 exit=$?"
echo; echo "########## G6c: b2geo35 canary (teeth) ##########"
MEMRA_STEP37_GGUF="$M" bash tools/step35-b2-geometry-gate.sh --canary --port $PORT
echo "b2geo35c exit=$?"

echo; echo "########## P1: decode aggregate c-sweep, DEFAULT batched serve, N=3 ##########"
(
  flock -w 28800 9 || { echo "LOCK TIMEOUT"; exit 75; }
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
