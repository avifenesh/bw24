#!/usr/bin/env bash
# pp2spec-crash STEP 8 — the full gate battery on the FIXED tree (4c72d637: fence on all
# three ppN bodies + sentinel traps). This is the lift-the-quarantine bar:
#   1. crash repro clean at c=4/c=8 over >=200 requests  (round 7: DONE, 204/204 on 7450928b;
#      re-run here on 4c72d637 since the batched/eager fences landed after)
#   2. run-spec K=1..8 PASS over PP-2 (+ acceptance == door-shut)
#   3. decode-batch ppspec mode green (bit-identity — the fence must not have moved bytes)
#   4. kernel-check ALL GREEN
#   5. single-card spec serve unregressed (door-shut run-spec + serve smoke)
# Receipts ~/receipts/pp2crash/gates. tee first, parse second.
set -uo pipefail
cd ~/memra
export PATH=$HOME/.cargo/bin:$PATH
OUT=~/receipts/pp2crash/gates
mkdir -p "$OUT"
Q9=/scratch-models/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
Q27=/scratch-models/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
BIN=target/release
ADDR=127.0.0.1:8123
BASE=http://$ADDR
FAILS=0

exec 9>/tmp/memra-gpu.lock
flock -w 14400 9 || { echo "FAIL: gpu lock timeout"; exit 1; }
echo "gpu lock acquired $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=index,memory.used,temperature.gpu --format=csv > "$OUT/gpu-pre.csv"

run() { local log="$OUT/$1"; shift
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  echo "=== $log: env[${envs[*]:-}] $*"
  if ! env "${envs[@]}" "$@" 2>&1 | tee "$log" >/dev/null; then echo "FAIL: $log"; FAILS=$((FAILS+1)); fi
}

wait_up() {
  for _ in $(seq 1 "$1"); do
    curl -sf "$BASE/v1/models" >/dev/null 2>&1 && return 0
    sleep 2
  done
  return 1
}

# ---- 4. kernel-check first (cheapest, catches build rot) ----
run kernel-check.log -- $BIN/kernel-check

# ---- 3. ppspec bit-identity, both placements + singledev (fence must not move bytes) ----
run ppspec-q9-dev01.log MEMRA_PP_DEVICES=0,1 -- \
  $BIN/decode-batch-gate "$Q9" --mode ppspec --stages 2 --steps 16 --ts 2,5,9 --reps 3
run ppspec-q9-dev10.log MEMRA_PP_DEVICES=1,0 -- \
  $BIN/decode-batch-gate "$Q9" --mode ppspec --stages 2 --steps 16 --ts 2,5,9 --reps 3
run ppspec-q9-singledev.log -- \
  $BIN/decode-batch-gate "$Q9" --mode ppspec --stages 2 --steps 16 --ts 2,5,9 --reps 3
# batched pp gate: the batched body took the fence too
run ppbatch-q9-dev01.log MEMRA_PP_DEVICES=0,1 -- \
  $BIN/decode-batch-gate "$Q9" --mode pp --stages 2 --steps 16 --batch 1,4,8 --reps 2

# ---- 2. run-spec K=1..8 over PP-2 + acceptance vs door-shut ----
DC0=MEMRA_QWEN_DC=0
run runspec-q9-dev10.log MEMRA_PP_DEVICES=1,0 MEMRA_PP_STAGES=2 $DC0 MEMRA_NGEN=64 -- \
  $BIN/run-spec "$Q9" 55
run runspec-q9-dev01.log MEMRA_PP_DEVICES=0,1 MEMRA_PP_STAGES=2 $DC0 MEMRA_NGEN=64 -- \
  $BIN/run-spec "$Q9" 55
run runspec-q9-doorshut-dc0.log $DC0 MEMRA_NGEN=64 -- $BIN/run-spec "$Q9" 55
# ---- 5a. single-card spec unregressed (naked door-shut run-spec) ----
run runspec-q9-doorshut.log MEMRA_NGEN=64 -- $BIN/run-spec "$Q9" 55
run run-gen-q9-naked.log MEMRA_NGEN=8 -- $BIN/run-gen "$Q9" 55

# ---- 1. crash gate on THIS build: c=4 x 100 + c=8 x 104, plus the trigger sequence ----
if curl -sf "$BASE/v1/models" >/dev/null 2>&1; then
  echo "FAIL: something already serving $ADDR"; FAILS=$((FAILS+1))
else
  env MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=1,0 MEMRA_PP2SPEC_UNQUARANTINE=1 \
    MEMRA_SPEC_GATE=0 MEMRA_MODELS="q9=$Q9" MEMRA_ADDR=$ADDR \
    $BIN/memra-server > "$OUT/crash-server.log" 2>&1 &
  PID=$!
  if ! wait_up 180; then echo "FAIL: crash-gate server never came up"; FAILS=$((FAILS+1)); kill $PID 2>/dev/null;
  else
    python3 tools/load-serve.py --base "$BASE" --model q9 --concurrency 2 \
      --requests 8 --max-tokens 96 --greedy --warmup 1 --label gate-c2 \
      --out "$OUT/crash-points.jsonl" > "$OUT/crash-c2.log" 2>&1
    python3 tools/load-serve.py --base "$BASE" --model q9 --concurrency 4 \
      --requests 100 --max-tokens 96 --greedy --warmup 0 --label gate-c4x100 \
      --out "$OUT/crash-points.jsonl" > "$OUT/crash-c4.log" 2>&1
    python3 tools/load-serve.py --base "$BASE" --model q9 --concurrency 8 \
      --requests 104 --max-tokens 96 --greedy --warmup 0 --label gate-c8x104 \
      --out "$OUT/crash-points.jsonl" > "$OUT/crash-c8.log" 2>&1
    kill $PID 2>/dev/null; wait $PID 2>/dev/null; sleep 4
    if grep -qi "sentinel\|illegal" "$OUT/crash-server.log"; then
      echo "FAIL: crash gate saw sentinel/illegal lines"; FAILS=$((FAILS+1))
    fi
  fi
fi

# ---- 5b. single-card spec serve smoke (door shut, spec ON default) ----
if curl -sf "$BASE/v1/models" >/dev/null 2>&1; then
  echo "FAIL: something already serving $ADDR (smoke)"; FAILS=$((FAILS+1))
else
  env MEMRA_MODELS="q9=$Q9" MEMRA_ADDR=$ADDR \
    $BIN/memra-server > "$OUT/smoke-server.log" 2>&1 &
  PID=$!
  if ! wait_up 180; then echo "FAIL: smoke server never came up"; FAILS=$((FAILS+1)); kill $PID 2>/dev/null;
  else
    python3 tools/load-serve.py --base "$BASE" --model q9 --concurrency 4 \
      --requests 16 --max-tokens 96 --greedy --warmup 1 --label smoke-c4 \
      --out "$OUT/crash-points.jsonl" > "$OUT/smoke-c4.log" 2>&1
    kill $PID 2>/dev/null; wait $PID 2>/dev/null; sleep 4
    grep -qi "sentinel\|illegal" "$OUT/smoke-server.log" \
      && { echo "FAIL: door-shut smoke saw sentinel/illegal"; FAILS=$((FAILS+1)); }
    grep -q "spec-acc" "$OUT/smoke-server.log" \
      || { echo "FAIL: door-shut smoke never ran spec"; FAILS=$((FAILS+1)); }
  fi
fi

nvidia-smi --query-gpu=index,memory.used --format=csv > "$OUT/gpu-post.csv"

echo; echo "==== verdicts ===="
grep -H "ppspec mode verdict\|pp mode verdict\|failing arm" $OUT/ppspec-*.log $OUT/ppbatch-*.log 2>/dev/null | sed "s|$OUT/||" | head -12
grep -H "SELF-CONSISTENCY\|acceptance:" $OUT/runspec-*.log 2>/dev/null | sed "s|$OUT/||"
grep -H "ALL GREEN\|FAIL" $OUT/kernel-check.log | tail -2 | sed "s|$OUT/||"
grep -H "MATCH\|MISMATCH" $OUT/run-gen-q9-naked.log | tail -2 | sed "s|$OUT/||"
python3 - <<'EOF'
import json
try:
    for l in open("/home/ubuntu/receipts/pp2crash/gates/crash-points.jsonl"):
        d = json.loads(l)
        print(d["label"], "ok", d["n_ok"], "err", d["n_err"], "agg", round(d["agg_tok_s"],1))
except FileNotFoundError:
    print("no crash points")
EOF
echo "script-detected failures: $FAILS"
echo PP2CRASH_GATES_DONE
exit $FAILS
