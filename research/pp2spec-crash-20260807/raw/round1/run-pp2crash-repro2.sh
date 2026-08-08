#!/usr/bin/env bash
# pp2spec-crash STEP 1 (v2) — repro + localization of the sticky ILLEGAL_ADDRESS (task #87).
# Tree: ~/memra @ 3f56c8ce (lane/pp2spec-crash: quarantine + diagnostic door + spec-gate #89).
# The quarantine now refuses this regime at parse time; MEMRA_PP2SPEC_UNQUARANTINE=1 is the
# diagnostic door added for exactly this battery.
# NOTE the spec-gate (#89, merged since the finding lane): at c>=4 new arrivals are routed to
# batched decode, which would HIDE the trigger (two live spec sessions). MEMRA_SPEC_GATE=0
# restores always-spec so the repro hits the same path the finding lane measured.
# A: bare repro confirm (dev10, spec ON, c=2 then c=4) — expect the quoted illegal address.
# L: CUDA_LAUNCH_BLOCKING=1 — synchronous launches so the erroring call IS the faulting kernel.
# B: compute-sanitizer memcheck, SPEC_NOGRAPH=1 (graph exonerated by F2; eager draft gives
#    memcheck clean attribution), c=2 (the minimal trigger: 15/24 lost at c=2 in F4).
set -uo pipefail
cd ~/memra
export PATH=$HOME/.cargo/bin:$PATH
OUT=~/receipts/pp2crash
mkdir -p "$OUT"
Q9=/scratch-models/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
BIN=target/release
ADDR=127.0.0.1:8123
BASE=http://$ADDR
SAN=/usr/local/cuda-13.2/bin/compute-sanitizer

COMMON_ENV=(MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=1,0 MEMRA_PP2SPEC_UNQUARANTINE=1
            MEMRA_SPEC_GATE=0 MEMRA_MODELS="q9=$Q9" MEMRA_ADDR=$ADDR)

exec 9>/tmp/memra-gpu.lock
flock -w 1800 9 || { echo "FAIL: gpu lock timeout"; exit 1; }
echo "gpu lock acquired $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=index,memory.used,temperature.gpu --format=csv > "$OUT/gpu-pre.csv"

wait_up() { # $1 = tries
  for _ in $(seq 1 "$1"); do
    curl -sf "$BASE/v1/models" >/dev/null 2>&1 && return 0
    sleep 2
  done
  return 1
}

stop_srv() {
  kill "$1" 2>/dev/null; wait "$1" 2>/dev/null; sleep 4
}

if curl -sf "$BASE/v1/models" >/dev/null 2>&1; then
  echo "FAIL: something already serving $ADDR"; exit 1
fi

echo "=== PHASE A: bare repro (dev10 spec ON, c=2 then c=4) ==="
env "${COMMON_ENV[@]}" $BIN/memra-server > "$OUT/A-server.log" 2>&1 &
PID=$!
if ! wait_up 180; then echo "FAIL: phase A server never came up"; tail -20 "$OUT/A-server.log"; stop_srv $PID; exit 1; fi
python3 tools/load-serve.py --base "$BASE" --model q9 --concurrency 2 \
  --requests 8 --max-tokens 96 --greedy --warmup 1 --label A-c2 \
  --out "$OUT/A-points.jsonl" > "$OUT/A-c2.log" 2>&1
python3 tools/load-serve.py --base "$BASE" --model q9 --concurrency 4 \
  --requests 16 --max-tokens 96 --greedy --warmup 0 --label A-c4 \
  --out "$OUT/A-points.jsonl" > "$OUT/A-c4.log" 2>&1
stop_srv $PID
echo "--- phase A server log hits ---"
grep -n -i "illegal\|panic\|step error\|alloc failed" "$OUT/A-server.log" | head -20

echo "=== PHASE L: CUDA_LAUNCH_BLOCKING=1 (dev10 spec ON, c=2 then c=4) ==="
env CUDA_LAUNCH_BLOCKING=1 "${COMMON_ENV[@]}" $BIN/memra-server > "$OUT/L-server.log" 2>&1 &
PID=$!
if ! wait_up 240; then echo "FAIL: phase L server never came up"; tail -20 "$OUT/L-server.log"; stop_srv $PID; exit 1; fi
python3 tools/load-serve.py --base "$BASE" --model q9 --concurrency 2 \
  --requests 12 --max-tokens 96 --greedy --warmup 1 --label L-c2 \
  --out "$OUT/L-points.jsonl" > "$OUT/L-c2.log" 2>&1
python3 tools/load-serve.py --base "$BASE" --model q9 --concurrency 4 \
  --requests 16 --max-tokens 96 --greedy --warmup 0 --label L-c4 \
  --out "$OUT/L-points.jsonl" > "$OUT/L-c4.log" 2>&1
stop_srv $PID
echo "--- phase L server log hits ---"
grep -n -i "illegal\|panic\|step error\|alloc failed" "$OUT/L-server.log" | head -30

echo "=== PHASE B: compute-sanitizer memcheck (dev10 spec ON NOGRAPH, c=2 then c=4) ==="
env MEMRA_SPEC_NOGRAPH=1 "${COMMON_ENV[@]}" \
  $SAN --tool memcheck --print-limit 40 --error-exitcode 66 \
  $BIN/memra-server > "$OUT/B-server-sanitizer.log" 2>&1 &
PID=$!
if ! wait_up 600; then echo "FAIL: phase B server never came up (sanitizer)"; tail -30 "$OUT/B-server-sanitizer.log"; stop_srv $PID; exit 1; fi
python3 tools/load-serve.py --base "$BASE" --model q9 --concurrency 2 \
  --requests 8 --max-tokens 48 --greedy --warmup 0 --label B-c2-san \
  --out "$OUT/B-points.jsonl" > "$OUT/B-c2.log" 2>&1
python3 tools/load-serve.py --base "$BASE" --model q9 --concurrency 4 \
  --requests 8 --max-tokens 48 --greedy --warmup 0 --label B-c4-san \
  --out "$OUT/B-points.jsonl" > "$OUT/B-c4.log" 2>&1
sleep 15
stop_srv $PID
echo "--- sanitizer findings ---"
grep -n -B2 -A 14 "Invalid \|ERROR SUMMARY\|Program hit" "$OUT/B-server-sanitizer.log" | head -200
nvidia-smi --query-gpu=index,memory.used,temperature.gpu --format=csv > "$OUT/gpu-post.csv"
echo PP2CRASH_REPRO_DONE
