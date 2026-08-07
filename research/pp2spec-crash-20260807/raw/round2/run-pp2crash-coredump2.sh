#!/usr/bin/env bash
# pp2spec-crash STEP 2b — coredump arm, EXACT phase-A sequence.
# Phase C (c=4-only, coredump on) was 16/16 CLEAN — but in every reproducing run (round-1 A,
# finding-lane F4) the first fault fired during the c=2-with-warmup phase; c=4 merely
# inherited the poisoned context. So the coredump arm must replay A verbatim:
#   warmup 1 + c=2 x8 (max_tokens 96), then c=4 x16 — same knobs, coredump env added.
# If this is ALSO clean twice in a row, the coredump env itself masks the race (like L/B)
# and the localization pivot is instrumented builds, not driver tooling.
set -uo pipefail
cd ~/memra
export PATH=$HOME/.cargo/bin:$PATH
OUT=~/receipts/pp2crash
mkdir -p "$OUT"
Q9=/scratch-models/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
BIN=target/release
ADDR=127.0.0.1:8123
BASE=http://$ADDR

exec 9>/tmp/memra-gpu.lock
flock -w 1800 9 || { echo "FAIL: gpu lock timeout"; exit 1; }
echo "gpu lock acquired $(date -u +%FT%TZ)"

wait_up() {
  for _ in $(seq 1 "$1"); do
    curl -sf "$BASE/v1/models" >/dev/null 2>&1 && return 0
    sleep 2
  done
  return 1
}

if curl -sf "$BASE/v1/models" >/dev/null 2>&1; then
  echo "FAIL: something already serving $ADDR"; exit 1
fi

for rep in 1 2; do
  echo "=== PHASE D rep $rep: coredump env, EXACT A sequence (c=2+warmup, then c=4) ==="
  env MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=1,0 MEMRA_PP2SPEC_UNQUARANTINE=1 \
    MEMRA_SPEC_GATE=0 MEMRA_MODELS="q9=$Q9" MEMRA_ADDR=$ADDR \
    CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1 \
    CUDA_ENABLE_LIGHTWEIGHT_COREDUMP=1 \
    CUDA_COREDUMP_FILE="$OUT/core-r${rep}-%p.nvcudmp" \
    $BIN/memra-server > "$OUT/D${rep}-server.log" 2>&1 &
  PID=$!
  if ! wait_up 180; then echo "FAIL: rep $rep server never came up"; tail -20 "$OUT/D${rep}-server.log"; kill $PID 2>/dev/null; exit 1; fi
  python3 tools/load-serve.py --base "$BASE" --model q9 --concurrency 2 \
    --requests 8 --max-tokens 96 --greedy --warmup 1 --label D${rep}-c2 \
    --out "$OUT/D-points.jsonl" > "$OUT/D${rep}-c2.log" 2>&1
  python3 tools/load-serve.py --base "$BASE" --model q9 --concurrency 4 \
    --requests 16 --max-tokens 96 --greedy --warmup 0 --label D${rep}-c4 \
    --out "$OUT/D-points.jsonl" > "$OUT/D${rep}-c4.log" 2>&1
  sleep 20
  kill $PID 2>/dev/null; wait $PID 2>/dev/null; sleep 4
  echo "--- rep $rep hits ---"
  grep -n -i "illegal\|panic\|alloc failed" "$OUT/D${rep}-server.log" | head -6
  ls -la "$OUT"/core-r${rep}-*.nvcudmp 2>/dev/null && break
done

for core in "$OUT"/core-*.nvcudmp; do
  [ -e "$core" ] || continue
  echo "=== cuda-gdb analysis: $core ==="
  /usr/local/cuda-13.2/bin/cuda-gdb --batch \
    -ex "target cudacore $core" \
    -ex "info cuda kernels" \
    -ex "bt" \
    > "$OUT/D-cudagdb-$(basename "$core" .nvcudmp).log" 2>&1
  head -80 "$OUT/D-cudagdb-$(basename "$core" .nvcudmp).log"
done
nvidia-smi --query-gpu=index,memory.used --format=csv > "$OUT/D-gpu-post.csv"
echo PP2CRASH_COREDUMP2_DONE
