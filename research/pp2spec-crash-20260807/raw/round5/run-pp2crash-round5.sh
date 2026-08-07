#!/usr/bin/env bash
# pp2spec-crash STEP 5 — settle the SENTINEL-TOKEN hypothesis + the cross-stream-race class.
# Arithmetic from rounds 2/3: fault VA 0x484_C6B3C500 = base 0x4_C6B3_CE00 + 0x7fffffff*2304,
# where 2304 = NVFP4 row_bytes at n_embd=4096 and 0x7fffffff is argmax_partial/final_f32's
# init sentinel (kernels.cu:129/166) — unbeaten only when EVERY logit is NaN/-inf. One NaN in
# the draft-head input NaNs the whole row. So: NaN h_seed -> argmax sentinel -> embed row
# 2^31-1 -> MMU fault. Two probes:
#  G (fulldump): FULL coredump (param+global) -> quote token_d[0] and the embd param.
#  H (evt):      MEMRA_EVT=1 re-enables cudarc cross-stream event tracking in EVERY engine.
#                Clean => the NaN producer is a cross-stream WAR/UAF that tracking guards.
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
flock -w 14400 9 || { echo "FAIL: gpu lock timeout"; exit 1; }
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

echo "=== ARM G: FULL coredump, exact A sequence ==="
rm -f "$OUT"/core-full-*.nvcudmp
env MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=1,0 MEMRA_PP2SPEC_UNQUARANTINE=1 \
  MEMRA_SPEC_GATE=0 MEMRA_MODELS="q9=$Q9" MEMRA_ADDR=$ADDR \
  CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1 \
  CUDA_COREDUMP_FILE="$OUT/core-full-%p.nvcudmp" \
  $BIN/memra-server > "$OUT/G-server.log" 2>&1 &
PID=$!
if ! wait_up 180; then echo "FAIL: G server never came up"; tail -20 "$OUT/G-server.log"; kill $PID 2>/dev/null; exit 1; fi
python3 tools/load-serve.py --base "$BASE" --model q9 --concurrency 2 \
  --requests 8 --max-tokens 96 --greedy --warmup 1 --label G-c2 \
  --out "$OUT/G-points.jsonl" > "$OUT/G-c2.log" 2>&1
python3 tools/load-serve.py --base "$BASE" --model q9 --concurrency 4 \
  --requests 16 --max-tokens 96 --greedy --warmup 0 --label G-c4 \
  --out "$OUT/G-points.jsonl" > "$OUT/G-c4.log" 2>&1
# full dump of ~6-10GB VRAM takes a while; wait for the file size to settle
for _ in $(seq 1 90); do
  sz1=$(stat -c%s "$OUT"/core-full-*.nvcudmp 2>/dev/null | head -1 || echo 0)
  sleep 10
  sz2=$(stat -c%s "$OUT"/core-full-*.nvcudmp 2>/dev/null | head -1 || echo 0)
  [ -n "$sz1" ] && [ "$sz1" = "$sz2" ] && [ "$sz1" != "0" ] && break
done
kill $PID 2>/dev/null; wait $PID 2>/dev/null; sleep 4
grep -n -i "illegal\|abort" "$OUT/G-server.log" | head -3
CORE=$(ls -t "$OUT"/core-full-*.nvcudmp 2>/dev/null | head -1)
if [ -n "${CORE:-}" ]; then
  echo "=== cuda-gdb operand readout: $CORE ==="
  /usr/local/cuda-13.2/bin/cuda-gdb --batch \
    -ex "target cudacore $CORE" \
    -ex "info cuda kernels" \
    -ex "echo \n--- raw params @0x0 (embd, token_d, x_out, n_embd|qtype, row_bytes) ---\n" \
    -ex "x/5gx (@parameter unsigned long long*)0x0" \
    -ex "set \$embd = *((@parameter unsigned long long*)0x0)" \
    -ex "set \$tokp = *((@parameter unsigned long long*)0x8)" \
    -ex "printf \"embd=%#lx token_d=%#lx\\n\", \$embd, \$tokp" \
    -ex "echo \n--- token_d[0] (the token id the gather dereferenced) ---\n" \
    -ex "x/1wx \$tokp" \
    -ex "echo \n--- embd base readable? ---\n" \
    -ex "x/2gx \$embd" \
    > "$OUT/G-cudagdb-operands.log" 2>&1
  cat "$OUT/G-cudagdb-operands.log"
else
  echo "G: NO DUMP PRODUCED (run did not fault?)"
fi

echo "=== ARM H: MEMRA_EVT=1 (cross-stream event tracking ON), exact A sequence x2 reps ==="
for rep in 1 2; do
  env MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=1,0 MEMRA_PP2SPEC_UNQUARANTINE=1 \
    MEMRA_SPEC_GATE=0 MEMRA_EVT=1 MEMRA_MODELS="q9=$Q9" MEMRA_ADDR=$ADDR \
    $BIN/memra-server > "$OUT/H${rep}-server.log" 2>&1 &
  PID=$!
  if ! wait_up 180; then echo "FAIL: H$rep server never came up"; tail -20 "$OUT/H${rep}-server.log"; kill $PID 2>/dev/null; exit 1; fi
  python3 tools/load-serve.py --base "$BASE" --model q9 --concurrency 2 \
    --requests 8 --max-tokens 96 --greedy --warmup 1 --label H${rep}-c2 \
    --out "$OUT/H-points.jsonl" > "$OUT/H${rep}-c2.log" 2>&1
  python3 tools/load-serve.py --base "$BASE" --model q9 --concurrency 4 \
    --requests 16 --max-tokens 96 --greedy --warmup 0 --label H${rep}-c4 \
    --out "$OUT/H-points.jsonl" > "$OUT/H${rep}-c4.log" 2>&1
  kill $PID 2>/dev/null; wait $PID 2>/dev/null; sleep 4
  echo "--- H$rep hits ---"
  grep -n -i "illegal\|abort\|alloc failed" "$OUT/H${rep}-server.log" | head -4 || echo "(H$rep clean)"
done
nvidia-smi --query-gpu=index,memory.used --format=csv > "$OUT/GH-gpu-post.csv"
echo PP2CRASH_ROUND5_DONE
