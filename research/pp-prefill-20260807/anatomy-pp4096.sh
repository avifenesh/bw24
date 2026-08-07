#!/usr/bin/env bash
# pp-prefill lane, step 1: ANATOMY of the 90.9 tok/s pp4096 prefill over PP-2.
# nsys profile of concat-prime-probe ppprime (1 warmup + 1 timed rep) — the exact receipt
# config from research/step-sku-20260807 item 4 (capacity.sh P1).
# .nsys-rep stays in /tmp (NEVER committed); numbers extracted to $RAW logs.
set -uo pipefail
export PATH=/opt/nvidia/nsight-systems/2026.1.3/bin:/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
cd "$HOME/tokparity-memra"
M=$HOME/step37/models/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf
P4096=$HOME/step37/prompt-pp4096.txt
RAW=$HOME/ppserve-raw; mkdir -p "$RAW"
TS=$(date -u +%Y%m%dT%H%M%SZ); LOG=$RAW/anatomy-$TS.log
REP=/tmp/anatomy-pp4096-$TS   # .nsys-rep in /tmp ONLY

{
echo "=== pp-prefill anatomy $TS ==="
nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm,memory.used --format=csv,noheader
(
  flock -w 7200 9 || { echo "LOCK TIMEOUT"; exit 75; }
  echo "lock acquired $(date -u +%FT%TZ)"

  echo "--- arm 1: nsys-traced ppprime (warmup 1 + reps 1) ---"
  MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 timeout 3600 \
    nsys profile -o "$REP" --trace=cuda --sample=none --cpuctxsw=none \
      --cuda-memory-usage=false --force-overwrite=true \
      ./target/release/concat-prime-probe "$M" ppprime --prompt-a "@$P4096" --reps 1 --warmup 1
  echo "nsys exit=$?"
  nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm,memory.used --format=csv,noheader

  echo "--- arm 2: untraced control, same window (nsys overhead check), reps 2 ---"
  MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 timeout 1200 \
    ./target/release/concat-prime-probe "$M" ppprime --prompt-a "@$P4096" --reps 2 --warmup 0
  echo "control exit=$?"

  echo "lock released $(date -u +%FT%TZ)"
) 9>/tmp/memra-gpu.lock
echo "=== window rc=$?"

echo "--- nsys stats: kernel summary ---"
nsys stats --report cuda_gpu_kern_sum --format csv "$REP.nsys-rep" > "$RAW/anatomy-$TS-kernsum.csv" 2>&1
head -40 "$RAW/anatomy-$TS-kernsum.csv"
echo "--- nsys stats: api summary ---"
nsys stats --report cuda_api_sum --format csv "$REP.nsys-rep" > "$RAW/anatomy-$TS-apisum.csv" 2>&1
head -25 "$RAW/anatomy-$TS-apisum.csv"
echo "--- nsys stats: memop summary ---"
nsys stats --report cuda_gpu_mem_time_sum --format csv "$REP.nsys-rep" > "$RAW/anatomy-$TS-memsum.csv" 2>&1
cat "$RAW/anatomy-$TS-memsum.csv"
echo "--- gpu trace gaps: total GPU-busy per device ---"
nsys stats --report cuda_gpu_sum --format csv "$REP.nsys-rep" > "$RAW/anatomy-$TS-gpusum.csv" 2>&1 || true
ls -la "$REP".* 
echo "=== done $(date -u +%FT%TZ)"
} > "$LOG" 2>&1
echo "LOG=$LOG"
