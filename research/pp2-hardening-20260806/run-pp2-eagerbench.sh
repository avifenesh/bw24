#!/usr/bin/env bash
# pp2-hardening — the eager PP-2 perf story on the PRO pair (ppn-bench, interleaved N=5).
# Two invocations because the door is a LOAD-TIME decision (an in-process toggle after a
# sharded load would time peer reads and pollute the baseline — ppn-bench's own header law):
#   inv 1: no pp env  -> serial-off  = the single-GPU baseline
#   inv 2: door open  -> serial-pp + pipelined-pp, interleaved rep-major in-process
# The 1.87x deferred-pipelined prize was minted on 8xH100 NVSwitch; this is its first
# measurement on a PRO 6000 pair over PCIe P2P.
set -uo pipefail
cd ~/memra
export PATH=$HOME/.cargo/bin:/usr/local/cuda-13.2/bin:$PATH
OUT=~/receipts/pp2/eagerbench
mkdir -p "$OUT"
Q9=/scratch-models/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
Q27=/scratch-models/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
BIN=target/release
# COHABITATION (2026-08-06): the step37-p2 lane shares this box. Every GPU measurement
# window runs under the shared lock so its bring-up boots cannot overlap an interleaved
# A/B run (cross-run clock/contention drift would invalidate the comparison).
LOCK="flock /tmp/memra-gpu.lock"

nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm,power.draw --format=csv > "$OUT/gpu-pre.csv"

echo "######## q9 ########"
$LOCK $BIN/ppn-bench "$Q9" 32 128 5 > "$OUT/q9-baseline-doorshut.log" 2>&1
echo "-- baseline (door shut) --"; tail -6 "$OUT/q9-baseline-doorshut.log"
$LOCK env MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 $BIN/ppn-bench "$Q9" 32 128 5 > "$OUT/q9-pp2-dev01.log" 2>&1
echo "-- pp2 dev01 (serial + pipelined) --"; tail -8 "$OUT/q9-pp2-dev01.log"

echo "######## q27 (the daily model) ########"
$LOCK $BIN/ppn-bench "$Q27" 32 128 5 > "$OUT/q27-baseline-doorshut.log" 2>&1
echo "-- baseline (door shut) --"; tail -6 "$OUT/q27-baseline-doorshut.log"
$LOCK env MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 $BIN/ppn-bench "$Q27" 32 128 5 > "$OUT/q27-pp2-dev01.log" 2>&1
echo "-- pp2 dev01 (serial + pipelined) --"; tail -8 "$OUT/q27-pp2-dev01.log"

nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm,power.draw --format=csv > "$OUT/gpu-post.csv"
echo EAGERBENCH_DONE
