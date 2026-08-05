#!/usr/bin/env bash
# pro6000-dev: microbench sweep — NVFP4 batched variants per 27B shape, DRAM-cold (8 copies),
# rp layout (trunk default) + auto non-rp reference. 188-SM die.
set -u
export LD_LIBRARY_PATH=/usr/local/cuda-13.1/compat
cd /root/bw24
R=/root/receipts-dev/msweep
mkdir -p "$R"
M=/root/models/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
nvidia-smi --query-gpu=timestamp,power.draw,clocks.sm,temperature.gpu --format=csv -l 1 > "$R/gpu-1hz.csv" 2>&1 &
SMPID=$!
trap 'kill $SMPID 2>/dev/null' EXIT
SHAPES="blk.1.attn_qkv.weight blk.1.attn_gate.weight blk.1.ffn_gate.weight blk.1.ffn_down.weight blk.1.ssm_out.weight"
# rp-layout variants (trunk default layout) — auto + every forced rp variant
for T in $SHAPES; do
  for BV in auto rp rpr2 rpr2w8 rpsc rpca rpcar2 rpms rpmsc; do
    MSWEEP_TENSOR=$T MSWEEP_COPIES=8 MSWEEP_RP=1 MSWEEP_M1=1 MSWEEP_BATCHED_ONLY=1 \
      MEMRA_MMVQ_BV=$BV timeout 300 target/release/mvq-msweep "$M" \
      > "$R/rp-$T-$BV.log" 2>&1
    echo "rc=$? $T $BV"
  done
  # grid.y=m reference (m=1 mr2 kernel timing) — original layout, one run per shape
  MSWEEP_TENSOR=$T MSWEEP_COPIES=8 timeout 300 target/release/mvq-msweep "$M" \
    > "$R/ref-$T.log" 2>&1
  echo "rc=$? $T ref"
done
echo SWEEP_B_DONE
