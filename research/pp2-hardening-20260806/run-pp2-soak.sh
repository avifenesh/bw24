#!/usr/bin/env bash
# pp2-hardening — the FLAKE-ARM REPRO on P2P-capable PRO 6000 silicon.
# The ~0.5% cross-device pipelined flake (1 in ~190 runs) was minted on NON-P2P 5090s /
# NVSwitch H100s. Question: does it reproduce on a native-P2P PRO pair?
# x40 cross-device pipelined (dev01) + x20 same-device quarantine-NOTE confirm.
# Receipts to ~/receipts/pp2/soak/. Params baked as literals.
set -uo pipefail
cd ~/memra
export PATH=$HOME/.cargo/bin:/usr/local/cuda-13.2/bin:$PATH
OUT=~/receipts/pp2/soak
mkdir -p "$OUT/dev01" "$OUT/singledev"
Q9=/scratch-models/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
BIN=target/release

for i in $(seq 1 40); do
  MEMRA_PP_DEVICES=0,1 $BIN/ppn-gate "$Q9" 2 16 32 > "$OUT/dev01/r$i.log" 2>&1
done
for i in $(seq 1 20); do
  $BIN/ppn-gate "$Q9" 2 16 32 > "$OUT/singledev/r$i.log" 2>&1
done

echo "==== dev01 x40 cross-device (serial / pipelined) ===="
echo "serial   PASS: $(grep -lc 'ppn gate PASS \[serial\]' $OUT/dev01/*.log 2>/dev/null | wc -l)/40"
echo "serial   FAIL: $(grep -l 'ppn gate FAIL \[serial\]' $OUT/dev01/*.log 2>/dev/null | wc -l)/40"
echo "pipelined PASS: $(grep -l 'ppn gate PASS \[pipelined\]' $OUT/dev01/*.log 2>/dev/null | wc -l)/40"
echo "pipelined FAIL: $(grep -l 'ppn gate FAIL \[pipelined\]' $OUT/dev01/*.log 2>/dev/null | wc -l)/40"
grep -H 'FAIL' $OUT/dev01/*.log | head -20
echo "==== singledev x20 ===="
echo "serial   PASS: $(grep -l 'ppn gate PASS \[serial\]' $OUT/singledev/*.log 2>/dev/null | wc -l)/20"
echo "quarantine NOTE: $(grep -l 'NOTE' $OUT/singledev/*.log 2>/dev/null | wc -l)/20"
grep -H 'FAIL' $OUT/singledev/*.log | head -20
echo SOAK_DONE
