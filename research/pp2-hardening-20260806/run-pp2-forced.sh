#!/usr/bin/env bash
# pp2-hardening — the QUARANTINE ROOT TEST: force the refused same-device pipelined
# placement and see whether the 35%-flake class (H100, 2026-08-02 x20 soak) reproduces on
# PRO 6000. MEASUREMENT ONLY — this placement stays refused by default.
# Also: x40 more dev01 pipelined to push the cross-device record.
set -uo pipefail
cd ~/memra
export PATH=$HOME/.cargo/bin:/usr/local/cuda-13.2/bin:$PATH
OUT=~/receipts/pp2/soak
mkdir -p "$OUT/forced-singledev" "$OUT/dev01-x40b"
Q9=/scratch-models/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
BIN=target/release

for i in $(seq 1 20); do
  MEMRA_PP_FORCE_SAME_DEV_PIPELINED=1 $BIN/ppn-gate "$Q9" 2 16 32 > "$OUT/forced-singledev/r$i.log" 2>&1
done
echo "==== FORCED same-device pipelined x20 (the quarantined placement) ===="
echo "pipelined PASS: $(grep -l 'ppn gate PASS \[pipelined\]' $OUT/forced-singledev/*.log 2>/dev/null | wc -l)/20"
echo "pipelined FAIL: $(grep -l 'ppn gate FAIL \[pipelined\]' $OUT/forced-singledev/*.log 2>/dev/null | wc -l)/20"
echo "serial    PASS: $(grep -l 'ppn gate PASS \[serial\]' $OUT/forced-singledev/*.log 2>/dev/null | wc -l)/20"
grep -h 'ppn gate FAIL' $OUT/forced-singledev/*.log | head -20

for i in $(seq 1 40); do
  MEMRA_PP_DEVICES=0,1 $BIN/ppn-gate "$Q9" 2 16 32 > "$OUT/dev01-x40b/r$i.log" 2>&1
done
echo "==== dev01 x40 batch B (cumulative cross-device record) ===="
echo "pipelined PASS: $(grep -l 'ppn gate PASS \[pipelined\]' $OUT/dev01-x40b/*.log 2>/dev/null | wc -l)/40"
echo "pipelined FAIL: $(grep -l 'ppn gate FAIL \[pipelined\]' $OUT/dev01-x40b/*.log 2>/dev/null | wc -l)/40"
grep -h 'ppn gate FAIL' $OUT/dev01-x40b/*.log | head -20
echo FORCED_DONE
