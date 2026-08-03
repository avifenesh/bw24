#!/usr/bin/env bash
# Bound the singledev pipelined intermittent: x20 same-build repro, quoted verdicts.
set -u
cd ~/memra
OUT=~/receipts/m2-pp8/soak-singledev
mkdir -p "$OUT"
Q9=/opt/dlami/nvme/models/Qwen3.5-9B-Q8_0.gguf
for i in $(seq 1 20); do
  flock /tmp/gpu-box.lock ./target/release/ppn-gate "$Q9" 2 16 32 > "$OUT/r$i.log" 2>&1
done
grep -h "\[pipelined\]" "$OUT"/r*.log | grep -c "gate PASS" > "$OUT/pass-count.txt"
grep -h "\[pipelined\]" "$OUT"/r*.log | grep -c "gate FAIL" >> "$OUT/pass-count.txt"
