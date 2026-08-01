#!/bin/bash
# Lane 3 (GPU 3): qwen adaptive-K exactness gates — run-spec K=1..8 self-consistency with
# MEMRA_SPEC_ADAPT=1 on q35 (MoE, embedded head, eager draft) and q27 (dense, graph draft).
# The battery asserts spec output identical to plain generate at every K.
set -u
cd "$HOME/lane3" || exit 1
BW="$HOME/lane3/target/release"
OUT="$HOME/lane3/research/qwen-adaptive-k-20260801"
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=3

echo "== q35 K=1..8 self-consistency, MEMRA_SPEC_ADAPT=1 =="
MEMRA_SPEC_ADAPT=1 timeout 3600 "$BW/run-spec" "$HOME/models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf" \
  >"$OUT/gate-q35-adapt-k1-8.log" 2>&1
grep -E "SELF-CONSISTENCY|self-consistency" "$OUT/gate-q35-adapt-k1-8.log" | tail -10

echo "== q27 K=1..8 self-consistency, MEMRA_SPEC_ADAPT=1 =="
MEMRA_SPEC_ADAPT=1 timeout 3600 "$BW/run-spec" /opt/dlami/nvme/models/Qwen3.6-27B-Q4_K_M.gguf \
  >"$OUT/gate-q27-adapt-k1-8.log" 2>&1
grep -E "SELF-CONSISTENCY|self-consistency" "$OUT/gate-q27-adapt-k1-8.log" | tail -10
echo "GATES DONE"
