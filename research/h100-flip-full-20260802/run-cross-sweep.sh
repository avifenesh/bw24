#!/bin/bash
# MEMRA_F16G_SK_CROSS re-sweep {16,32,64} on the winning sk arm (runs UNDER the gpu flock).
# Tile economics changed twice since 32 was swept (direct loaders now cover ~100% of the
# bank; the deep tail replaced the 2-stage sub-cross form) — stale-verdict law.
# Sweep-grade: 1 process per arm, median of 5 in-process reps (+1 warmup), MEMRA_PP_ONLY —
# the sk-bm128 protocol. NOT the interleaved claim; the claim number comes from the probe.
cd ~/memra
FOX=research/e2e/prompts/board-2048.txt
G=$HOME/models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
BW=target/release/run-gen
for X in 16 32 64; do
  env MEMRA_MOE_F16G=2 MEMRA_F16G_SK_CROSS=$X \
      MEMRA_PP_ONLY=1 MEMRA_PP_REPS=5 MEMRA_PP_WARMUP=1 MEMRA_PROMPT_FILE=$FOX \
      timeout 900 $BW $G > /tmp/sweep-cross$X.log 2>&1
done
echo SWEEP-DONE
