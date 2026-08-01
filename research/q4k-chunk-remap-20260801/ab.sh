#!/bin/bash
# q4k chunk-remap A/B: base binary (v1 KQRP mirror) vs new binary (v2 lane->chunk remap).
# Interleaved x3, plain [generate] + spec (K=3+HPOST+PMIN0.3) per invocation, 3 prompt classes.
# Dedicated GPU 2 on the 8xH100 box (no co-tenant on this device).
cd $HOME/arc2
LOGD=$HOME/arc2/research/q4k-chunk-remap-20260801/ab-logs
mkdir -p $LOGD
M=/opt/dlami/nvme/models/Qwen3.6-27B-Q4_K_M.gguf
for r in 1 2 3; do
  for bin in base new; do
    B=./target/release/run-spec
    [ $bin = base ] && B=$HOME/arc2/base-bins/run-spec
    for cls in short:research/e2e/prompts/p1-code-short.txt board2048:research/e2e/prompts/board-2048.txt agentic500:research/q27-mtp-20260801/prompt-agentic-500w.txt; do
      name=${cls%%:*}; pf=${cls#*:}
      CUDA_VISIBLE_DEVICES=2 MEMRA_SPEC_K=3 MEMRA_NGEN=256 MEMRA_SPEC_HPOST=1 MEMRA_SPEC_PMIN=0.3 MEMRA_PROMPT_FILE=$pf \
        $B $M > $LOGD/$bin-$name-r$r.log 2>&1
      echo "$bin $name r$r rc=$?"
    done
  done
done
echo AB-DONE
CUDA_VISIBLE_DEVICES=2 MEMRA_SPEC_HPOST=1 MEMRA_SPEC_PMIN=0.3 MEMRA_NGEN=256 MEMRA_PROMPT_FILE=research/e2e/prompts/board-2048.txt \
  ./target/release/run-spec $M > $HOME/arc2/research/q4k-chunk-remap-20260801/gate-runspec-k1-8-final.log 2>&1
echo GATE-DONE rc=$?
