#!/bin/bash
# q27 decode-bw A/B: base binary (GGUF-layout decode) vs new binary (K-quant rp mirrors).
# Interleaved x3, plain [generate] + spec (K=3+HPOST+PMIN0.3) per invocation, 3 prompt classes.
cd $HOME/lane1
LOGD=$HOME/lane1/research/q27-decode-bw-20260801/ab-logs
mkdir -p $LOGD
M=/opt/dlami/nvme/models/Qwen3.6-27B-Q4_K_M.gguf
for r in 1 2 3; do
  for bin in base new; do
    B=./target/release/run-spec
    [ $bin = base ] && B=$HOME/lane1/base-bins/run-spec
    for cls in short:research/e2e/prompts/p1-code-short.txt board2048:research/e2e/prompts/board-2048.txt agentic500:research/q27-mtp-20260801/prompt-agentic-500w.txt; do
      name=${cls%%:*}; pf=${cls#*:}
      CUDA_VISIBLE_DEVICES=1 MEMRA_SPEC_K=3 MEMRA_NGEN=256 MEMRA_SPEC_HPOST=1 MEMRA_SPEC_PMIN=0.3 MEMRA_PROMPT_FILE=$pf \
        $B $M > $LOGD/$bin-$name-r$r.log 2>&1
      echo "$bin $name r$r rc=$?"
    done
  done
done
echo AB-DONE
