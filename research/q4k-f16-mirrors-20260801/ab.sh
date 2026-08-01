#!/bin/bash
# q4k-f16-mirrors prefill A/B (round 49, GPU 3): base binary (memra-int v059: Q6_K f16
# mirrors + KQRP) vs new binary (arc4: + Q4_K f16 mirrors). Interleaved x3, board-2048
# (2048-tok prime = the pp2048 carrier) + agentic500. Each run-spec invocation gives
# prime seconds (prefill), plain decode tok/s, and spec K=3 decode tok/s + acceptance.
set -u
cd $HOME/arc4
D=$HOME/arc4/research/q4k-f16-mirrors-20260801
LOGD=$D/ab-logs
mkdir -p $LOGD
M=/opt/dlami/nvme/models/Qwen3.6-27B-Q4_K_M.gguf
for r in 1 2 3; do
  for bin in base new; do
    B=$HOME/arc4/target/release/run-spec
    [ $bin = base ] && B=$HOME/memra-int/target/release/run-spec
    for cls in board2048:research/e2e/prompts/board-2048.txt agentic500:research/q27-mtp-20260801/prompt-agentic-500w.txt; do
      name=${cls%%:*}; pf=${cls#*:}
      CUDA_VISIBLE_DEVICES=3 MEMRA_SPEC_K=3 MEMRA_NGEN=256 MEMRA_SPEC_HPOST=1 MEMRA_SPEC_PMIN=0.3 \
        MEMRA_PROMPT_FILE=$pf $B $M > $LOGD/$bin-$name-r$r.log 2>&1
      echo "$bin $name r$r rc=$?"
    done
  done
done
echo AB-DONE
