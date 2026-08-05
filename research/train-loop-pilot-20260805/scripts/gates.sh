#!/bin/bash
# Stage 4: gate battery on the pilot artifact.
#  4a kernel-check with the pilot GGUF as weight oracle (CLI arg = oracle for matching
#     sections; synthetic arms all run)
#  4b run-gen argmax probe depth (pp22-class) on the pilot GGUF — MATCH required
#  4c run-gen argmax deep (p3-agentic-long) — MATCH required
#  4d run-gen HF-reference cross-check: pilot GGUF vs merged-HF dir, same prompt, greedy
#     token streams must be IDENTICAL (Q8_0-vs-bf16 argmax agreement, run-gen MATCH lines
#     arbitrate each arm internally).
set -uo pipefail
export LD_LIBRARY_PATH=/usr/local/cuda-13.1/compat
export PATH=$HOME/.cargo/bin:/usr/local/cuda-13.1/bin:$PATH
cd /root/bw24
M=/root/pilot/q9pilot-Q8_0.gguf
HF=/root/pilot/merged-hf

case "$1" in
  kc)
    target/release/kernel-check "$M"
    ;;
  argmax-probe)
    MEMRA_NGEN=20 MEMRA_PROMPT_FILE=tools/fast-gate/prompts/probe.txt target/release/run-gen "$M"
    ;;
  argmax-deep)
    MEMRA_NGEN=20 MEMRA_PROMPT_FILE=research/e2e/prompts/p3-agentic-long.txt target/release/run-gen "$M"
    ;;
  hf-xcheck)
    MEMRA_CHAT=1 MEMRA_NGEN=64 target/release/run-gen "$M" --prompt "Explain what a hash map is." | tee /root/pilot/xcheck-gguf.txt
    MEMRA_CHAT=1 MEMRA_NGEN=64 target/release/run-gen "$HF" --prompt "Explain what a hash map is." | tee /root/pilot/xcheck-hf.txt
    G=$(grep "^tokens:" /root/pilot/xcheck-gguf.txt | tail -1)
    H=$(grep "^tokens:" /root/pilot/xcheck-hf.txt | tail -1)
    echo "GGUF $G"
    echo "HF   $H"
    if [ -n "$G" ] && [ "$G" = "$H" ]; then echo "HF-XCHECK: IDENTICAL"; else echo "HF-XCHECK: DIVERGED"; fi
    ;;
  *) echo "unknown gate $1"; exit 2;;
esac
