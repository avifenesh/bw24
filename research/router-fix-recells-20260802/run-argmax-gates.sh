#!/bin/bash
cd ~/memra
FOX=research/e2e/prompts/board-2048.txt
MEMRA_NGEN=32 MEMRA_PROMPT_FILE=$FOX timeout 900 target/release/run-gen $HOME/models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf > /tmp/argmax-q35-naked.log 2>&1
MEMRA_NGEN=32 MEMRA_PROMPT_FILE=$FOX timeout 900 target/release/run-gen /opt/dlami/nvme/models/Qwen3.6-27B-Q4_K_M.gguf > /tmp/argmax-q27-naked.log 2>&1
echo GATES-DONE
