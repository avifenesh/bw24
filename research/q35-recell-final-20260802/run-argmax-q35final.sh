#!/bin/bash
# argmax sanity, q35 naked (runs UNDER the gpu flock).
cd ~/memra
FOX=research/e2e/prompts/board-2048.txt
MEMRA_NGEN=32 MEMRA_PROMPT_FILE=$FOX timeout 900 target/release/run-gen $HOME/models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf > /tmp/argmax-q35final.log 2>&1
