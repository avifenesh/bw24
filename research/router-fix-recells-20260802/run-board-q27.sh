#!/bin/bash
cd ~/memra
export PATH=$HOME/vllm-env/bin:$PATH
export HF_HOME=/opt/dlami/nvme/hf
exec bash tools/h100-vllm-board.sh "q27" research/tune-data/h100board-vllm-20260731-realtext.jsonl
