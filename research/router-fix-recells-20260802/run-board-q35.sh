#!/bin/bash
cd ~/memra
export PATH=$HOME/vllm-env/bin:$PATH
exec bash tools/h100-vllm-board.sh "q35" research/tune-data/h100board-vllm-20260731-realtext.jsonl
