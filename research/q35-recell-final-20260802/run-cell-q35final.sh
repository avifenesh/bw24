#!/bin/bash
# Full q35 board cell N=5 both arms (runs UNDER the gpu flock).
cd ~/memra
export PATH=$HOME/vllm-env/bin:$PATH
exec bash tools/h100-vllm-board.sh "q35" research/tune-data/h100board-vllm-20260731-realtext.jsonl
