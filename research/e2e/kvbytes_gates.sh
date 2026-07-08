#!/usr/bin/env bash
# kvbytes lane gate battery (2026-07-08): per KV-format arm on the 35B.
# For each arm: run-gen argmax (MATCH within config) + run-spec full-K sweep (self-consistency
# gate + acceptance table, NGEN=32) + K=3 NGEN=128 low-noise acceptance probe, on p2 + p3.
set -uo pipefail
cd "$(dirname "$0")/../.."
M=/data/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
ENVL="BW24_FAST=1 BW24_GEMM=1 BW24_MMVQ=1 BW24_FA_VEC=1 BW24_MOE_CACHE=1"

gpu_wait() {  # wait until no non-embedder compute process (up to 15 min), then report
    for _ in $(seq 1 15); do
        nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader \
            | grep -v llama-server | grep -q . || return 0
        sleep 60
    done
    echo "WARN: GPU still contended after 15min — proceeding (correctness unaffected)"
}

run_arm() {  # $1 = label, rest = env pairs
    local label=$1; shift
    echo "########## ARM: $label ($*) ##########"
    gpu_wait
    echo "--- run-gen argmax ---"
    env "$@" BW24_FAST=1 BW24_GEMM=1 BW24_MMVQ=1 BW24_FA_VEC=1 BW24_MOE_CACHE=1 \
        ./target/release/run-gen "$M" 2>&1 | grep -E 'argmax|MATCH|MISMATCH'
    for p in p2-code-medium p3-agentic-long; do
        echo "--- run-spec sweep $p (NGEN=32) ---"
        gpu_wait
        env "$@" BW24_PROMPT_FILE=research/e2e/prompts/$p.txt \
            BW24_FAST=1 BW24_GEMM=1 BW24_MMVQ=1 BW24_FA_VEC=1 BW24_MOE_CACHE=1 \
            ./target/release/run-spec "$M" 2>&1 | grep -E 'generate_spec K|acceptance|FAIL'
        echo "--- run-spec K=3 NGEN=128 $p ---"
        gpu_wait
        env "$@" BW24_PROMPT_FILE=research/e2e/prompts/$p.txt BW24_SPEC_K=3 BW24_NGEN=128 \
            BW24_FAST=1 BW24_GEMM=1 BW24_MMVQ=1 BW24_FA_VEC=1 BW24_MOE_CACHE=1 \
            ./target/release/run-spec "$M" 2>&1 | grep -E 'generate_spec K|acceptance|FAIL'
    done
}

run_arm baseline-k3-128only BW24_KV_K=q8_0 BW24_KV_V=q5_1
run_arm V-q4_0  BW24_KV_V=q4_0
run_arm K-fp8   BW24_KV_K=fp8
run_arm V-fp8   BW24_KV_V=fp8
run_arm K-fp8+V-q4_0 BW24_KV_K=fp8 BW24_KV_V=q4_0
echo "########## DONE ##########"
