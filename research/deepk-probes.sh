#!/usr/bin/env bash
# Deep-K probe battery (2026-07-10 plan, data-validated): the K=3 cap truncates chains whose
# tail slots still accept at 54-80% on 27B-p2 / 35B-p1/p2/p3. One command when the GPU frees.
# Evidence discipline: raw logs kept, idle-gated, N=2 per arm, interleaved base-vs-probe.
set -u
cd /home/avifenesh/projects/bw24
LOGD=research/tune-data/deepk-logs; mkdir -p $LOGD
idle() { while [ "$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits)" -gt 1000 ]; do sleep 5; done; }

M27=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
D27=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/mtp-Qwen3.6-27B-Q4_K_M.gguf
T27=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/mtp-Qwen3.6-27B-Q4_K_M-frspec-code75-32768.gguf
M35=/data/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
T35=$(grep -o '/data[^",]*frspec[^",]*32768[^",]*gguf' research/tune-data/rig5090.jsonl | grep -i '35b\|balanced\|frspec32768' | tail -1)
P1=research/e2e/prompts/p1-code-short.txt
P2=research/e2e/prompts/p2-code-medium.txt
P3=research/e2e/prompts/p3-agentic-long-v3.txt

run() { # run <tag> <model> <prompt> <env...>
  local tag=$1 model=$2 prompt=$3; shift 3
  for n in 1 2; do
    idle
    env "$@" BW24_NGEN=256 BW24_SPEC_STATS=1 BW24_PROMPT_FILE=$prompt \
      ./target/release/run-spec "$model" > $LOGD/$tag-run$n.log 2>&1
    grep -hE 'generate_spec K|acceptance|per_slot' $LOGD/$tag-run$n.log | sed "s/^/[$tag r$n] /"
  done
}

echo "=== 27B p2 deep-K (base K3/pmin.4 = 88.8 band) ==="
run 27b-p2-base   $M27 $P2 BW24_SPEC_K=3 BW24_SPEC_PMIN=0.4 BW24_MTP_DRAFT=$D27 BW24_FRSPEC_TRIM=$T27
run 27b-p2-k4p4   $M27 $P2 BW24_SPEC_K=4 BW24_SPEC_PMIN=0.4 BW24_MTP_DRAFT=$D27 BW24_FRSPEC_TRIM=$T27
run 27b-p2-k4p3   $M27 $P2 BW24_SPEC_K=4 BW24_SPEC_PMIN=0.3 BW24_MTP_DRAFT=$D27 BW24_FRSPEC_TRIM=$T27
run 27b-p2-k5p3   $M27 $P2 BW24_SPEC_K=5 BW24_SPEC_PMIN=0.3 BW24_MTP_DRAFT=$D27 BW24_FRSPEC_TRIM=$T27
run 27b-p2-k4p2   $M27 $P2 BW24_SPEC_K=4 BW24_SPEC_PMIN=0.2 BW24_MTP_DRAFT=$D27 BW24_FRSPEC_TRIM=$T27

echo "=== 35B deep-K (base K3 PMIN0; cells p2 1.02x / p3 1.03x) ==="
B35="BW24_SPEC_PMIN=0.4 BW24_SPEC_PMIN0=1 BW24_FRSPEC_TRIM=$T35"
run 35b-p2-base $M35 $P2 BW24_SPEC_K=3 $B35
run 35b-p2-k4   $M35 $P2 BW24_SPEC_K=4 $B35
run 35b-p2-k5   $M35 $P2 BW24_SPEC_K=5 $B35
S35="BW24_SPEC_TEMP=0.7 BW24_SEED=42 BW24_CHAT=1"
run 35b-p3-base $M35 $P3 BW24_SPEC_K=3 $B35 $S35
run 35b-p3-k4   $M35 $P3 BW24_SPEC_K=4 $B35 $S35
run 35b-p3-k5   $M35 $P3 BW24_SPEC_K=5 $B35 $S35
run 35b-p3-k6   $M35 $P3 BW24_SPEC_K=6 $B35 $S35
run 35b-p1-base $M35 $P1 BW24_SPEC_K=3 $B35
run 35b-p1-k4   $M35 $P1 BW24_SPEC_K=4 $B35

echo "=== done; summarize per_slot + tok/s per tag ==="
