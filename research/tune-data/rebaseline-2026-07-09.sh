#!/usr/bin/env bash
# FULL re-baseline 2026-07-09: all 5 models, both formats, both engines, N=2 per arm interleaved.
# BINDING: idle-gate every run, tee logs, record clocks, share hour-regime per model.
set -euo pipefail

LOGDIR=/home/avifenesh/projects/bw24/research/tune-data/rebaseline-logs
JSONL=/home/avifenesh/projects/bw24/research/tune-data/rig5090.jsonl
PDIR=/home/avifenesh/projects/bw24/research/e2e/prompts
export GGML_CUDA_GRAPH_OPT=1

# Idle gate: wait for GPU clocks < 1000 MHz
wait_idle() {
  echo "[$(date +%H:%M:%S)] Waiting for GPU idle (<1000 MHz)..."
  while true; do
    CLK=$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits)
    [ "$CLK" -lt 1000 ] && break
    sleep 5
  done
  echo "[$(date +%H:%M:%S)] GPU idle at ${CLK} MHz"
}

# Record GPU state
gpu_state() {
  nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw --format=csv,noheader,nounits
}

# Check free RAM (need >8GB)
check_ram() {
  FREE=$(free -g | awk '/^Mem:/ {print $7}')
  if [ "$FREE" -lt 8 ]; then
    echo "ERROR: Only ${FREE}GB free RAM, need >8GB"
    exit 1
  fi
  echo "RAM check: ${FREE}GB available"
}

echo "=== REBASELINE 2026-07-09 START: $(date -Is) ==="
check_ram

# MODEL A: 9B GGUF
echo ""
echo "############### MODEL A: 9B GGUF ###############"
MODEL_A=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
TRIM_A=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/frspec-9b-32768.gguf

# A1: bw24 plain d512
for RUN in 1 2; do
  wait_idle
  echo "[A1-bw24-$RUN] plain d512 - pre: $(gpu_state)"
  /home/avifenesh/projects/bw24/target/release/decode-bench "$MODEL_A" 512 128 2>&1 | tee "$LOGDIR/A1-bw24-plain-d512-run$RUN.log"
  echo "[A1-bw24-$RUN] post: $(gpu_state)"
done

# A1: llama plain d512
for RUN in 1 2; do
  wait_idle
  echo "[A1-llama-$RUN] plain d512 - pre: $(gpu_state)"
  /home/avifenesh/projects/llama.cpp/build/bin/llama-bench -m "$MODEL_A" -ngl 999 -fa 1 -ctk q8_0 -ctv q5_1 -n 128 -d 512 -r 1 2>&1 | tee "$LOGDIR/A1-llama-plain-d512-run$RUN.log"
  echo "[A1-llama-$RUN] post: $(gpu_state)"
done

# A2: bw24 plain d6257
for RUN in 1 2; do
  wait_idle
  echo "[A2-bw24-$RUN] plain d6257 - pre: $(gpu_state)"
  /home/avifenesh/projects/bw24/target/release/decode-bench "$MODEL_A" 6257 128 2>&1 | tee "$LOGDIR/A2-bw24-plain-d6257-run$RUN.log"
  echo "[A2-bw24-$RUN] post: $(gpu_state)"
done

# A2: llama plain d6257
for RUN in 1 2; do
  wait_idle
  echo "[A2-llama-$RUN] plain d6257 - pre: $(gpu_state)"
  /home/avifenesh/projects/llama.cpp/build/bin/llama-bench -m "$MODEL_A" -ngl 999 -fa 1 -ctk q8_0 -ctv q5_1 -n 128 -d 6257 -r 1 2>&1 | tee "$LOGDIR/A2-llama-plain-d6257-run$RUN.log"
  echo "[A2-llama-$RUN] post: $(gpu_state)"
done

# A3: bw24 spec p1
for RUN in 1 2; do
  wait_idle
  echo "[A3-bw24-p1-$RUN] spec K=3 p1 - pre: $(gpu_state)"
  BW24_NGEN=256 BW24_SPEC_STATS=1 BW24_SPEC_K=3 BW24_SPEC_PMIN=0.3 BW24_FRSPEC_TRIM="$TRIM_A" \
    BW24_PROMPT_FILE="$PDIR/p1-code-short.txt" \
    /home/avifenesh/projects/bw24/target/release/run-spec "$MODEL_A" 2>&1 | tee "$LOGDIR/A3-bw24-spec-p1-run$RUN.log"
  echo "[A3-bw24-p1-$RUN] post: $(gpu_state)"
done

# A3: llama spec p1 (via bash script)
for RUN in 1 2; do
  wait_idle
  echo "[A3-llama-p1-$RUN] spec p1 - pre: $(gpu_state)"
  bash /home/avifenesh/projects/bw24/research/tune-data/st-pairing-logs/llama-spec-round.sh 9b "$LOGDIR/A3-llama-spec-run$RUN.log"
  echo "[A3-llama-p1-$RUN] post: $(gpu_state)"
done

# A3: bw24 spec p2
for RUN in 1 2; do
  wait_idle
  echo "[A3-bw24-p2-$RUN] spec K=3 p2 - pre: $(gpu_state)"
  BW24_NGEN=256 BW24_SPEC_STATS=1 BW24_SPEC_K=3 BW24_SPEC_PMIN=0.3 BW24_FRSPEC_TRIM="$TRIM_A" \
    BW24_PROMPT_FILE="$PDIR/p2-code-medium.txt" \
    /home/avifenesh/projects/bw24/target/release/run-spec "$MODEL_A" 2>&1 | tee "$LOGDIR/A3-bw24-spec-p2-run$RUN.log"
  echo "[A3-bw24-p2-$RUN] post: $(gpu_state)"
done

# A3: bw24 spec p2 text-audit (one run with BW24_PRINT_TEXT=1)
wait_idle
echo "[A3-audit] spec p2 text-audit - pre: $(gpu_state)"
BW24_NGEN=256 BW24_SPEC_STATS=1 BW24_SPEC_K=3 BW24_SPEC_PMIN=0.3 BW24_FRSPEC_TRIM="$TRIM_A" \
  BW24_PROMPT_FILE="$PDIR/p2-code-medium.txt" BW24_PRINT_TEXT=1 \
  /home/avifenesh/projects/bw24/target/release/run-spec "$MODEL_A" 2>&1 | tee "$LOGDIR/A-audit-p2.log"

echo "[MODEL A DONE]"

# MODEL B: 9B ST
echo ""
echo "############### MODEL B: 9B ST ###############"
MODEL_B=/data/ai-ml/hf-models/qwen35-9b-nvfp4-st-modelopt
TRIM_B=/data/ai-ml/hf-models/qwen35-9b-nvfp4-st-modelopt/frspec-9bst-modelopt-32768.gguf

# B1: bw24 plain d512
for RUN in 1 2; do
  wait_idle
  echo "[B1-bw24-$RUN] plain d512 - pre: $(gpu_state)"
  /home/avifenesh/projects/bw24/target/release/decode-bench "$MODEL_B" 512 128 2>&1 | tee "$LOGDIR/B1-bw24-plain-d512-run$RUN.log"
  echo "[B1-bw24-$RUN] post: $(gpu_state)"
done

# B1: llama plain d512 (uses same 9B GGUF as A)
for RUN in 1 2; do
  wait_idle
  echo "[B1-llama-$RUN] plain d512 - pre: $(gpu_state)"
  /home/avifenesh/projects/llama.cpp/build/bin/llama-bench -m "$MODEL_A" -ngl 999 -fa 1 -ctk q8_0 -ctv q5_1 -n 128 -d 512 -r 1 2>&1 | tee "$LOGDIR/B1-llama-plain-d512-run$RUN.log"
  echo "[B1-llama-$RUN] post: $(gpu_state)"
done

# B2: bw24 plain d6257
for RUN in 1 2; do
  wait_idle
  echo "[B2-bw24-$RUN] plain d6257 - pre: $(gpu_state)"
  /home/avifenesh/projects/bw24/target/release/decode-bench "$MODEL_B" 6257 128 2>&1 | tee "$LOGDIR/B2-bw24-plain-d6257-run$RUN.log"
  echo "[B2-bw24-$RUN] post: $(gpu_state)"
done

# B2: llama plain d6257
for RUN in 1 2; do
  wait_idle
  echo "[B2-llama-$RUN] plain d6257 - pre: $(gpu_state)"
  /home/avifenesh/projects/llama.cpp/build/bin/llama-bench -m "$MODEL_A" -ngl 999 -fa 1 -ctk q8_0 -ctv q5_1 -n 128 -d 6257 -r 1 2>&1 | tee "$LOGDIR/B2-llama-plain-d6257-run$RUN.log"
  echo "[B2-llama-$RUN] post: $(gpu_state)"
done

# B3: bw24 spec p1
for RUN in 1 2; do
  wait_idle
  echo "[B3-bw24-p1-$RUN] spec K=2 p1 - pre: $(gpu_state)"
  BW24_NGEN=256 BW24_SPEC_STATS=1 BW24_SPEC_K=2 BW24_SPEC_PMIN=0.3 BW24_FRSPEC_TRIM="$TRIM_B" \
    BW24_PROMPT_FILE="$PDIR/p1-code-short.txt" \
    /home/avifenesh/projects/bw24/target/release/run-spec "$MODEL_B" 2>&1 | tee "$LOGDIR/B3-bw24-spec-p1-run$RUN.log"
  echo "[B3-bw24-p1-$RUN] post: $(gpu_state)"
done

# B3: llama spec p1 (reuse 9B script)
for RUN in 1 2; do
  wait_idle
  echo "[B3-llama-p1-$RUN] spec p1 - pre: $(gpu_state)"
  bash /home/avifenesh/projects/bw24/research/tune-data/st-pairing-logs/llama-spec-round.sh 9b "$LOGDIR/B3-llama-spec-run$RUN.log"
  echo "[B3-llama-p1-$RUN] post: $(gpu_state)"
done

# B3: bw24 spec p2
for RUN in 1 2; do
  wait_idle
  echo "[B3-bw24-p2-$RUN] spec K=2 p2 - pre: $(gpu_state)"
  BW24_NGEN=256 BW24_SPEC_STATS=1 BW24_SPEC_K=2 BW24_SPEC_PMIN=0.3 BW24_FRSPEC_TRIM="$TRIM_B" \
    BW24_PROMPT_FILE="$PDIR/p2-code-medium.txt" \
    /home/avifenesh/projects/bw24/target/release/run-spec "$MODEL_B" 2>&1 | tee "$LOGDIR/B3-bw24-spec-p2-run$RUN.log"
  echo "[B3-bw24-p2-$RUN] post: $(gpu_state)"
done

# B3: text-audit
wait_idle
BW24_NGEN=256 BW24_SPEC_STATS=1 BW24_SPEC_K=2 BW24_SPEC_PMIN=0.3 BW24_FRSPEC_TRIM="$TRIM_B" \
  BW24_PROMPT_FILE="$PDIR/p2-code-medium.txt" BW24_PRINT_TEXT=1 \
  /home/avifenesh/projects/bw24/target/release/run-spec "$MODEL_B" 2>&1 | tee "$LOGDIR/B-audit-p2.log"

echo "[MODEL B DONE]"

# MODEL C: 27B GGUF
echo ""
echo "############### MODEL C: 27B GGUF ###############"
MODEL_C=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
DRAFT_C=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/mtp-Qwen3.6-27B-Q4_K_M.gguf
TRIM_C=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/mtp-Qwen3.6-27B-Q4_K_M-frspec-code75-32768.gguf

# C1: bw24 plain d512
for RUN in 1 2; do
  wait_idle
  echo "[C1-bw24-$RUN] plain d512 - pre: $(gpu_state)"
  /home/avifenesh/projects/bw24/target/release/decode-bench "$MODEL_C" 512 128 2>&1 | tee "$LOGDIR/C1-bw24-plain-d512-run$RUN.log"
  echo "[C1-bw24-$RUN] post: $(gpu_state)"
done

# C1: llama plain d512
for RUN in 1 2; do
  wait_idle
  echo "[C1-llama-$RUN] plain d512 - pre: $(gpu_state)"
  /home/avifenesh/projects/llama.cpp/build/bin/llama-bench -m "$MODEL_C" -ngl 999 -fa 1 -ctk q8_0 -ctv q5_1 -n 128 -d 512 -r 1 2>&1 | tee "$LOGDIR/C1-llama-plain-d512-run$RUN.log"
  echo "[C1-llama-$RUN] post: $(gpu_state)"
done

# C2: bw24 plain d6257
for RUN in 1 2; do
  wait_idle
  echo "[C2-bw24-$RUN] plain d6257 - pre: $(gpu_state)"
  /home/avifenesh/projects/bw24/target/release/decode-bench "$MODEL_C" 6257 128 2>&1 | tee "$LOGDIR/C2-bw24-plain-d6257-run$RUN.log"
  echo "[C2-bw24-$RUN] post: $(gpu_state)"
done

# C2: llama plain d6257
for RUN in 1 2; do
  wait_idle
  echo "[C2-llama-$RUN] plain d6257 - pre: $(gpu_state)"
  /home/avifenesh/projects/llama.cpp/build/bin/llama-bench -m "$MODEL_C" -ngl 999 -fa 1 -ctk q8_0 -ctv q5_1 -n 128 -d 6257 -r 1 2>&1 | tee "$LOGDIR/C2-llama-plain-d6257-run$RUN.log"
  echo "[C2-llama-$RUN] post: $(gpu_state)"
done

# C3: bw24 spec p1
for RUN in 1 2; do
  wait_idle
  echo "[C3-bw24-p1-$RUN] spec K=3 p1 - pre: $(gpu_state)"
  BW24_NGEN=256 BW24_SPEC_STATS=1 BW24_SPEC_K=3 BW24_SPEC_PMIN=0.4 \
    BW24_MTP_DRAFT="$DRAFT_C" BW24_FRSPEC_TRIM="$TRIM_C" \
    BW24_PROMPT_FILE="$PDIR/p1-code-short.txt" \
    /home/avifenesh/projects/bw24/target/release/run-spec "$MODEL_C" 2>&1 | tee "$LOGDIR/C3-bw24-spec-p1-run$RUN.log"
  echo "[C3-bw24-p1-$RUN] post: $(gpu_state)"
done

# C3: llama spec p1
for RUN in 1 2; do
  wait_idle
  echo "[C3-llama-p1-$RUN] spec p1 - pre: $(gpu_state)"
  bash /home/avifenesh/projects/bw24/research/tune-data/st-pairing-logs/llama-spec-round.sh 27b "$LOGDIR/C3-llama-spec-run$RUN.log"
  echo "[C3-llama-p1-$RUN] post: $(gpu_state)"
done

# C3: bw24 spec p2
for RUN in 1 2; do
  wait_idle
  echo "[C3-bw24-p2-$RUN] spec K=3 p2 - pre: $(gpu_state)"
  BW24_NGEN=256 BW24_SPEC_STATS=1 BW24_SPEC_K=3 BW24_SPEC_PMIN=0.4 \
    BW24_MTP_DRAFT="$DRAFT_C" BW24_FRSPEC_TRIM="$TRIM_C" \
    BW24_PROMPT_FILE="$PDIR/p2-code-medium.txt" \
    /home/avifenesh/projects/bw24/target/release/run-spec "$MODEL_C" 2>&1 | tee "$LOGDIR/C3-bw24-spec-p2-run$RUN.log"
  echo "[C3-bw24-p2-$RUN] post: $(gpu_state)"
done

# C3: text-audit
wait_idle
BW24_NGEN=256 BW24_SPEC_STATS=1 BW24_SPEC_K=3 BW24_SPEC_PMIN=0.4 \
  BW24_MTP_DRAFT="$DRAFT_C" BW24_FRSPEC_TRIM="$TRIM_C" \
  BW24_PROMPT_FILE="$PDIR/p2-code-medium.txt" BW24_PRINT_TEXT=1 \
  /home/avifenesh/projects/bw24/target/release/run-spec "$MODEL_C" 2>&1 | tee "$LOGDIR/C-audit-p2.log"

echo "[MODEL C DONE]"

# MODEL D: 27B ST
echo ""
echo "############### MODEL D: 27B ST ###############"
MODEL_D=/data/ai-ml/hf-models/nvidia-qwen36-27b-nvfp4
TRIM_D=/data/ai-ml/hf-models/nvidia-qwen36-27b-nvfp4/frspec-corpus-32768.gguf

# D1: bw24 plain d512 (with BW24_NV_W4=1)
for RUN in 1 2; do
  wait_idle
  echo "[D1-bw24-$RUN] plain d512 - pre: $(gpu_state)"
  BW24_NV_W4=1 /home/avifenesh/projects/bw24/target/release/decode-bench "$MODEL_D" 512 128 2>&1 | tee "$LOGDIR/D1-bw24-plain-d512-run$RUN.log"
  echo "[D1-bw24-$RUN] post: $(gpu_state)"
done

# D1: llama plain d512 (uses 27B GGUF)
for RUN in 1 2; do
  wait_idle
  echo "[D1-llama-$RUN] plain d512 - pre: $(gpu_state)"
  /home/avifenesh/projects/llama.cpp/build/bin/llama-bench -m "$MODEL_C" -ngl 999 -fa 1 -ctk q8_0 -ctv q5_1 -n 128 -d 512 -r 1 2>&1 | tee "$LOGDIR/D1-llama-plain-d512-run$RUN.log"
  echo "[D1-llama-$RUN] post: $(gpu_state)"
done

# D2: bw24 plain d6257
for RUN in 1 2; do
  wait_idle
  echo "[D2-bw24-$RUN] plain d6257 - pre: $(gpu_state)"
  BW24_NV_W4=1 /home/avifenesh/projects/bw24/target/release/decode-bench "$MODEL_D" 6257 128 2>&1 | tee "$LOGDIR/D2-bw24-plain-d6257-run$RUN.log"
  echo "[D2-bw24-$RUN] post: $(gpu_state)"
done

# D2: llama plain d6257
for RUN in 1 2; do
  wait_idle
  echo "[D2-llama-$RUN] plain d6257 - pre: $(gpu_state)"
  /home/avifenesh/projects/llama.cpp/build/bin/llama-bench -m "$MODEL_C" -ngl 999 -fa 1 -ctk q8_0 -ctv q5_1 -n 128 -d 6257 -r 1 2>&1 | tee "$LOGDIR/D2-llama-plain-d6257-run$RUN.log"
  echo "[D2-llama-$RUN] post: $(gpu_state)"
done

# D3: bw24 spec p1 (BW24_NV_W4=1 + HPOST=1 + pmin 0.4)
for RUN in 1 2; do
  wait_idle
  echo "[D3-bw24-p1-$RUN] spec K=3 p1 - pre: $(gpu_state)"
  BW24_NV_W4=1 BW24_NGEN=256 BW24_SPEC_STATS=1 BW24_SPEC_K=3 BW24_SPEC_HPOST=1 BW24_SPEC_PMIN=0.4 \
    BW24_FRSPEC_TRIM="$TRIM_D" \
    BW24_PROMPT_FILE="$PDIR/p1-code-short.txt" \
    /home/avifenesh/projects/bw24/target/release/run-spec "$MODEL_D" 2>&1 | tee "$LOGDIR/D3-bw24-spec-p1-run$RUN.log"
  echo "[D3-bw24-p1-$RUN] post: $(gpu_state)"
done

# D3: llama spec p1 (reuse 27B script)
for RUN in 1 2; do
  wait_idle
  echo "[D3-llama-p1-$RUN] spec p1 - pre: $(gpu_state)"
  bash /home/avifenesh/projects/bw24/research/tune-data/st-pairing-logs/llama-spec-round.sh 27b "$LOGDIR/D3-llama-spec-run$RUN.log"
  echo "[D3-llama-p1-$RUN] post: $(gpu_state)"
done

# D3: bw24 spec p2 (pmin 0.3 for p2)
for RUN in 1 2; do
  wait_idle
  echo "[D3-bw24-p2-$RUN] spec K=3 p2 - pre: $(gpu_state)"
  BW24_NV_W4=1 BW24_NGEN=256 BW24_SPEC_STATS=1 BW24_SPEC_K=3 BW24_SPEC_HPOST=1 BW24_SPEC_PMIN=0.3 \
    BW24_FRSPEC_TRIM="$TRIM_D" \
    BW24_PROMPT_FILE="$PDIR/p2-code-medium.txt" \
    /home/avifenesh/projects/bw24/target/release/run-spec "$MODEL_D" 2>&1 | tee "$LOGDIR/D3-bw24-spec-p2-run$RUN.log"
  echo "[D3-bw24-p2-$RUN] post: $(gpu_state)"
done

# D3: text-audit
wait_idle
BW24_NV_W4=1 BW24_NGEN=256 BW24_SPEC_STATS=1 BW24_SPEC_K=3 BW24_SPEC_HPOST=1 BW24_SPEC_PMIN=0.3 \
  BW24_FRSPEC_TRIM="$TRIM_D" \
  BW24_PROMPT_FILE="$PDIR/p2-code-medium.txt" BW24_PRINT_TEXT=1 \
  /home/avifenesh/projects/bw24/target/release/run-spec "$MODEL_D" 2>&1 | tee "$LOGDIR/D-audit-p2.log"

echo "[MODEL D DONE]"

# MODEL E: 35B GGUF
echo ""
echo "############### MODEL E: 35B GGUF ###############"
MODEL_E=/data/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
# Find spec config from recent JSONL (K=3, trim from memory-card 35B spec config)
TRIM_E=/data/ai-ml/hf-models/qwen36-35b-moe/frspec-35b-32768.gguf  # placeholder - will check if exists

# Check VRAM before loading 35B
echo "[MODEL E] Checking VRAM/RAM before 35B load..."
check_ram
nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits

# E1: bw24 plain d512
for RUN in 1 2; do
  wait_idle
  echo "[E1-bw24-$RUN] plain d512 - pre: $(gpu_state)"
  /home/avifenesh/projects/bw24/target/release/decode-bench "$MODEL_E" 512 128 2>&1 | tee "$LOGDIR/E1-bw24-plain-d512-run$RUN.log"
  echo "[E1-bw24-$RUN] post: $(gpu_state)"
done

# E1: llama plain d512
for RUN in 1 2; do
  wait_idle
  echo "[E1-llama-$RUN] plain d512 - pre: $(gpu_state)"
  /home/avifenesh/projects/llama.cpp/build/bin/llama-bench -m "$MODEL_E" -ngl 999 -fa 1 -ctk q8_0 -ctv q5_1 -n 128 -d 512 -r 1 2>&1 | tee "$LOGDIR/E1-llama-plain-d512-run$RUN.log"
  echo "[E1-llama-$RUN] post: $(gpu_state)"
done

# E2: bw24 plain d6257
for RUN in 1 2; do
  wait_idle
  echo "[E2-bw24-$RUN] plain d6257 - pre: $(gpu_state)"
  /home/avifenesh/projects/bw24/target/release/decode-bench "$MODEL_E" 6257 128 2>&1 | tee "$LOGDIR/E2-bw24-plain-d6257-run$RUN.log"
  echo "[E2-bw24-$RUN] post: $(gpu_state)"
done

# E2: llama plain d6257
for RUN in 1 2; do
  wait_idle
  echo "[E2-llama-$RUN] plain d6257 - pre: $(gpu_state)"
  /home/avifenesh/projects/llama.cpp/build/bin/llama-bench -m "$MODEL_E" -ngl 999 -fa 1 -ctk q8_0 -ctv q5_1 -n 128 -d 6257 -r 1 2>&1 | tee "$LOGDIR/E2-llama-plain-d6257-run$RUN.log"
  echo "[E2-llama-$RUN] post: $(gpu_state)"
done

# E3: bw24 spec p1 (copy config from last 35B rows in JSONL - K=3 typical)
for RUN in 1 2; do
  wait_idle
  echo "[E3-bw24-p1-$RUN] spec K=3 p1 - pre: $(gpu_state)"
  BW24_NGEN=256 BW24_SPEC_STATS=1 BW24_SPEC_K=3 BW24_SPEC_PMIN=0.3 \
    BW24_PROMPT_FILE="$PDIR/p1-code-short.txt" \
    /home/avifenesh/projects/bw24/target/release/run-spec "$MODEL_E" 2>&1 | tee "$LOGDIR/E3-bw24-spec-p1-run$RUN.log"
  echo "[E3-bw24-p1-$RUN] post: $(gpu_state)"
done

# E3: llama spec p1 (serve config from COMPETITOR-SETUP.md sec 1b for 35B)
# NOTE: 35B llama MTP serve config not in llama-spec-round.sh - must be manual or skip for now
echo "[E3-llama] 35B llama spec setup not in llama-spec-round.sh - skipping or needs manual serve"

# E3: bw24 spec p2
for RUN in 1 2; do
  wait_idle
  echo "[E3-bw24-p2-$RUN] spec K=3 p2 - pre: $(gpu_state)"
  BW24_NGEN=256 BW24_SPEC_STATS=1 BW24_SPEC_K=3 BW24_SPEC_PMIN=0.3 \
    BW24_PROMPT_FILE="$PDIR/p2-code-medium.txt" \
    /home/avifenesh/projects/bw24/target/release/run-spec "$MODEL_E" 2>&1 | tee "$LOGDIR/E3-bw24-spec-p2-run$RUN.log"
  echo "[E3-bw24-p2-$RUN] post: $(gpu_state)"
done

# E3: text-audit
wait_idle
BW24_NGEN=256 BW24_SPEC_STATS=1 BW24_SPEC_K=3 BW24_SPEC_PMIN=0.3 \
  BW24_PROMPT_FILE="$PDIR/p2-code-medium.txt" BW24_PRINT_TEXT=1 \
  /home/avifenesh/projects/bw24/target/release/run-spec "$MODEL_E" 2>&1 | tee "$LOGDIR/E-audit-p2.log"

echo "[MODEL E DONE]"

echo ""
echo "=== REBASELINE 2026-07-09 COMPLETE: $(date -Is) ==="
echo "All logs in: $LOGDIR"
echo "Next: parse logs, write JSONL rows, generate summary table"
