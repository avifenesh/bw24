#!/bin/bash
# h100-spec-coverage: two H100 spec gaps (lane/h100-spec-coverage, 2026-08-02).
#   (1) q35 spec-on-Hopper — the flip lane's caveat (research/h100-flip-full-20260802:
#       "structurally unavailable, drafter not on this box"). Drafter now staged:
#       ~/models/draft-35b-owntrim-nvfp4head-q4blk.gguf
#       (sha256 ae5b7797cc10188bddd00d7e46394e6b8676c1d4e4c6768c8b7b3b10d8870b6a, byte-
#       identical to the 5090 source). run-spec K=1..8 on the flipped-naked tree.
#   (2) q27 MTP spec on sm_90a under the v0.65-tip tree — K=1..8 battery, then the
#       spec-vs-plain e2e cell (board-2048, N=3, best K of {2,3,4}), then vLLM
#       best-config comparator (FP8 + MTP spec, bench_vllm.py --spec-k) same session.
# Box: darklanes-bench H100 80GB (Mumbai). Tree ~/memra = rsync of
# restructure/public-split a70a13c2 (SOURCE-COMMIT.txt), MEMRA_CUDA_ARCH=90a release.
# EVERY GPU-touching process under flock /tmp/gpu-h100.lock (shared-box rule).
# usage: run-speccov.sh <kc|q35spec|q27spec|q27e2e K|vllm K>
set -u
PHASE=${1:?usage: run-speccov.sh <kc|q35spec|q27spec|q27e2e K|vllm K>}
T=$HOME/memra
R=$HOME/spec-scratch
M35=$HOME/models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
D35=$HOME/models/draft-35b-owntrim-nvfp4head-q4blk.gguf
M27=/opt/dlami/nvme/models/Qwen3.6-27B-Q4_K_M.gguf
P=$T/research/e2e/prompts/board-2048.txt
mkdir -p "$R"

gpustate() {
  echo "[gpu $(date -u +%FT%TZ)] $(nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm,memory.used --format=csv,noheader)"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | sed 's/^/[apps] /'
}

case $PHASE in
kc)
  L=$R/kc-speccov.log
  { echo "tree $(cat "$T"/SOURCE-COMMIT.txt)"; gpustate; } > "$L" 2>&1
  flock /tmp/gpu-h100.lock timeout 3600 "$T/target/release/kernel-check" >> "$L" 2>&1
  rc=$?
  { gpustate; echo "KC rc=$rc"; } >> "$L" 2>&1
  echo "kc rc=$rc $(tail -3 "$L" | head -1)"
  tail -1 "$L"
  ;;
q35spec)
  # One process = plain oracle + K=1..8 battery (no MEMRA_SPEC_K), single model load.
  # NAKED mode arm: the flipped Hopper default (mode 2, direct+tail) — the exact class
  # the caveat is about. Drafter head via MEMRA_MTP_DRAFT (owntrim, same file as the
  # 5090 PASS x8 receipt in research/iq-direct-loaders-20260802). REP=rN tags the log.
  L=$R/q35-spec-k1-8${REP:+-$REP}.log
  { echo "tree $(cat "$T"/SOURCE-COMMIT.txt)"; sha256sum "$D35"; gpustate; } > "$L" 2>&1
  MEMRA_MTP_DRAFT=$D35 MEMRA_PROMPT_FILE=$P MEMRA_NGEN=128 \
    flock /tmp/gpu-h100.lock timeout 3600 "$T/target/release/run-spec" "$M35" >> "$L" 2>&1
  rc=$?
  { gpustate; echo "Q35SPEC rc=$rc"; } >> "$L" 2>&1
  echo "q35spec rc=$rc PASS-lines=$(grep -c 'self-consistency: PASS' "$L") FAIL-lines=$(grep -ci 'self-consistency: FAIL' "$L")"
  grep -E 'generate\] |generate_spec K=|acceptance' "$L"
  ;;
q27spec)
  # K=1..8 battery on the MTP-baked q27 artifact (nextn head in-file, no drafter env).
  L=$R/q27-spec-k1-8${REP:+-$REP}.log
  { echo "tree $(cat "$T"/SOURCE-COMMIT.txt)"; gpustate; } > "$L" 2>&1
  MEMRA_PROMPT_FILE=$P MEMRA_NGEN=256 \
    flock /tmp/gpu-h100.lock timeout 3600 "$T/target/release/run-spec" "$M27" >> "$L" 2>&1
  rc=$?
  { gpustate; echo "Q27SPEC rc=$rc"; } >> "$L" 2>&1
  echo "q27spec rc=$rc PASS-lines=$(grep -c 'self-consistency: PASS' "$L") FAIL-lines=$(grep -ci 'self-consistency: FAIL' "$L")"
  grep -E 'generate\] |generate_spec K=|acceptance' "$L"
  ;;
q27e2e)
  # Board-class cell p2048/g512: one run-spec process per rep = plain oracle + spec at
  # the swept-best K, interleaved by construction (same load, same clock). N=3.
  K=${2:?q27e2e needs K}
  for rep in 1 2 3; do
    L=$R/q27-e2e-k$K-r$rep.log
    { echo "tree $(cat "$T"/SOURCE-COMMIT.txt)"; gpustate; } > "$L" 2>&1
    MEMRA_SPEC_K=$K MEMRA_PROMPT_FILE=$P MEMRA_NGEN=512 \
      flock /tmp/gpu-h100.lock timeout 3600 "$T/target/release/run-spec" "$M27" >> "$L" 2>&1
    rc=$?
    { gpustate; echo "Q27E2E rep=$rep rc=$rc"; } >> "$L" 2>&1
    echo "q27e2e k=$K rep=$rep rc=$rc"
    grep -E 'generate\] |generate_spec K=|acceptance' "$L"
  done
  ;;
vllm)
  # vLLM best-config comparator: FP8 artifact + its own MTP head via speculative_config
  # (method=mtp, num_speculative_tokens=K) — the v0.59 vLLM-BEST class. Same prompt
  # class (board-2048 text, 2048 tok), ngen 512, N=3, same session as our spec arm.
  K=${2:?vllm needs K}
  # nvrtc shim: the FP8+MTP spec path JIT-compiles flashinfer's fp8_blockscale_gemm_90
  # (deep_gemm) through ninja; the system CUDA at /usr/local/cuda has no nvrtc-dev
  # headers ("fatal error: nvrtc.h" — attempt 1, q27-vllm-spec{3,4}-FAILED-nvrtc.log).
  # Putting the whole vllm-env pip cu13 include dir on CPATH shadows the nvcc 13.1
  # toolkit headers ("CUDA compiler and CUDA toolkit headers are incompatible" —
  # attempt 2, q27-vllm-spec3-FAILED-cpath.log). Shim = ONLY nvrtc.h + libnvrtc*
  # symlinked into an otherwise-empty dir; no system or vllm-env mutation.
  NVPIP=$HOME/vllm-env/lib/python3.12/site-packages/nvidia/cu13
  SHIM=$HOME/nvrtc-shim
  mkdir -p "$SHIM/include" "$SHIM/lib"
  ln -sf "$NVPIP/include/nvrtc.h" "$SHIM/include/nvrtc.h"
  ln -sf "$NVPIP"/lib/libnvrtc* "$SHIM/lib/"
  L=$R/q27-vllm-spec$K.log
  { gpustate; } > "$L" 2>&1
  cd "$T"
  HF_HOME=/opt/dlami/nvme/hf CUDA_HOME=/usr/local/cuda PATH=$HOME/vllm-env/bin:/usr/local/cuda/bin:$PATH \
    CPATH=$SHIM/include LIBRARY_PATH=$SHIM/lib LD_LIBRARY_PATH=$SHIM/lib:${LD_LIBRARY_PATH:-} \
    flock /tmp/gpu-h100.lock timeout 3600 "$HOME/vllm-env/bin/python3" bench_vllm.py \
    --model Qwen/Qwen3.6-27B-FP8 --runs 3 --spec-k "$K" \
    --out "$R/q27-vllm-spec$K.json" >> "$L" 2>&1
  rc=$?
  { gpustate; echo "VLLM rc=$rc"; } >> "$L" 2>&1
  echo "vllm k=$K rc=$rc"
  cat "$R/q27-vllm-spec$K.json" 2>/dev/null | tail -8
  ;;
esac
echo "SPECCOV-$PHASE-DONE $(date -u +%FT%TZ)"
