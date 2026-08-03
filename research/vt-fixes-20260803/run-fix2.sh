#!/bin/bash
# vt-fixes fix 2 measurement battery (lane/vt-fixes, 2026-08-03, RTX 5090 Laptop 24GB sm_120a).
# Fix 2 = batched-verify epilogue re-fuse (add_rms_norm_q8_1 / rms_norm_q8_1 /
# silu_mul_scaled_q8_1 / gated_rmsnorm_q8_1 twins at nrows=T wired into decode_step_t_core).
#
# GPU protocol: every GPU-touching run under its own short `flock /tmp/gpu5090.lock` hold.
# Interleaved A/B: pre-fix binary = research/vt-fixes-20260803/prefix-bin (built at the branch
# point 50bf95bb before any lane change), post-fix = target/release. Arms alternate per rep
# (A B A B ...) so clock drift is shared.
set -u
PHASE=${1:?usage: run-fix2.sh <probe27|probe9|spec27 K CLASS REP ARM|spec9 K CLASS REP ARM>}
W=/home/avifenesh/projects/wt-vt-fixes
R=$W/research/vt-fixes-20260803/logs
PRE=$W/research/vt-fixes-20260803/prefix-bin
M27=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
D27=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/draft-daily-owntrim-nvfp4head-q4blk.gguf
M9=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
D9=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/draft-9b-owntrim-nvfp4head-q4blk.gguf
PROSE=$W/research/e2e/prompts/board-2048.txt
CODE=$W/research/e2e/prompts/p1-code-short.txt
mkdir -p "$R"

gpustate() {
  echo "[gpu $(date -u +%FT%TZ)] $(nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm,memory.used --format=csv,noheader)"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | sed 's/^/[apps] /'
}

case $PHASE in
probe27|probe9)
  # v(T) cost curve, post-fix vs pre-fix binary, interleaved per arm (same spec-econ
  # fixed-position probe as verify-tier-20260802: N=50 + 3 warmups, sync-bounded).
  ARM=${2:?probe needs arm: pre|post}
  if [ "$PHASE" = probe27 ]; then M=$M27; TAG=q27; else M=$M9; TAG=q9; fi
  BIN=$([ "$ARM" = pre ] && echo "$PRE/spec-econ" || echo "$W/target/release/spec-econ")
  L=$R/probe-$TAG-$ARM.log
  { echo "tree $(git -C "$W" rev-parse HEAD) arm=$ARM"; gpustate; } > "$L" 2>&1
  MEMRA_ECON_N=50 MEMRA_ECON_TMAX=9 MEMRA_PROMPT_FILE=$PROSE \
    flock /tmp/gpu5090.lock timeout 2400 "$BIN" "$M" >> "$L" 2>&1
  rc=$?; { gpustate; echo "PROBE $TAG $ARM rc=$rc"; } >> "$L" 2>&1; echo "probe $TAG $ARM rc=$rc"
  ;;
spec27|spec9)
  # run-spec e2e at a fixed K, one rep (interleave arms/K at the caller level).
  K=${2:?K}; CLASS=${3:?prose|code}; REP=${4:?rep}; ARM=${5:?pre|post}
  if [ "$PHASE" = spec27 ]; then M=$M27; D=$D27; TAG=q27; else M=$M9; D=$D9; TAG=q9; fi
  BIN=$([ "$ARM" = pre ] && echo "$PRE/run-spec" || echo "$W/target/release/run-spec")
  P=$([ "$CLASS" = prose ] && echo "$PROSE" || echo "$CODE")
  L=$R/spec-$TAG-k$K-$CLASS-r$REP-$ARM.log
  { echo "tree $(git -C "$W" rev-parse HEAD) arm=$ARM k=$K"; gpustate; } > "$L" 2>&1
  MEMRA_MTP_DRAFT=$D MEMRA_PROMPT_FILE=$P MEMRA_NGEN=256 MEMRA_SPEC_K=$K MEMRA_SPEC_STATS=1 \
    flock /tmp/gpu5090.lock timeout 3600 "$BIN" "$M" >> "$L" 2>&1
  rc=$?; { gpustate; echo "SPEC $TAG k$K $CLASS r$REP $ARM rc=$rc"; } >> "$L" 2>&1
  echo "spec $TAG k$K $CLASS r$REP $ARM rc=$rc PASS=$(grep -c 'self-consistency: PASS' "$L") FAIL=$(grep -ci 'self-consistency: FAIL' "$L")"
  ;;
esac
echo "FIX2-$PHASE-DONE $(date -u +%FT%TZ)"
