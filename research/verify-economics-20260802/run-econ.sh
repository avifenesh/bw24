#!/bin/bash
# verify-economics battery (lane/verify-economics, 2026-08-02, RTX 5090 Laptop 24GB).
# ECONOMICS FIRST: per-step decode(T=1) vs verify(T=2..6) cost (spec-econ probe, fixed
# position, interleaved arms, sync-bounded), then the REAL-loop decomposition + acceptance
# (run-spec K=1..8, MEMRA_SPEC_PHASE=1) on prose (board-2048) vs code (p1-code-short).
# EVERY GPU-touching process under flock /tmp/gpu5090.lock (three lanes share the rig;
# one flock per process = short holds). Usage: run-econ.sh <phase> [rep]
set -u
PHASE=${1:?usage: run-econ.sh <econ27|econ27code|econ35|sweep27prose N|sweep27code N|sweep35prose N> }
W=/home/avifenesh/projects/wt-verify-economics
R=$W/research/verify-economics-20260802/logs
M27=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
D27=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/draft-daily-owntrim-nvfp4head-q4blk.gguf
M35=/data/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
D35=/data/ai-ml/hf-models/qwen36-35b-moe/draft-35b-owntrim-nvfp4head-q4blk.gguf
PROSE=$W/research/e2e/prompts/board-2048.txt
CODE=$W/research/e2e/prompts/p1-code-short.txt
mkdir -p "$R"

gpustate() {
  echo "[gpu $(date -u +%FT%TZ)] $(nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm,memory.used --format=csv,noheader)"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | sed 's/^/[apps] /'
}

case $PHASE in
econ27)
  L=$R/econ-q27-board.log
  { echo "tree $(git -C "$W" rev-parse HEAD)"; gpustate; } > "$L" 2>&1
  MEMRA_ECON_N=50 MEMRA_ECON_TMAX=6 MEMRA_PROMPT_FILE=$PROSE \
    flock /tmp/gpu5090.lock timeout 1800 "$W/target/release/spec-econ" "$M27" >> "$L" 2>&1
  rc=$?; { gpustate; echo "ECON27 rc=$rc"; } >> "$L" 2>&1; echo "econ27 rc=$rc"
  ;;
econ27code)
  L=$R/econ-q27-code.log
  { echo "tree $(git -C "$W" rev-parse HEAD)"; gpustate; } > "$L" 2>&1
  MEMRA_ECON_N=50 MEMRA_ECON_TMAX=6 MEMRA_PROMPT_FILE=$CODE \
    flock /tmp/gpu5090.lock timeout 1800 "$W/target/release/spec-econ" "$M27" >> "$L" 2>&1
  rc=$?; { gpustate; echo "ECON27CODE rc=$rc"; } >> "$L" 2>&1; echo "econ27code rc=$rc"
  ;;
econ35)
  L=$R/econ-q35-board.log
  { echo "tree $(git -C "$W" rev-parse HEAD)"; gpustate; } > "$L" 2>&1
  MEMRA_ECON_N=50 MEMRA_ECON_TMAX=6 MEMRA_PROMPT_FILE=$PROSE \
    flock /tmp/gpu5090.lock timeout 1800 "$W/target/release/spec-econ" "$M35" >> "$L" 2>&1
  rc=$?; { gpustate; echo "ECON35 rc=$rc"; } >> "$L" 2>&1; echo "econ35 rc=$rc"
  ;;
sweep27prose)
  REP=${2:?rep}
  L=$R/sweep-q27-prose-r$REP.log
  { echo "tree $(git -C "$W" rev-parse HEAD)"; gpustate; } > "$L" 2>&1
  MEMRA_MTP_DRAFT=$D27 MEMRA_PROMPT_FILE=$PROSE MEMRA_NGEN=256 MEMRA_SPEC_PHASE=1 MEMRA_SPEC_STATS=1 \
    flock /tmp/gpu5090.lock timeout 3600 "$W/target/release/run-spec" "$M27" >> "$L" 2>&1
  rc=$?; { gpustate; echo "SWEEP27PROSE rep=$REP rc=$rc"; } >> "$L" 2>&1
  echo "sweep27prose r$REP rc=$rc PASS=$(grep -c 'self-consistency: PASS' "$L") FAIL=$(grep -ci 'self-consistency: FAIL' "$L")"
  ;;
sweep27code)
  REP=${2:?rep}
  L=$R/sweep-q27-code-r$REP.log
  { echo "tree $(git -C "$W" rev-parse HEAD)"; gpustate; } > "$L" 2>&1
  MEMRA_MTP_DRAFT=$D27 MEMRA_PROMPT_FILE=$CODE MEMRA_NGEN=256 MEMRA_SPEC_PHASE=1 MEMRA_SPEC_STATS=1 \
    flock /tmp/gpu5090.lock timeout 3600 "$W/target/release/run-spec" "$M27" >> "$L" 2>&1
  rc=$?; { gpustate; echo "SWEEP27CODE rep=$REP rc=$rc"; } >> "$L" 2>&1
  echo "sweep27code r$REP rc=$rc PASS=$(grep -c 'self-consistency: PASS' "$L") FAIL=$(grep -ci 'self-consistency: FAIL' "$L")"
  ;;
sweep35prose)
  REP=${2:?rep}
  L=$R/sweep-q35-prose-r$REP.log
  { echo "tree $(git -C "$W" rev-parse HEAD)"; gpustate; } > "$L" 2>&1
  MEMRA_MTP_DRAFT=$D35 MEMRA_PROMPT_FILE=$PROSE MEMRA_NGEN=256 MEMRA_SPEC_PHASE=1 MEMRA_SPEC_STATS=1 \
    flock /tmp/gpu5090.lock timeout 3600 "$W/target/release/run-spec" "$M35" >> "$L" 2>&1
  rc=$?; { gpustate; echo "SWEEP35PROSE rep=$REP rc=$rc"; } >> "$L" 2>&1
  echo "sweep35prose r$REP rc=$rc PASS=$(grep -c 'self-consistency: PASS' "$L") FAIL=$(grep -ci 'self-consistency: FAIL' "$L")"
  ;;
msweep)
  # per-shape m-curve of the trunk batched matvec (DRAM-cold, copies=8): the b-tier cliff
  # attribution. usage: run-econ.sh msweep <q27|q35> <tensor>
  MODEL_TAG=${2:?msweep needs q27|q35}; TEN=${3:?msweep needs tensor}
  M=$([ "$MODEL_TAG" = q27 ] && echo "$M27" || echo "$M35")
  L=$R/msweep-$MODEL_TAG-$(echo "$TEN" | tr '/.' '--').log
  { echo "tree $(git -C "$W" rev-parse HEAD)"; gpustate; } > "$L" 2>&1
  MSWEEP_TENSOR=$TEN MSWEEP_COPIES=8 \
    flock /tmp/gpu5090.lock timeout 1800 "$W/target/release/mvq-msweep" "$M" >> "$L" 2>&1
  rc=$?; { gpustate; echo "MSWEEP $MODEL_TAG $TEN rc=$rc"; } >> "$L" 2>&1
  echo "msweep $MODEL_TAG $TEN rc=$rc"; grep -E "m=|weight " "$L" | tail -14
  ;;
nsysarm)
  # kernel-level attribution of one probe arm: nsys the whole probe process (prime included,
  # but the arm loop dominates: N iterations of ONE arm), then cuda_gpu_kern_sum.
  # usage: run-econ.sh nsysarm <q27|q35> <decode_h|verify_tN>
  MODEL_TAG=${2:?nsysarm needs q27|q35}; ARM=${3:?nsysarm needs arm}
  M=$([ "$MODEL_TAG" = q27 ] && echo "$M27" || echo "$M35")
  OUT=$R/nsys-$MODEL_TAG-$ARM
  L=$OUT.log
  { echo "tree $(git -C "$W" rev-parse HEAD)"; gpustate; } > "$L" 2>&1
  MEMRA_ECON_N=15 MEMRA_ECON_TMAX=6 MEMRA_ECON_ONLY=$ARM MEMRA_PROMPT_FILE=$PROSE \
    flock /tmp/gpu5090.lock timeout 1800 nsys profile -o "$OUT" --force-overwrite=true \
    "$W/target/release/spec-econ" "$M" >> "$L" 2>&1
  rc=$?; { gpustate; echo "NSYSARM $MODEL_TAG $ARM rc=$rc"; } >> "$L" 2>&1
  nsys stats --report cuda_gpu_kern_sum --format csv -o "$OUT" "$OUT.nsys-rep" >> "$L" 2>&1
  echo "nsysarm $MODEL_TAG $ARM rc=$rc"
  ;;
nsysspec)
  # live-loop attribution: run-spec at one K, cudaProfilerApi capture (MEMRA_PROFILE_SPEC=1
  # brackets the generate_spec call). usage: run-econ.sh nsysspec <q27|q35> <K>
  MODEL_TAG=${2:?nsysspec needs q27|q35}; K=${3:?nsysspec needs K}
  M=$([ "$MODEL_TAG" = q27 ] && echo "$M27" || echo "$M35")
  D=$([ "$MODEL_TAG" = q27 ] && echo "$D27" || echo "$D35")
  OUT=$R/nsys-spec-$MODEL_TAG-k$K
  L=$OUT.log
  { echo "tree $(git -C "$W" rev-parse HEAD)"; gpustate; } > "$L" 2>&1
  MEMRA_PROFILE_SPEC=1 MEMRA_SPEC_K=$K MEMRA_MTP_DRAFT=$D MEMRA_PROMPT_FILE=$PROSE MEMRA_NGEN=96 \
    flock /tmp/gpu5090.lock timeout 1800 nsys profile -o "$OUT" --force-overwrite=true \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    "$W/target/release/run-spec" "$M" >> "$L" 2>&1
  rc=$?; { gpustate; echo "NSYSSPEC $MODEL_TAG K=$K rc=$rc"; } >> "$L" 2>&1
  nsys stats --report cuda_gpu_kern_sum --report cuda_gpu_mem_time_sum --format csv -o "$OUT" "$OUT.nsys-rep" >> "$L" 2>&1
  echo "nsysspec $MODEL_TAG K=$K rc=$rc"
  ;;
esac
echo "ECON-$PHASE-DONE $(date -u +%FT%TZ)"
