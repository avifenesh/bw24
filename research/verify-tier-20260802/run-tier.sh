#!/bin/bash
# verify-tier diagnosis battery (lane/verify-tier, 2026-08-02, RTX 5090 Laptop 24GB sm_120a).
# Lever #3 (perf-frontier REPORT.md §4): DIAGNOSIS ONLY — no kernel changes. Names which
# kernels carry the b-tier verify premium (vT2/3/4 = 1.13/1.19/1.30x), what limits them
# (BW% / occupancy / stalls), and prices the µs/verify-column curve at T=1..9 on q27
# (gemma-class dense NVFP4 trunk) and q9 (NVFP4 + k-quant mix).
#
# GPU-sharing protocol: EVERY GPU-touching process under its own short
# `flock /tmp/gpu5090.lock` hold — one flock per run, released between runs, never one
# lock around a phase. Builds/parsing happen without the lock.
# ncu needs sudo on this rig (RmProfilingAdminOnly=1); ncu locks clocks to base by
# default — ncu cells are BASE-CLOCK cells, labeled as such, never compared to boost cells.
set -u
PHASE=${1:?usage: run-tier.sh <probe27|probe9|nsys27 ARM|nsys9 ARM|ncu27 TAG|ncu9 TAG|sweep9 CLASS REP>}
W=/home/avifenesh/projects/wt-verify-tier
R=$W/research/verify-tier-20260802/logs
NC=$W/research/verify-tier-20260802/ncu
M27=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
D27=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/draft-daily-owntrim-nvfp4head-q4blk.gguf
M9=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
D9=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/draft-9b-owntrim-nvfp4head-q4blk.gguf
PROSE=$W/research/e2e/prompts/board-2048.txt
CODE=$W/research/e2e/prompts/p1-code-short.txt
NCU=/usr/local/cuda-13.1/bin/ncu
mkdir -p "$R" "$NC"

gpustate() {
  echo "[gpu $(date -u +%FT%TZ)] $(nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm,memory.used --format=csv,noheader)"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | sed 's/^/[apps] /'
}

# ncu budget section set (H100-lane lesson: full = 20 replay passes/launch; this is ~12):
NCUSEC="--section SpeedOfLight --section MemoryWorkloadAnalysis --section SchedulerStats --section WarpStateStats --section Occupancy --section LaunchStats"

case $PHASE in
probe27|probe9)
  # Cost curve: decode(T=1) vs verify_t1..t9 (t9 = the K=8 off-tier cliff: NVFP4 has no b16
  # twin -> grid.y=m per-row MMVQ). Same fixed-position interleaved-arm probe as
  # verify-economics (N=50 + 3 warmups, sync-bounded, rollback outside the timed region).
  if [ "$PHASE" = probe27 ]; then M=$M27; TAG=q27; else M=$M9; TAG=q9; fi
  L=$R/probe-$TAG-t9.log
  { echo "tree $(git -C "$W" rev-parse HEAD)"; gpustate; } > "$L" 2>&1
  MEMRA_ECON_N=50 MEMRA_ECON_TMAX=9 MEMRA_PROMPT_FILE=$PROSE \
    flock /tmp/gpu5090.lock timeout 2400 "$W/target/release/spec-econ" "$M" >> "$L" 2>&1
  rc=$?; { gpustate; echo "PROBE $TAG rc=$rc"; } >> "$L" 2>&1; echo "probe $TAG rc=$rc"
  ;;
nsys27|nsys9)
  # Per-kernel share of ONE arm's pass (decode_h or verify_tN): nsys the single-arm probe,
  # kern-sum CSV. Arm-only kernels (the batched b-tier) are attributable directly; shared
  # kernels are compared decode_h-vs-verify_tN at equal iteration count (N=15+3).
  ARM=${2:?nsys needs arm: decode_h|verify_tN}
  if [ "$PHASE" = nsys27 ]; then M=$M27; TAG=q27; else M=$M9; TAG=q9; fi
  TM=9; [ "$ARM" != decode_h ] && TM=${ARM#verify_t}
  OUT=$R/nsys-$TAG-$ARM
  L=$OUT.log
  { echo "tree $(git -C "$W" rev-parse HEAD)"; gpustate; } > "$L" 2>&1
  MEMRA_ECON_N=15 MEMRA_ECON_TMAX=$TM MEMRA_ECON_ONLY=$ARM MEMRA_PROMPT_FILE=$PROSE \
    flock /tmp/gpu5090.lock timeout 1800 nsys profile -o "$OUT" --force-overwrite=true \
    "$W/target/release/spec-econ" "$M" >> "$L" 2>&1
  rc=$?; { gpustate; echo "NSYS $TAG $ARM rc=$rc"; } >> "$L" 2>&1
  nsys stats --report cuda_gpu_kern_sum --format csv -o "$OUT" "$OUT.nsys-rep" >> "$L" 2>&1
  echo "nsys $TAG $ARM rc=$rc"
  ;;
ncu27|ncu9)
  # Limiter analysis of the premium carriers: sudo ncu on the single-arm probe, kernel-name
  # regex + launch-skip past one steady pass. BASE-CLOCK cells (ncu clock control).
  # usage: run-tier.sh ncu27 <tag> <arm> <kregex> <skip> <count>
  TAG2=${2:?ncu needs tag}; ARM=${3:?arm}; KRE=${4:?kernel regex}; SKIP=${5:?skip}; CNT=${6:?count}
  if [ "$PHASE" = ncu27 ]; then M=$M27; TAG=q27; else M=$M9; TAG=q9; fi
  TM=9; [ "$ARM" != decode_h ] && TM=${ARM#verify_t}
  OUT=$NC/ncu-$TAG-$TAG2
  L=$OUT.log
  { echo "tree $(git -C "$W" rev-parse HEAD)"; gpustate; } > "$L" 2>&1
  flock /tmp/gpu5090.lock timeout 3000 sudo -n env \
    MEMRA_ECON_N=6 MEMRA_ECON_TMAX=$TM MEMRA_ECON_ONLY=$ARM MEMRA_PROMPT_FILE=$PROSE \
    "$NCU" -k "regex:$KRE" --launch-skip "$SKIP" --launch-count "$CNT" \
    $NCUSEC --clock-control base -o "$OUT" -f \
    "$W/target/release/spec-econ" "$M" >> "$L" 2>&1
  rc=$?
  sudo -n chown "$USER:$USER" "$OUT.ncu-rep" 2>/dev/null
  { gpustate; echo "NCU $TAG $TAG2 rc=$rc"; } >> "$L" 2>&1
  "$NCU" --import "$OUT.ncu-rep" --csv --page raw > "$OUT-raw.csv" 2>>"$L"
  echo "ncu $TAG $TAG2 rc=$rc"
  ;;
sweep9)
  # q9 acceptance-vs-K receipts (for the K=4-5 unlock pricing; acceptance is
  # greedy-deterministic, timings from a single rep are labeled single-run).
  CLASS=${2:?sweep9 needs prose|code}; REP=${3:?rep}
  P=$([ "$CLASS" = prose ] && echo "$PROSE" || echo "$CODE")
  L=$R/sweep-q9-$CLASS-r$REP.log
  { echo "tree $(git -C "$W" rev-parse HEAD)"; gpustate; } > "$L" 2>&1
  MEMRA_MTP_DRAFT=$D9 MEMRA_PROMPT_FILE=$P MEMRA_NGEN=256 MEMRA_SPEC_PHASE=1 MEMRA_SPEC_STATS=1 \
    flock /tmp/gpu5090.lock timeout 3600 "$W/target/release/run-spec" "$M9" >> "$L" 2>&1
  rc=$?; { gpustate; echo "SWEEP9 $CLASS r$REP rc=$rc"; } >> "$L" 2>&1
  echo "sweep9 $CLASS r$REP rc=$rc PASS=$(grep -c 'self-consistency: PASS' "$L") FAIL=$(grep -ci 'self-consistency: FAIL' "$L")"
  ;;
esac
echo "TIER-$PHASE-DONE $(date -u +%FT%TZ)"
