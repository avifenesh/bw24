#!/bin/bash
# q27 Q8_0 extreme-perf deep dive, PHASE 1 (plain decode + prefill + serve).
# Rig: pro6000wk-runpod-community (RTX PRO 6000 Blackwell WK 96GB, 188 SM, driver 570.211.01,
#      510W cap, mem clock droops to 13365/14001 => ~1711 GB/s effective, 89C under spin).
#      BOARD CAVEAT: community board ~9% below the prod board at identical code. RELATIVE
#      deltas are the currency; absolute rows get re-minted on prod-class silicon.
# GPU is ours alone on this pod (no flock needed) but every phase still records gpustate.
set -u
PHASE=${1:?usage: dd.sh <phase> [args]}
W=/root/bw24
R=/root/receipts-dd
Q8=/root/models/Qwen3.6-27B-Q8_0.gguf
NV=/root/models/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
export LD_LIBRARY_PATH=/usr/local/cuda-13.1/compat
export PATH=$HOME/.cargo/bin:/usr/local/cuda-13.1/bin:$PATH
mkdir -p "$R"/{nsys,logs,ncu}

gpustate() {
  echo "[gpu $(date -u +%FT%TZ)] $(nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used --format=csv,noheader)"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | sed 's/^/[apps] /'
}

# ncu section set (H100-lane lesson: the full set is ~20 replay passes/launch; this is ~12).
NCUSEC="--section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy --section LaunchStats --section SchedulerStats"

# The EXACT prod-anchor prompt (research/e2e/prompts/pp512.txt = 512 tokens) so every cell here
# shares a denominator with the 20260804 prod rows (4591 pp512 / 52.61 d512).
PROMPT=$W/research/e2e/prompts/pp512.txt
PLONG=$W/research/e2e/prompts/p3-agentic-long.txt

case $PHASE in

# ---------------------------------------------------------------- gates before numbers
gate)
  L=$R/logs/gate-rungen-argmax-q8.log
  { gpustate; } > "$L" 2>&1
  MEMRA_PROMPT_FILE=$PROMPT MEMRA_NGEN=48 \
    timeout 1800 "$W/target/release/run-gen" "$Q8" >> "$L" 2>&1
  rc=$?; { gpustate; echo "GATE-ARGMAX rc=$rc"; } >> "$L" 2>&1
  echo "gate rc=$rc"; grep -E "MATCH|MISMATCH|tok/s" "$L" | tail -5
  ;;

# ---------------------------------------------------------------- baseline anchor cells
# Re-mint the prod anchor cells on THIS board so every A/B has a same-board denominator.
anchor)
  REP=${2:?anchor needs rep}
  for cell in d512 pp512; do
    L=$R/logs/anchor-q8-$cell-r$REP.log
    { gpustate; } > "$L" 2>&1
    case $cell in
      d512) MEMRA_PROMPT_FILE=$PROMPT MEMRA_NGEN=128 \
              timeout 1800 "$W/target/release/run-gen" "$Q8" >> "$L" 2>&1 ;;
      pp512) MEMRA_PROMPT_FILE=$PROMPT MEMRA_PP_ONLY=1 MEMRA_PP_REPS=3 \
              timeout 1800 "$W/target/release/run-gen" "$Q8" >> "$L" 2>&1 ;;
    esac
    rc=$?; { gpustate; echo "ANCHOR $cell r$REP rc=$rc"; } >> "$L" 2>&1
    echo "anchor $cell r$REP rc=$rc $(grep -E 'tok/s' "$L" | tail -1)"
  done
  ;;

# ---------------------------------------------------------------- 1. nsys kernel shares
# DECODE c=1: MEMRA_PROFILE_GEN=2 starts the capture AT the decode loop (prime excluded) —
# the 2026-07-10 lesson: window-cutting a whole-run capture misattributes prime + the
# argmax-gate loop into the decode share map.
nsys-decode)
  OUT=$R/nsys/nsys-q8-decode-c1; L=$OUT.log
  { gpustate; } > "$L" 2>&1
  MEMRA_PROMPT_FILE=$PROMPT MEMRA_NGEN=128 MEMRA_PROFILE_GEN=2 \
    timeout 2400 nsys profile -o "$OUT" --force-overwrite=true -c cudaProfilerApi \
    --trace=cuda,nvtx --cuda-memory-usage=false \
    "$W/target/release/run-gen" "$Q8" >> "$L" 2>&1
  rc=$?; { gpustate; echo "NSYS decode-c1 rc=$rc"; } >> "$L" 2>&1
  nsys stats --report cuda_gpu_kern_sum --format csv -o "$OUT" "$OUT.nsys-rep" >> "$L" 2>&1
  nsys stats --report cuda_gpu_trace  --format csv -o "$OUT-trace" "$OUT.nsys-rep" >> "$L" 2>&1
  echo "nsys decode-c1 rc=$rc"
  ;;

# PREFILL pp512: MEMRA_PP_ONLY returns BEFORE run_gen's MEMRA_PROFILE_GEN bracket (it is its own
# early-exit arm), so `-c cudaProfilerApi` captures nothing. It does not need the bracket: the
# PP_ONLY arm runs ONLY warmup+reps prefill forwards then exits, so the whole-process timeline IS
# prefill. WARMUP=1 REPS=1 => exactly 2 identical forwards, and any load-phase kernel is
# identifiable offline by an instance count that is not a multiple of 2 x 64 layers.
nsys-prefill)
  CH=${2:-default}
  OUT=$R/nsys/nsys-q8-pp512-$CH; L=$OUT.log
  { gpustate; } > "$L" 2>&1
  ENVX=""; [ "$CH" != default ] && ENVX="MEMRA_PRIME_CHUNK=$CH"
  env $ENVX MEMRA_PROMPT_FILE=$PROMPT MEMRA_PP_ONLY=1 MEMRA_PP_REPS=1 MEMRA_PP_WARMUP=1 \
    timeout 2400 nsys profile -o "$OUT" --force-overwrite=true \
    --trace=cuda,nvtx --cuda-memory-usage=false \
    "$W/target/release/run-gen" "$Q8" >> "$L" 2>&1
  rc=$?; { gpustate; echo "NSYS pp512 $CH rc=$rc"; } >> "$L" 2>&1
  nsys stats --report cuda_gpu_kern_sum --format csv -o "$OUT" "$OUT.nsys-rep" >> "$L" 2>&1
  nsys stats --report cuda_gpu_trace  --format csv -o "$OUT-trace" "$OUT.nsys-rep" >> "$L" 2>&1
  echo "nsys pp512 $CH rc=$rc"
  ;;

# DECODE c=8: the batched tick at the saturation point. decode-batch-bench has no profiler
# bracket, so the capture is whole-process and the prime phase is separated OFFLINE by the
# kernel-name split (prime uses mul_mat_q/gemm classes absent from the m=8 decode tick) —
# plus the --reps window makes the decode loop ~95% of the timeline.
nsys-decode8)
  OUT=$R/nsys/nsys-q8-decode-c8; L=$OUT.log
  { gpustate; } > "$L" 2>&1
  timeout 3000 nsys profile -o "$OUT" --force-overwrite=true \
    --trace=cuda,nvtx --cuda-memory-usage=false \
    "$W/target/release/decode-batch-bench" "$Q8" --steps 128 --reps 3 --batches 8 --ctx 512 \
    >> "$L" 2>&1
  rc=$?; { gpustate; echo "NSYS decode-c8 rc=$rc"; } >> "$L" 2>&1
  nsys stats --report cuda_gpu_kern_sum --format csv -o "$OUT" "$OUT.nsys-rep" >> "$L" 2>&1
  echo "nsys decode-c8 rc=$rc"
  ;;

# The serial-fraction question for c=8, via the in-tree sync-bounded phase accumulator.
# Sync-bounded => the TOTAL inflates; the value is the RANKING/shares (header note).
phase8)
  B=${2:-8}
  L=$R/logs/phase-q8-c$B.log
  { gpustate; } > "$L" 2>&1
  MEMRA_BATCH_PHASE=1 \
    timeout 3000 "$W/target/release/decode-batch-bench" "$Q8" \
    --steps 64 --reps 2 --batches "$B" --ctx 512 >> "$L" 2>&1
  rc=$?; { gpustate; echo "PHASE c=$B rc=$rc"; } >> "$L" 2>&1
  echo "phase c=$B rc=$rc"; sed -n '/batch-phase/,$p' "$L"
  ;;

# c=1 vs c=8 batched-tick scaling on THIS board (the batch-lever denominator).
batchscale)
  REP=${2:?batchscale needs rep}
  L=$R/logs/batchscale-q8-r$REP.log
  { gpustate; } > "$L" 2>&1
  timeout 3600 "$W/target/release/decode-batch-bench" "$Q8" \
    --steps 128 --reps 3 --batches 1,2,4,8 --ctx 512 >> "$L" 2>&1
  rc=$?; { gpustate; echo "BATCHSCALE r$REP rc=$rc"; } >> "$L" 2>&1
  echo "batchscale r$REP rc=$rc"; grep -E "^B=|^scale" "$L"
  ;;

# ---------------------------------------------------------------- 2. levers
# MEMRA_PRIME_CHUNK sweep at prefill (96GB card has no transient pressure -> is monolithic
# actually the fastest, or does a chunk size win on L2 reuse?). Interleaved by rep.
lever-chunk)
  REP=${2:?lever-chunk needs rep}
  for ch in default 0 1024 2048 4096 8192; do
    L=$R/logs/lever-chunk-$ch-r$REP.log
    ENVX=""; [ "$ch" != default ] && ENVX="MEMRA_PRIME_CHUNK=$ch"
    { gpustate; echo "chunk=$ch"; } > "$L" 2>&1
    env $ENVX MEMRA_PROMPT_FILE=$PROMPT MEMRA_PP_ONLY=1 MEMRA_PP_REPS=3 \
      timeout 1800 "$W/target/release/run-gen" "$Q8" >> "$L" 2>&1
    rc=$?; { gpustate; echo "LEVER-CHUNK $ch r$REP rc=$rc"; } >> "$L" 2>&1
    echo "chunk=$ch r$REP rc=$rc $(grep 'MEDIAN' "$L" | tail -1)"
  done
  ;;

# CHUNK sweep where chunking ACTUALLY ENGAGES: pp512 is a null test (every chunk >= 1024 is
# monolithic on a 512-token prompt — measured flat 4137-4174, spread 0.9%). p3-agentic-long is
# 6257 tokens, so 1024/2048/4096 are real chunk counts.
lever-chunk-long)
  REP=${2:?needs rep}
  for ch in default 0 1024 2048 4096; do
    L=$R/logs/lever-chunklong-$ch-r$REP.log
    ENVX=""; [ "$ch" != default ] && ENVX="MEMRA_PRIME_CHUNK=$ch"
    { gpustate; echo "chunk=$ch prompt=p3-agentic-long"; } > "$L" 2>&1
    env $ENVX MEMRA_PROMPT_FILE=$PLONG MEMRA_PP_ONLY=1 MEMRA_PP_REPS=3 \
      timeout 1800 "$W/target/release/run-gen" "$Q8" >> "$L" 2>&1
    rc=$?; { gpustate; echo "LEVER-CHUNKLONG $ch r$REP rc=$rc"; } >> "$L" 2>&1
    echo "chunklong=$ch r$REP rc=$rc $(grep 'MEDIAN' "$L" | tail -1)"
  done
  ;;

# ---------------------------------------------------------------- ncu limiter analysis
# Achieved DRAM throughput + occupancy of the kernels that own the tick. ncu locks clocks to
# BASE by default => these are BASE-CLOCK cells, never compared against the boost anchor rows;
# the %-of-peak they yield is read against the ncu-reported DRAM peak in the same report.
ncu-decode)
  KRE=${2:-qmatvec_q8_0}
  TAG=${3:-mmvq}
  OUT=$R/ncu/ncu-q8-decode-$TAG; L=$OUT.log
  { gpustate; } > "$L" 2>&1
  MEMRA_PROMPT_FILE=$PROMPT MEMRA_NGEN=6 \
    timeout 3000 /usr/local/cuda-13.1/bin/ncu -k "regex:$KRE" \
    --launch-skip 400 --launch-count 24 $NCUSEC --clock-control base -o "$OUT" -f \
    "$W/target/release/run-gen" "$Q8" >> "$L" 2>&1
  rc=$?; { gpustate; echo "NCU decode $TAG rc=$rc"; } >> "$L" 2>&1
  /usr/local/cuda-13.1/bin/ncu --import "$OUT.ncu-rep" --csv --page raw > "$OUT-raw.csv" 2>>"$L"
  echo "ncu decode $TAG rc=$rc"
  ;;

ncu-prefill)
  KRE=${2:-mul_mat_q_q8_0}
  TAG=${3:-mmq}
  OUT=$R/ncu/ncu-q8-prefill-$TAG; L=$OUT.log
  { gpustate; } > "$L" 2>&1
  MEMRA_PROMPT_FILE=$PROMPT MEMRA_PP_ONLY=1 MEMRA_PP_REPS=1 MEMRA_PP_WARMUP=1 \
    timeout 3000 /usr/local/cuda-13.1/bin/ncu -k "regex:$KRE" \
    --launch-skip 400 --launch-count 24 $NCUSEC --clock-control base -o "$OUT" -f \
    "$W/target/release/run-gen" "$Q8" >> "$L" 2>&1
  rc=$?; { gpustate; echo "NCU prefill $TAG rc=$rc"; } >> "$L" 2>&1
  /usr/local/cuda-13.1/bin/ncu --import "$OUT.ncu-rep" --csv --page raw > "$OUT-raw.csv" 2>>"$L"
  echo "ncu prefill $TAG rc=$rc"
  ;;

# DRIFT CONTROL for the chunk verdict. The ascending sweeps above declined MONOTONICALLY in run
# ORDER (r1 default 3976 -> r3 chunk4096 3839, every step down, both within and across reps) — that
# is a thermal-drift signature, not a chunk effect, and `default` was always measured first. This
# arm alternates A=default and B=$CH within each pass and also runs the REVERSED order (B,A), so a
# drift-free delta is the mean of the two orderings.
lever-chunk-ab)
  CH=${2:?needs chunk}; REP=${3:?needs rep}
  for ord in ab ba; do
    for arm in 1 2; do
      case $ord$arm in
        ab1|ba2) TAG=A; ENVX="" ;;
        ab2|ba1) TAG=B; ENVX="MEMRA_PRIME_CHUNK=$CH" ;;
      esac
      L=$R/logs/chunkab-$CH-$ord-$TAG-r$REP.log
      { gpustate; echo "arm=$TAG chunk=${ENVX:-default} order=$ord"; } > "$L" 2>&1
      env $ENVX MEMRA_PROMPT_FILE=$PLONG MEMRA_PP_ONLY=1 MEMRA_PP_REPS=3 \
        timeout 1800 "$W/target/release/run-gen" "$Q8" >> "$L" 2>&1
      rc=$?; { gpustate; echo "CHUNKAB $TAG $ord r$REP rc=$rc"; } >> "$L" 2>&1
      echo "chunkab $ord arm=$TAG chunk=$CH r$REP rc=$rc $(grep 'MEDIAN' "$L" | tail -1)"
    done
  done
  ;;

# ---------------------------------------------------------------- LEVER 1: q8 dense-FFN gate+up fusion
# A/B of the fused2 dense-FFN pair (MEMRA_Q8_FFN_FUSE2). Interleaved arms WITHIN each rep, both
# orderings across reps, N=5 -> drift-immune. Exactness: run-gen prints the prefill-vs-decode
# argmax gate every run (MATCH required), and the fused kernel body is qmatvec_q8_0_mmvq VERBATIM
# per (tensor,row) so the arm is also a BIT-identity claim — gated separately by `fuse-bits`.
lever-ffnfuse)
  REP=${2:?needs rep}
  # alternate which arm goes first per rep so thermal drift cancels in the pair mean
  if [ $((REP % 2)) -eq 1 ]; then ORD="0 1"; else ORD="1 0"; fi
  for a in $ORD; do
    TAG=off; [ "$a" = 1 ] && TAG=on
    L=$R/logs/ffnfuse-$TAG-r$REP.log
    { gpustate; echo "arm=$TAG MEMRA_Q8_FFN_FUSE2=$a"; } > "$L" 2>&1
    MEMRA_Q8_FFN_FUSE2=$a MEMRA_PROMPT_FILE=$PROMPT MEMRA_NGEN=128 \
      timeout 1800 "$W/target/release/run-gen" "$Q8" >> "$L" 2>&1
    rc=$?; { gpustate; echo "FFNFUSE $TAG r$REP rc=$rc"; } >> "$L" 2>&1
    echo "ffnfuse $TAG r$REP rc=$rc $(grep -oE 'generated 128 tokens in [0-9.]+s = [0-9.]+ tok/s' "$L" | head -1) | $(grep -oE '(MATCH|MISMATCH)' "$L" | head -2 | tr '\n' ' ')"
  done
  ;;

# BIT-IDENTITY arm for the fusion lever: same prompt, same 128 tokens, arms off/on — the emitted
# token stream and the final logits sha must be IDENTICAL, not merely argmax-equal.
fuse-bits)
  for a in 0 1; do
    TAG=off; [ "$a" = 1 ] && TAG=on
    L=$R/logs/fusebits-$TAG.log
    { gpustate; } > "$L" 2>&1
    MEMRA_Q8_FFN_FUSE2=$a MEMRA_PROMPT_FILE=$PROMPT MEMRA_NGEN=128 MEMRA_PRINT_TEXT=1 \
      timeout 1800 "$W/target/release/run-gen" "$Q8" >> "$L" 2>&1
    echo "fusebits $TAG rc=$? $(grep -oE 'logits sha[0-9a-z:= ]*' "$L" | head -1)"
  done
  # token-stream diff (the OUTPUT TEXT block run-gen prints under MEMRA_PRINT_TEXT)
  for a in off on; do sed -n '/OUTPUT TEXT/,$p' "$R/logs/fusebits-$a.log" > "$R/logs/fusebits-$a.txt"; done
  if diff -q "$R/logs/fusebits-off.txt" "$R/logs/fusebits-on.txt" >/dev/null; then
    echo "FUSE-BITS: token streams IDENTICAL"
  else
    echo "FUSE-BITS: token streams DIFFER"; diff "$R/logs/fusebits-off.txt" "$R/logs/fusebits-on.txt" | head -20
  fi
  ;;

esac
echo "DD-$PHASE-DONE $(date -u +%FT%TZ)"
