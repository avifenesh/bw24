#!/bin/bash
# inc3 (3a) chunk-size sweep: 32 sequences advanced per tick via ceil(32/C) chunked
# decode_step_batch calls — the per-tick cost of chunking policy C. One arm per process
# (env-dependent dispatch reads once); 5 INTERLEAVED rounds, median per arm at parse time.
# Arms:
#   c8-naked  : chunk 8, naked (deployment baseline)
#   c8-q8rp   : chunk 8 + mirror (mirror-overhead control)
#   c16-q8rp  : chunk 16 + mirror (the EXACT-16 tier, auto verify_exact scope)
#   c16-gemm  : chunk 16 naked via the cap door (NON-exact GEMM tier — the exactness-tax ref)
#   c32-gemm  : chunk 32 via the cap door (NON-exact GEMM tier at 32)
set -u
W=/home/avifenesh/projects/wt-batched-tick-3
R=$W/research/batched-tick-inc3-20260801
M=/home/avifenesh/models/qwen3.5-9b-judge-q8_0.gguf
B=$W/target/release/decode-batch-bench
OUT=$R/chunksweep.log
: >"$OUT"
for round in 1 2 3 4 5; do
  echo "--- round $round $(date -u +%FT%TZ)" | tee -a "$OUT"
  for arm in c8-naked c8-q8rp c16-q8rp c16-gemm c32-gemm; do
    case $arm in
      c8-naked) ENVV=""; C=8;;
      c8-q8rp)  ENVV="MEMRA_Q8RP=1"; C=8;;
      c16-q8rp) ENVV="MEMRA_Q8RP=1"; C=16;;
      c16-gemm) ENVV="MEMRA_DECODE_BATCH_CAP=16"; C=16;;
      c32-gemm) ENVV="MEMRA_DECODE_BATCH_CAP=32"; C=32;;
    esac
    line=$(flock /tmp/gpu5090.lock env $ENVV "$B" "$M" --seqs 32 --chunk "$C" --steps 128 --reps 1 2>&1 | grep -E "CHUNKSWEEP|out of memory|Error" | tail -1)
    echo "round=$round arm=$arm $line" | tee -a "$OUT"
  done
done
echo CHUNKSWEEP-DONE | tee -a "$OUT"
