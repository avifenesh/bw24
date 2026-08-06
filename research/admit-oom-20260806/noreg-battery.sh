set -uo pipefail
cd /home/avifenesh/projects/wt-admitoom
OUT=research/admit-oom-20260806/logs
M=/data/ai-ml/hf-models
Q9=$M/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
D9=$M/qwen35-9b-nvfp4-gguf/draft-9b-owntrim-nvfp4head-q4blk.gguf
rc=0
echo "### serve-smoke (already OK, skipped)"; if false; then
tools/serve-smoke.sh > $OUT/serve-smoke.log 2>&1 && echo "serve-smoke OK" || { echo "serve-smoke FAIL"; rc=1; }
tail -3 $OUT/serve-smoke.log; fi
echo "### decode-batch-gate config B=8 (NVFP4)"
target/release/decode-batch-gate "$Q9" --steps 32 --batch 8 --mode config > $OUT/dbg-config.log 2>&1 \
  && grep -q "ALL GREEN" $OUT/dbg-config.log && echo "dbg config ALL GREEN" || { echo "dbg config FAIL"; tail -5 $OUT/dbg-config.log; rc=1; }
echo "### decode-batch-gate strict B=4 equalized (NVFP4)"
MEMRA_MMVQ=0 MEMRA_NO_FUSE_NORMQ=1 target/release/decode-batch-gate "$Q9" --steps 32 --batch 4 --mode strict > $OUT/dbg-strict.log 2>&1 \
  && grep -q "ALL GREEN" $OUT/dbg-strict.log && echo "dbg strict ALL GREEN" || { echo "dbg strict FAIL"; tail -5 $OUT/dbg-strict.log; rc=1; }
echo "### kernel-check"
target/release/kernel-check > $OUT/kernel-check.log 2>&1; tail -1 $OUT/kernel-check.log
grep -q "ALL GREEN" $OUT/kernel-check.log || { echo "kernel-check FAIL"; rc=1; }
echo "### run-spec K=1..8 (q9+draft)"
MEMRA_DRAFT="$D9" target/release/run-spec "$Q9" > $OUT/run-spec.log 2>&1; tail -12 $OUT/run-spec.log
exit $rc
