#!/usr/bin/env zsh
# Plain-quant reference arm pipeline: pinned-commit libggml build + full BF16 fetch + uniform Q4_K repack.
# Resumable: every stage checks its own completion marker / uses tool-native resume.
set -u
LANE=/home/avifenesh/projects/bw24-hy3lane
STAGE=$HOME/.local/share/bw24-staging/hy3-bf16-full
GGML_WT=$HOME/.local/share/bw24-staging/llama-99f3dc3
REV=716aa7241bd6d95896be4ebfc761162a9c4d49ef
PLAN=$LANE/research/per-expert-quant/plain-fullbank-uniform-q3k.plan.json
OUT=$HOME/.local/share/bw24-staging/hy3-plain-q3k-overlay
EV=$LANE/research/per-expert-quant/evidence/local-5090-plain-arm-20260725
LOG=$EV/build.log
mkdir -p $STAGE $EV $OUT
exec >> $LOG 2>&1
echo "=== plain_arm_build start $(date -u +%FT%TZ) ==="

# --- stage 1: pinned libggml-base (commit 99f3dc3, same encoder commit as served artifact) ---
GGML_LIB=""
if [ ! -f $GGML_WT/BUILD_OK ]; then
  cd $HOME/projects/llama.cpp
  git worktree add --detach $GGML_WT 99f3dc32296f825fec94f202da1e9fede1e78cf9 || true
  cd $GGML_WT
  cmake -B build-cpu -DBUILD_SHARED_LIBS=ON -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DCMAKE_BUILD_TYPE=Release
  nice -n 10 cmake --build build-cpu --target ggml-base -j 8
  rc=$?
  if [ $rc -ne 0 ]; then echo "GGML BUILD FAILED rc=$rc"; exit 1; fi
  # verify the artifact really changed/exists, not just checker output
  ls -la build-cpu/bin/libggml-base.so* || { echo "GGML LIB MISSING"; exit 1; }
  touch $GGML_WT/BUILD_OK
fi
GGML_LIB=$(ls $GGML_WT/build-cpu/bin/libggml-base.so.0.* | head -1)
GGML_SHA=$(sha256sum $GGML_LIB | cut -d' ' -f1)
echo "GGML_LIB=$GGML_LIB sha256=$GGML_SHA"

# --- stage 2: full BF16 checkpoint fetch (99 shards + index + config), resumable ---
TOKEN=$(cat /data/ai-ml/hf-models/token 2>/dev/null)
BASE="https://huggingface.co/tencent/Hy3/resolve/$REV"
fetch() {
  local f=$1
  local free_g=$(df --output=avail -BG $STAGE | tail -1 | tr -dc '0-9')
  if [ "$free_g" -lt 50 ]; then echo "DISK GUARD: ${free_g}G free, aborting"; exit 2; fi
  for try in 1 2 3 4 5; do
    curl -sfL -C - -o $STAGE/$f -H "Authorization: Bearer $TOKEN" "$BASE/$f" && return 0
    echo "retry $try: $f rc=$?"; sleep 20
  done
  echo "FETCH FAILED: $f"; return 1
}
for f in config.json model.safetensors.index.json tokenizer_config.json; do
  [ -s $STAGE/$f ] || fetch $f || exit 3
done
FAILED=0
for i in $(seq -w 1 99); do
  f="model-000${i}-of-00099.safetensors"
  if [ -f $STAGE/.done-$i ]; then continue; fi
  fetch $f || { FAILED=1; break; }
  # verify size against index-declared shard usage is impractical per-file; rely on curl -f + resume
  touch $STAGE/.done-$i
  echo "shard $i done $(date -u +%T) free=$(df --output=avail -BG $STAGE | tail -1 | tr -dc '0-9')G"
done
if [ $FAILED -ne 0 ]; then echo "DOWNLOAD INCOMPLETE"; exit 3; fi
echo "DOWNLOAD COMPLETE $(du -sh $STAGE | cut -f1)"

# --- stage 3: uniform Q4_K repack from BF16 (pinned external quantizer), resumable ---
cd $LANE
nice -n 10 python3 tools/prepare_mixed_expert_repack.py prepare \
  $STAGE $OUT \
  --plan $PLAN \
  --fallback-dir $STAGE \
  --workers 8 \
  --resume \
  --ggml-lib $GGML_LIB \
  --ggml-lib-sha256 $GGML_SHA \
  --ggml-source-commit 99f3dc32296f825fec94f202da1e9fede1e78cf9
rc=$?
if [ $rc -ne 0 ]; then echo "REPACK FAILED rc=$rc"; exit 4; fi
[ -s $OUT/manifest.json ] || { echo "REPACK MANIFEST MISSING"; exit 4; }
echo "PLAIN ARM OVERLAY DONE $(du -sh $OUT | cut -f1)"
echo "PLAIN_ARM_BUILD_DONE $(date -u +%FT%TZ)" > $EV/plain-arm-overlay-done.marker
