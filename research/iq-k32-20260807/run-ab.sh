#!/bin/bash
# iq-k32 lane: exactness byte-compare + N=5 interleaved perf A/B, k16 vs k32 builds.
# Two md5-pinned binaries (/tmp/iqk32-bins/run-gen-k16, run-gen-k32) built from the SAME
# commit, differing only in the MEMRA_IQEXP_K16 compile seam. All GPU work under ONE
# flock hold per phase (benchmarks.md window law).
#
# Phase logits: MEMRA_PP_ONLY + MEMRA_PP_LOGITS on gemma (expert kernel) + KAT (dense
#   kernel) for both arms -> byte compare answers the bit-identity question on real weights.
# Phase perf: pp-only MEDIAN of 3 reps per binary per model, N=5 adjacent alternating pairs
#   (k16,k32),(k32,k16),... in one lock hold.
set -u
W=/home/avifenesh/projects/wt-iqexp
R=$W/research/iq-k32-20260807
RAW=$R/raw
B=/tmp/iqk32-bins
GEMMA=/data/ai-ml/hf-models/gemma4-26b-a4b-qat-gguf/gemma-4-26B_q4_0-it.gguf
KAT=/data/ai-ml/hf-models/kat-coder-v25-dev-gguf/Kwaipilot_KAT-Coder-V2.5-Dev-IQ4_XS.gguf
P2048=$W/research/depth-decode-20260802/depth-2048-kat.txt
mkdir -p "$RAW" /tmp/iqk32-logits

logits_point() { # arm model tag
  local arm=$1 model=$2 tag=$3
  local log=$RAW/pplogits-$tag-$arm.log
  MEMRA_PP_ONLY=1 MEMRA_PP_REPS=1 MEMRA_PP_WARMUP=0 \
    MEMRA_PP_LOGITS=/tmp/iqk32-logits/$tag-$arm.bin \
    MEMRA_PROMPT_FILE="$P2048" \
    timeout 1200 "$B/run-gen-$arm" "$model" > "$log" 2>&1
  echo "$tag $arm rc=$? $(grep -ao 'pp-only prime logits -> [^ ]* ([0-9]* f32)' "$log" | tail -1)"
}

perf_point() { # arm model tag pairidx
  local arm=$1 model=$2 tag=$3 pair=$4
  local log=$RAW/perf-$tag-$arm-p$pair.log
  MEMRA_PP_ONLY=1 MEMRA_PP_REPS=3 MEMRA_PP_WARMUP=1 MEMRA_PROMPT_FILE="$P2048" \
    timeout 1200 "$B/run-gen-$arm" "$model" > "$log" 2>&1
  local med
  med=$(grep -ao "pp-only MEDIAN: .* = [0-9.]* tok/s" "$log" | grep -oE "[0-9.]+ tok/s" | grep -oE "^[0-9.]+")
  local temp
  temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)
  echo "{\"cell\":\"iqk32-perf\",\"model\":\"$tag\",\"arm\":\"$arm\",\"pair\":$pair,\"metric\":\"pp2048_median3\",\"value\":${med:-null},\"temp\":$temp,\"ts\":\"$(date -u +%FT%TZ)\"}" >> "$R/perf-ab.jsonl"
  echo "$tag $arm pair$pair median=${med:-FAIL} temp=$temp"
}

case "${1:-help}" in
  logits)
    flock /tmp/gpu5090.lock bash -c "
      set -u
      $(declare -f logits_point)
      B=$B; RAW=$RAW; P2048=$P2048
      logits_point k32 $GEMMA gemma
      logits_point k16 $GEMMA gemma
      logits_point k32 $KAT kat
      logits_point k16 $KAT kat
    "
    for t in gemma kat; do
      if cmp -s /tmp/iqk32-logits/$t-k16.bin /tmp/iqk32-logits/$t-k32.bin; then
        echo "$t: prime logits BYTE-IDENTICAL k16 vs k32 ($(stat -c%s /tmp/iqk32-logits/$t-k32.bin) bytes)"
      else
        echo "$t: logits DIFFER — $(cmp -l /tmp/iqk32-logits/$t-k16.bin /tmp/iqk32-logits/$t-k32.bin | wc -l) differing bytes of $(stat -c%s /tmp/iqk32-logits/$t-k32.bin)"
      fi
    done | tee -a "$RAW/logits-verdict.txt"
    ;;
  perf)
    flock /tmp/gpu5090.lock bash -c "
      set -u
      $(declare -f perf_point)
      B=$B; RAW=$RAW; R=$R; P2048=$P2048
      for p in 1 2 3 4 5; do
        if [ \$((p % 2)) -eq 1 ]; then
          perf_point k16 $GEMMA gemma \$p; perf_point k32 $GEMMA gemma \$p
        else
          perf_point k32 $GEMMA gemma \$p; perf_point k16 $GEMMA gemma \$p
        fi
      done
      for p in 1 2 3 4 5; do
        if [ \$((p % 2)) -eq 1 ]; then
          perf_point k16 $KAT kat \$p; perf_point k32 $KAT kat \$p
        else
          perf_point k32 $KAT kat \$p; perf_point k16 $KAT kat \$p
        fi
      done
    "
    ;;
  *) echo "usage: run-ab.sh logits|perf"; exit 1 ;;
esac
echo DONE
