#!/usr/bin/env zsh
set -u
ROOT=/home/avifenesh/projects/bw24-hy3lane
EV=$ROOT/research/per-expert-quant/evidence/local-5090-plain-arm-20260725
S=/tmp/claude-1000/-home-avifenesh-projects-bw24/85f57f8d-b160-4461-b97c-cef54c245f7c/scratchpad
MODEL=$HOME/.local/share/bw24-models/hy3-plain-q3k-dual-nvme
MAP=$HOME/.local/share/bw24-models/hy3-plain-q3k-root-mirror/inode-alternates.tsv
TR=$S/calib-weight-trace-plain
mkdir -p $TR
cd $ROOT
n=0
while IFS= read -r ids; do
  n=$((n+1))
  [ -s $TR/routes-$n.trace ] && continue   # resumable
  while true; do
    t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)
    f=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | awk -F', ' '$3+0 > 500 {print}' | head -1)
    [[ -z "$f" && "$t" -le 60 ]] && break
    sleep 30
  done
  taskset --cpu-list 0-7 env GOMP_CPU_AFFINITY=0-7 \
    BW24_NGEN=1 \
    BW24_MOE_WEIGHT_TRACE=$TR/weights-$n.trace BW24_MOE_TRACE=$TR/routes-$n.trace \
    BW24_CPU_EXPERT_PIPELINE=0 BW24_CPU_EXPERT_FREEZE_PROFILE=$S/hy3-plain-freeze-profile.txt \
    BW24_MOE_CACHE=1 BW24_MOE_SIZE_AWARE=1 BW24_MOE_LFU=1 BW24_MOE_LFU_DECAY=1.0 \
    BW24_MOE_VRAM_FRAC=0.90 BW24_MOE_HARD_VRAM_FRAC=0.90 \
    BW24_SPILL_IO=direct BW24_SPILL_PREAD_DEPTH=32 BW24_SPILL_WORKER_EXPERT_WINDOW=8 \
    BW24_CPU_EXPERT_LIB=$ROOT/target/release/libbw24-cpu-experts.so \
    BW24_CPU_EXPERT_THREADS=8 BW24_CPU_EXPERT_IO_THREADS=8 \
    BW24_CPU_EXPERT_CACHE_GB=20 BW24_CPU_EXPERT_RESERVE_GB=4 BW24_CPU_EXPERT_IO=direct \
    BW24_CPU_EXPERT_MIRROR_MAP=$MAP \
    $ROOT/target/release/run-gen $MODEL ${=ids} > $TR/run-$n.log 2>&1
done < $S/calib-prompts.txt
echo "PLAIN CALIB TRACE DONE $(date -Is) prompts=$n" > $EV/calibtrace-done.marker
