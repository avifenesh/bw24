#!/usr/bin/env zsh
# Same-day m=1 decode throughput triads: plain-q3k vs fusedcand vs served layer103.5.
# Methodology identical to rebase_gate.zsh (NGEN=32, fixed prompt, freeze profile, guards).
set -u
ROOT=/home/avifenesh/projects/bw24-hy3lane
EV=$ROOT/research/per-expert-quant/evidence/local-5090-plain-arm-20260725
S=/tmp/claude-1000/-home-avifenesh-projects-bw24/85f57f8d-b160-4461-b97c-cef54c245f7c/scratchpad
cd $ROOT
exec >> $EV/throughput-pair.log 2>&1
echo "=== throughput_pair start $(date -Is) ==="

gpu_wait() {
  while true; do
    local t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)
    local f=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | awk -F', ' '$3+0 > 500 {print}' | head -1)
    local load=$(awk '{print int($1)}' /proc/loadavg)
    [[ -z "$f" && "$t" -le 56 && "$load" -le 6 ]] && break
    sleep 30
  done
}

COMMON_ENV=(
  BW24_CPU_EXPERT_PIPELINE=0
  BW24_MOE_CACHE=1 BW24_MOE_SIZE_AWARE=1 BW24_MOE_LFU=1 BW24_MOE_LFU_DECAY=1.0
  BW24_MOE_VRAM_FRAC=0.90 BW24_MOE_HARD_VRAM_FRAC=0.90
  BW24_SPILL_IO=direct BW24_SPILL_PREAD_DEPTH=32 BW24_SPILL_WORKER_EXPERT_WINDOW=8
  BW24_CPU_EXPERT_LIB=$ROOT/target/release/libbw24-cpu-experts.so
  BW24_CPU_EXPERT_IO_THREADS=8
  BW24_CPU_EXPERT_CACHE_GB=20 BW24_CPU_EXPERT_RESERVE_GB=4 BW24_CPU_EXPERT_IO=direct
)
PROMPT="Explain how CPU and GPU parallelism can hide NVMe spill latency in a large mixture-of-experts model, including cache and prefetch tradeoffs."

ensure_profile() { # $1=model $2=map $3=profile $4=label
  [ -s $3 ] && return 0
  gpu_wait
  echo "warming freeze profile for $4 $(date -Is)"
  taskset --cpu-list 0-7 env GOMP_CPU_AFFINITY=0-7 "${COMMON_ENV[@]}" \
    BW24_NGEN=1 BW24_CHAT=1 BW24_PROMPT="$PROMPT" \
    BW24_CPU_EXPERT_THREADS=8 BW24_CPU_EXPERT_MIRROR_MAP=$2 \
    BW24_CPU_EXPERT_FREEZE_CACHE=1 BW24_CPU_EXPERT_FREEZE_WARMUP_TOKENS=128 \
    BW24_CPU_EXPERT_FREEZE_PROFILE_ADMIT=1 BW24_CPU_EXPERT_FREEZE_PROFILE=$3 \
    $ROOT/target/release/run-gen $1 > $EV/tp-warm-$4.log 2>&1
  [ -s $3 ] || { echo "PROFILE SAVE FAILED for $4"; return 1; }
}

triad() { # $1=label $2=model $3=map $4=profile
  ensure_profile $2 $3 $4 $1 || return 1
  for r in a b c; do
    gpu_wait
    /usr/bin/time -v taskset --cpu-list 0-7 env GOMP_CPU_AFFINITY=0-7 "${COMMON_ENV[@]}" \
      BW24_LOCKSTEP_M=1 BW24_NGEN=32 BW24_CHAT=1 BW24_PROMPT="$PROMPT" \
      BW24_CPU_EXPERT_THREADS=8 BW24_CPU_EXPERT_MIRROR_MAP=$3 \
      BW24_CPU_EXPERT_FREEZE_CACHE=1 BW24_CPU_EXPERT_FREEZE_WARMUP_TOKENS=128 \
      BW24_CPU_EXPERT_FREEZE_PROFILE_ADMIT=1 BW24_CPU_EXPERT_FREEZE_PROFILE=$4 \
      $ROOT/target/release/run_lockstep $2 > $EV/tp-$1-$r.log 2>&1
    grep -h "tok/s" $EV/tp-$1-$r.log | tail -1
  done
}

triad plain-q3k $HOME/.local/share/bw24-models/hy3-plain-q3k-dual-nvme \
  $HOME/.local/share/bw24-models/hy3-plain-q3k-root-mirror/inode-alternates.tsv \
  $S/freeze-plain-q3k.txt
triad fusedcand $HOME/.local/share/bw24-models/hy3-fused-cand-dual-nvme \
  $HOME/.local/share/bw24-models/hy3-fused-cand-root-mirror/inode-alternates.tsv \
  $S/freeze-fusedcand.txt
triad layer103p5 $HOME/.local/share/bw24-models/hy3-layer103p5-dual-nvme \
  $HOME/.local/share/bw24-models/hy3-layer103p5-root-mirror/inode-alternates.tsv \
  $S/hy3-freeze-profile.txt
echo "THROUGHPUT PAIR DONE $(date -Is)" > $EV/throughput-pair-done.marker
