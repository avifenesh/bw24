#!/usr/bin/env zsh
# Paired hourish screens: plain-q3k (reference) then fusedcand — same RUN_ID, same runtime build.
# Candidate serving assets are built here first (rsync to /data must be complete).
set -u
ROOT=/home/avifenesh/projects/bw24-hy3lane
HERE=$ROOT/research/per-expert-quant
EV=$HERE/evidence/local-5090-plain-arm-20260725
S=/tmp/claude-1000/-home-avifenesh-projects-bw24/85f57f8d-b160-4461-b97c-cef54c245f7c/scratchpad
cd $ROOT
exec >> $EV/screens-fused.log 2>&1
echo "=== screens_fused start $(date -Is) ==="

# stage 0: wait for the overlay rsync to /data to finish
while pgrep -f 'rsync.*hy3-fused-cand-overlay' > /dev/null; do sleep 30; done
[ -f /data/ai-ml/hf-models/hy3-fused-cand-overlay/manifest.json ] || { echo "CAND OVERLAY MISSING"; exit 1; }

# stage 1: candidate serving assets (idempotent)
R=/data/ai-ml/hf-models/hy3-fused-cand-runtime
if [ ! -f $R/manifest.json ]; then
  python3 tools/relocate_hy3_expert_overlay.py \
    /data/ai-ml/hf-models/hy3-fused-cand-overlay \
    /data/ai-ml/hf-models/hy3-layer103p5-sparse-source \
    $R || exit 1
  rm $R/experts && mkdir $R/experts && mv /data/ai-ml/hf-models/hy3-fused-cand-overlay/experts/* $R/experts/ || exit 1
fi
if [ ! -f $HOME/.local/share/bw24-models/hy3-fused-cand-dual-nvme/dual-nvme-view.json ]; then
  python3 tools/build_dual_nvme_expert_view.py $R $HOME/.local/share/bw24-models/hy3-fused-cand-dual-nvme || exit 1
fi
if [ ! -f $HOME/.local/share/bw24-models/hy3-fused-cand-root-mirror/inode-alternates.tsv ]; then
  python3 tools/build_expert_mirror_map.py $HOME/.local/share/bw24-models/hy3-fused-cand-dual-nvme \
    $HOME/.local/share/bw24-models/hy3-fused-cand-root-mirror || exit 1
fi
echo "candidate serving assets ready $(date -Is)"

gpu_wait() {
  while true; do
    local t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)
    local f=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | awk -F', ' '$3+0 > 500 {print}' | head -1)
    [[ -z "$f" && "$t" -le 60 ]] && break
    sleep 60
  done
}

export RUN_ID=fusedcand-screen-20260726
run_screen() { # $1=arm $2=artifact $3=mirror $4=profile
  gpu_wait
  env ARM=$1 MODEL=$1 ARTIFACT=$2 RUN_ID=$RUN_ID \
    SERVER_BIN=$ROOT/target/release/bw24-server \
    SERVER_LOG=$EV/screen-server-$1.log \
    PANEL_LOCK=$HERE/hourish-panel.lock.json \
    BW24_SPILL_IO=direct BW24_SPILL_PREAD_DEPTH=8 BW24_SPILL_STATS=1 BW24_SERVE_SPEC=0 \
    BW24_CPU_EXPERT_LIB=$ROOT/target/release/libbw24-cpu-experts.so \
    BW24_CPU_EXPERT_THREADS=8 BW24_CPU_EXPERT_IO_THREADS=8 \
    BW24_CPU_EXPERT_CACHE_GB=20 BW24_CPU_EXPERT_RESERVE_GB=4 BW24_CPU_EXPERT_IO=direct \
    BW24_CPU_EXPERT_MIRROR_MAP=$3 \
    BW24_CPU_EXPERT_FREEZE_CACHE=1 BW24_CPU_EXPERT_FREEZE_WARMUP_TOKENS=128 \
    BW24_CPU_EXPERT_FREEZE_PROFILE_ADMIT=1 BW24_CPU_EXPERT_FREEZE_PROFILE=$4 \
    bash $HERE/run_hourish_one_arm.sh > $EV/screen-$1.log 2>&1
  echo "arm $1 rc=$?" >> $EV/screens-progress.txt
}
: > $EV/screens-progress.txt
run_screen plain-q3k $HOME/.local/share/bw24-models/hy3-plain-q3k-dual-nvme \
  $HOME/.local/share/bw24-models/hy3-plain-q3k-root-mirror/inode-alternates.tsv \
  $S/freeze-plain-q3k.txt
run_screen fusedcand $HOME/.local/share/bw24-models/hy3-fused-cand-dual-nvme \
  $HOME/.local/share/bw24-models/hy3-fused-cand-root-mirror/inode-alternates.tsv \
  $S/freeze-fusedcand.txt
echo "SCREENS DONE $(date -Is)" > $EV/screens-fused-done.marker
