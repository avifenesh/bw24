#!/usr/bin/env zsh

set -o nounset
set -o pipefail

evidence_dir=/home/avifenesh/.codex/worktrees/e25d/bw24/research/per-expert-quant/evidence/local-5090-sota-20260719
log_path=$evidence_dir/hy3-mtp-k1-verify-prequeue-ram28-clean-ngen64-run1.log

typeset -a active_bw24
active_bw24=(${(f)"$(pgrep -a -x run-gen; pgrep -a -x run-spec)"})
if (( ${#active_bw24} != 0 )); then
    print -u2 -r -- '[runner] refusing to start while another bw24 inference process is active:'
    print -u2 -l -- $active_bw24
    exit 3
fi
gpu_free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [[ $gpu_free_mib != <-> ]] || (( gpu_free_mib < 23000 )); then
    print -u2 -r -- "[runner] refusing to start with only ${gpu_free_mib:-unknown} MiB free VRAM"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader >&2
    exit 4
fi

{
    print -r -- '[runner] thermal_regime=clean-swap, default system power policy, no intentional cooldown'
    print -r -- '[runner] memory_regime=28-GiB CPU expert cache to retain live-stack headroom'
    print -r -- "[runner] preflight_free_vram_mib=$gpu_free_mib"
    uptime
    free -h
    nvidia-smi --query-gpu=timestamp,name,temperature.gpu,power.draw,utilization.gpu,memory.used --format=csv,noheader
} | tee $log_path

/usr/bin/time -v env \
    BW24_NGEN=64 \
    BW24_SPEC_K=1 \
    BW24_SPEC_HOST_EMBD=1 \
    BW24_SPEC_STATS=1 \
    BW24_SPEC_PHASE=1 \
    BW24_SPEC_PMIN=0.2 \
    BW24_SPEC_PMIN0=1 \
    BW24_CHAT=1 \
    BW24_PROMPT='Write a Python function that implements a doubly-linked list with insert, delete, and reverse operations. Include full type hints and docstrings.' \
    BW24_MOE_CACHE=1 \
    BW24_MOE_SIZE_AWARE=1 \
    BW24_MOE_LFU=1 \
    BW24_MOE_LFU_DECAY=1.0 \
    BW24_MOE_VRAM_FRAC=0.97 \
    BW24_MOE_HARD_VRAM_FRAC=0.97 \
    BW24_SPILL_IO=direct \
    BW24_SPILL_PREAD_DEPTH=32 \
    BW24_SPILL_WORKER_EXPERT_WINDOW=8 \
    BW24_CPU_EXPERT_LIB=/tmp/libbw24-cpu-experts-final.so \
    BW24_CPU_EXPERT_THREADS=8 \
    BW24_CPU_EXPERT_IO_THREADS=8 \
    BW24_CPU_EXPERT_CACHE_GB=28 \
    BW24_CPU_EXPERT_CACHE_POLICY=lru \
    BW24_CPU_EXPERT_IO=direct \
    BW24_CPU_EXPERT_MIRROR_MAP=/home/avifenesh/.local/share/bw24-models/hy3-layer103p5-root-mirror/inode-alternates.tsv \
    BW24_CPU_EXPERT_VERIFY_PREQUEUE=1 \
    BW24_CPU_EXPERT_FREEZE_CACHE=1 \
    BW24_CPU_EXPERT_FREEZE_WARMUP_TOKENS=128 \
    BW24_CPU_EXPERT_FREEZE_WARMUP_SPEC_K=3 \
    BW24_CPU_EXPERT_FREEZE_PROFILE_ADMIT=1 \
    GOMP_CPU_AFFINITY=0-7 \
    /home/avifenesh/.codex/worktrees/e25d/bw24/target/release/run-spec \
    /home/avifenesh/.local/share/bw24-models/hy3-layer103p5-dual-nvme \
    2>&1 | tee -a $log_path
exit_code=$?
{
    print -r -- "[runner] exit_status=$exit_code"
    uptime
    free -h
    nvidia-smi --query-gpu=timestamp,name,temperature.gpu,power.draw,utilization.gpu,memory.used --format=csv,noheader
} | tee -a $log_path
exit $exit_code
