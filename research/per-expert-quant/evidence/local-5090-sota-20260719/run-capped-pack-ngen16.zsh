#!/usr/bin/env zsh

set -o nounset
set -o pipefail

evidence_dir=/home/avifenesh/.codex/worktrees/e25d/bw24/research/per-expert-quant/evidence/local-5090-sota-20260719
log_path=$evidence_dir/hy3-mtp-k1-capped-pack-preserve-all-ram20-warm32-ngen16-run1.log
scope_name=bw24-k1-pack-$$
scope_unit=${scope_name}.scope

cleanup_scope() {
    systemctl --user stop $scope_unit >/dev/null 2>&1 || true
}
trap cleanup_scope EXIT
trap 'cleanup_scope; exit 130' INT
trap 'cleanup_scope; exit 143' TERM

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
    print -r -- '[runner] thermal_regime=no intentional cooldown; paired capped development run'
    print -r -- '[runner] cpu_regime=CPUQuota 200%, CPUWeight 10; at most 2 logical CPUs aggregate'
    print -r -- '[runner] memory_regime=20-GiB CPU expert cache, MemoryHigh 34G, MemoryMax 38G'
    print -r -- '[runner] arm=preserve-all whole-expert HBM compaction'
    print -r -- "[runner] preflight_free_vram_mib=$gpu_free_mib"
    uptime
    free -h
    nvidia-smi --query-gpu=timestamp,name,temperature.gpu,power.draw,utilization.gpu,memory.used --format=csv,noheader
} | tee $log_path

/usr/bin/time -v systemd-run --user --scope --quiet \
    --unit=$scope_name \
    -p CPUQuota=200% \
    -p CPUWeight=10 \
    -p MemoryHigh=34G \
    -p MemoryMax=38G \
    env \
    BW24_NGEN=16 \
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
    BW24_CPU_EXPERT_THREADS=4 \
    BW24_CPU_EXPERT_IO_THREADS=4 \
    BW24_CPU_EXPERT_CACHE_GB=20 \
    BW24_CPU_EXPERT_CACHE_POLICY=lru \
    BW24_CPU_EXPERT_IO=direct \
    BW24_CPU_EXPERT_MIRROR_MAP=/home/avifenesh/.local/share/bw24-models/hy3-layer103p5-root-mirror/inode-alternates.tsv \
    BW24_CPU_EXPERT_FREEZE_CACHE=1 \
    BW24_CPU_EXPERT_FREEZE_WARMUP_TOKENS=32 \
    BW24_CPU_EXPERT_FREEZE_WARMUP_SPEC_K=3 \
    BW24_CPU_EXPERT_FREEZE_PROFILE_ADMIT=1 \
    BW24_CPU_EXPERT_FREEZE_PACK=1 \
    GOMP_CPU_AFFINITY=0-3 \
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
trap - EXIT INT TERM
exit $exit_code
