#!/usr/bin/env zsh

# Archival winners-only runner for the fixed LRU control. Rejected fused/LFU/least-stale product
# arms were removed; this script intentionally cannot label or invoke those obsolete variants.

set -o nounset
set -o pipefail

if (( $# < 1 || $# > 2 )) || [[ $1 != control ]] || [[ ${2:-1} != <-> ]]; then
    print -u2 -r -- "usage: $0 control [run-number]"
    exit 2
fi

mode=$1
run_number=${2:-1}
evidence_dir=/home/avifenesh/.codex/worktrees/e25d/bw24/research/per-expert-quant/evidence/local-5090-sota-20260719
repo_dir=/home/avifenesh/.codex/worktrees/e25d/bw24
run_spec=${BW24_RUN_SPEC:-/tmp/bw24-run-spec}
cpu_expert_lib=${BW24_RUN_CPU_EXPERT_LIB:-/tmp/libbw24-cpu-experts-headroom-final.so}
ngen=${BW24_RUN_NGEN:-64}
cache_gb=${BW24_RUN_CACHE_GB:-36}
reserve_gb=${BW24_RUN_RESERVE_GB:-6}
min_free_vram_mib=${BW24_RUN_MIN_FREE_VRAM_MIB:-23000}
max_load=${BW24_RUN_MAX_LOAD:-4.0}
model_dir=/home/avifenesh/.local/share/bw24-models/hy3-layer103p5-dual-nvme
mirror_map=/home/avifenesh/.local/share/bw24-models/hy3-layer103p5-root-mirror/inode-alternates.tsv
log_path=$evidence_dir/hy3-mtp-k1-headroom6-wholepack-${mode}-clean-ngen${ngen}-run${run_number}.log
scope_name=bw24-headroom6-${mode}-$$
scope_unit=${scope_name}.scope

if [[ $ngen != <-> ]] || (( ngen < 1 )); then
    print -u2 -r -- 'BW24_RUN_NGEN must be a positive integer'
    exit 2
fi
if [[ ! $cache_gb =~ '^[0-9]+([.][0-9]+)?$' ]] \
    || [[ ! $reserve_gb =~ '^[0-9]+([.][0-9]+)?$' ]]; then
    print -u2 -r -- 'BW24_RUN_CACHE_GB and BW24_RUN_RESERVE_GB must be non-negative numbers'
    exit 2
fi
if [[ $min_free_vram_mib != <-> ]]; then
    print -u2 -r -- 'BW24_RUN_MIN_FREE_VRAM_MIB must be a non-negative integer'
    exit 2
fi
if [[ ! $max_load =~ '^[0-9]+([.][0-9]+)?$' ]]; then
    print -u2 -r -- 'BW24_RUN_MAX_LOAD must be a non-negative number'
    exit 2
fi

cleanup_scope() {
    systemctl --user stop $scope_unit >/dev/null 2>&1 || true
}
trap cleanup_scope EXIT
trap 'cleanup_scope; exit 130' INT
trap 'cleanup_scope; exit 143' TERM

typeset -a active_bw24
active_bw24=(${(f)"$(pgrep -a -x run-gen; pgrep -a -x run-spec; pgrep -a -x bw24-run-spec; pgrep -a -x kernel-check)"})
if (( ${#active_bw24} != 0 )); then
    print -u2 -r -- '[runner] refusing to start while another bw24 inference process is active:'
    print -u2 -l -- $active_bw24
    exit 3
fi

load_one=$(cut -d ' ' -f 1 /proc/loadavg)
if (( load_one > max_load )); then
    print -u2 -r -- "[runner] refusing to start at one-minute load $load_one (> $max_load)"
    ps -eo pid,stat,ni,psr,%cpu,comm,args --sort=-%cpu | head -20 >&2
    exit 6
fi
if pgrep -x rsync >/dev/null; then
    print -u2 -r -- '[runner] refusing to start while rsync may contend for CPU or model NVMe'
    pgrep -a -x rsync >&2
    exit 7
fi

gpu_free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [[ $gpu_free_mib != <-> ]] || (( gpu_free_mib < min_free_vram_mib )); then
    print -u2 -r -- "[runner] refusing to start with only ${gpu_free_mib:-unknown} MiB free VRAM"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader >&2
    exit 4
fi

if [[ ! -x $run_spec || ! -r $cpu_expert_lib || ! -r $mirror_map ]]; then
    print -u2 -r -- '[runner] missing runtime, CPU expert library, or mirror map'
    exit 5
fi

{
    print -r -- '[runner] thermal_regime=clean-swap, default system power policy, no intentional cooldown'
    print -r -- '[runner] cpu_regime=process-affine CPUs 0-7, unlimited max-weight systemd scope, 8 P-core CPU expert team'
    print -r -- '[runner] io_regime=mirrored direct I/O across both NVMe devices; IOWeight=10000 requested'
    print -r -- "[runner] memory_regime=requested ${cache_gb}-GiB CPU cache with ${reserve_gb}-GiB MemAvailable reserve"
    print -r -- '[runner] cgroup_memory_regime=unlimited sibling systemd scope'
    print -r -- "[runner] measurement_N=1 generation_tokens=$ngen"
    print -r -- "[runner] mode=$mode"
    print -r -- '[runner] cpu_expert_backend=fixed-lru-winner'
    print -r -- "[runner] preflight_free_vram_mib=$gpu_free_mib"
    print -r -- "[runner] min_free_vram_mib=$min_free_vram_mib"
    print -r -- "[runner] preflight_load_one=$load_one"
    print -r -- "[runner] max_load=$max_load"
    print -r -- "[runner] git_head=$(git -C $repo_dir rev-parse HEAD)"
    sha256sum $run_spec $cpu_expert_lib
    uptime
    free -h
    swapon --show
    sed -n -e '/^MemAvailable:/p' -e '/^SwapFree:/p' -e '/^Dirty:/p' -e '/^Writeback:/p' /proc/meminfo
    vmstat 1 3
    nvidia-smi --query-gpu=timestamp,name,temperature.gpu,power.draw,utilization.gpu,memory.used,memory.free --format=csv,noheader
} | tee $log_path

systemd-run --user --scope --quiet --expand-environment=no \
    --unit=$scope_name \
    -p AllowedCPUs=0-7 \
    -p CPUWeight=10000 \
    -p IOWeight=10000 \
    -p MemoryHigh=infinity \
    -p MemoryMax=infinity \
    -p MemorySwapMax=0 \
    /usr/bin/zsh -c '
        print -r -- "[runner-scope] pid=$$"
        cat /proc/self/cgroup
        cgroup_path=$(awk -F: '\''$1 == "0" { print $3 }'\'' /proc/self/cgroup)
        for setting in cpuset.cpus.effective cpu.max cpu.weight io.weight memory.high memory.max memory.swap.max; do
            setting_path=/sys/fs/cgroup${cgroup_path}/$setting
            if [[ -r $setting_path ]]; then
                print -r -- "[runner-scope] $setting=$(< $setting_path)"
            else
                print -r -- "[runner-scope] $setting=unavailable"
            fi
        done
        exec "$@"
    ' bw24-scope /usr/bin/time -v taskset --cpu-list 0-7 env \
    BW24_NGEN=$ngen \
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
    BW24_CPU_EXPERT_LIB=$cpu_expert_lib \
    BW24_CPU_EXPERT_THREADS=8 \
    BW24_CPU_EXPERT_IO_THREADS=8 \
    BW24_CPU_EXPERT_CACHE_GB=$cache_gb \
    BW24_CPU_EXPERT_RESERVE_GB=$reserve_gb \
    BW24_CPU_EXPERT_IO=direct \
    BW24_CPU_EXPERT_MIRROR_MAP=$mirror_map \
    BW24_CPU_EXPERT_FREEZE_CACHE=1 \
    BW24_CPU_EXPERT_FREEZE_WARMUP_TOKENS=128 \
    BW24_CPU_EXPERT_FREEZE_WARMUP_SPEC_K=3 \
    BW24_CPU_EXPERT_FREEZE_PROFILE_ADMIT=1 \
    GOMP_CPU_AFFINITY=0-7 \
    $run_spec $model_dir \
    2>&1 | tee -a $log_path
exit_code=$?

{
    print -r -- "[runner] exit_status=$exit_code"
    uptime
    free -h
    swapon --show
    sed -n -e '/^MemAvailable:/p' -e '/^SwapFree:/p' -e '/^Dirty:/p' -e '/^Writeback:/p' /proc/meminfo
    vmstat 1 3
    nvidia-smi --query-gpu=timestamp,name,temperature.gpu,power.draw,utilization.gpu,memory.used,memory.free --format=csv,noheader
} | tee -a $log_path
trap - EXIT INT TERM
exit $exit_code
