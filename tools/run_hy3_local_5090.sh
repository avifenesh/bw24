#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 || $# > 3 )); then
  echo "usage: $0 MODEL_DIR CPU_EXPERT_SO [INODE_ALTERNATES_TSV]" >&2
  exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$script_dir/.." && pwd)
model_dir=$(realpath -- "$1")
cpu_expert_lib=$(realpath -- "$2")
mirror_map=${3:-}
run_spec=$repo_dir/target/release/run-spec
# Compute threads stay on the P-cores; the process mask also covers E-cores so the
# companion's async read pool (BW24_CPU_EXPERT_IO_CPUSET) can run there. OMP workers
# must spin through the pipeline's read-wait gaps: with the default passive sleep,
# every ready-batch region entry pays a futex+C-state wake and decode compute
# inflated 3.0 s -> 8.4 s per 32 tokens (2026-07-22 bisect, local-5090-next3).
proc_list=${BW24_CPU_PROC_AFFINITY:-0-15}
cpu_list=${BW24_CPU_AFFINITY:-0-7}
io_cpuset=${BW24_CPU_EXPERT_IO_CPUSET:-8-15}
spec_env=()
if [[ ${BW24_SPEC_K:-1} != all ]]; then
  spec_env=(BW24_SPEC_K="${BW24_SPEC_K:-1}")
fi

if [[ ! -x $run_spec ]]; then
  echo "missing $run_spec; run cargo build --release first" >&2
  exit 3
fi
if [[ ! -d $model_dir || ! -r $cpu_expert_lib ]]; then
  echo "model directory or CPU expert companion is not readable" >&2
  exit 3
fi

mirror_env=()
if [[ -n $mirror_map ]]; then
  mirror_map=$(realpath -- "$mirror_map")
  if [[ ! -r $mirror_map ]]; then
    echo "mirror map is not readable: $mirror_map" >&2
    exit 3
  fi
  mirror_env=(BW24_CPU_EXPERT_MIRROR_MAP="$mirror_map")
fi

exec taskset --cpu-list "$proc_list" env \
  GOMP_CPU_AFFINITY="$cpu_list" \
  OMP_WAIT_POLICY="${OMP_WAIT_POLICY:-ACTIVE}" \
  BW24_CPU_EXPERT_PIPELINE="${BW24_CPU_EXPERT_PIPELINE:-1}" \
  BW24_CPU_EXPERT_IO_CPUSET="$io_cpuset" \
  "${spec_env[@]}" \
  BW24_SPEC_HOST_EMBD="${BW24_SPEC_HOST_EMBD:-1}" \
  BW24_CHAT="${BW24_CHAT:-1}" \
  BW24_MOE_CACHE="${BW24_MOE_CACHE:-1}" \
  BW24_MOE_SIZE_AWARE="${BW24_MOE_SIZE_AWARE:-1}" \
  BW24_MOE_LFU="${BW24_MOE_LFU:-1}" \
  BW24_MOE_LFU_DECAY="${BW24_MOE_LFU_DECAY:-1.0}" \
  BW24_MOE_VRAM_FRAC="${BW24_MOE_VRAM_FRAC:-0.90}" \
  BW24_MOE_HARD_VRAM_FRAC="${BW24_MOE_HARD_VRAM_FRAC:-0.90}" \
  BW24_SPILL_IO="${BW24_SPILL_IO:-direct}" \
  BW24_SPILL_PREAD_DEPTH="${BW24_SPILL_PREAD_DEPTH:-32}" \
  BW24_SPILL_WORKER_EXPERT_WINDOW="${BW24_SPILL_WORKER_EXPERT_WINDOW:-8}" \
  BW24_CPU_EXPERT_LIB="$cpu_expert_lib" \
  BW24_CPU_EXPERT_THREADS="${BW24_CPU_EXPERT_THREADS:-8}" \
  BW24_CPU_EXPERT_IO_THREADS="${BW24_CPU_EXPERT_IO_THREADS:-8}" \
  BW24_CPU_EXPERT_CACHE_GB="${BW24_CPU_EXPERT_CACHE_GB:-20}" \
  BW24_CPU_EXPERT_RESERVE_GB="${BW24_CPU_EXPERT_RESERVE_GB:-4}" \
  BW24_CPU_EXPERT_IO="${BW24_CPU_EXPERT_IO:-direct}" \
  BW24_CPU_EXPERT_FREEZE_CACHE="${BW24_CPU_EXPERT_FREEZE_CACHE:-1}" \
  BW24_CPU_EXPERT_FREEZE_WARMUP_TOKENS="${BW24_CPU_EXPERT_FREEZE_WARMUP_TOKENS:-128}" \
  BW24_CPU_EXPERT_FREEZE_WARMUP_SPEC_K="${BW24_CPU_EXPERT_FREEZE_WARMUP_SPEC_K:-3}" \
  BW24_CPU_EXPERT_FREEZE_PROFILE_ADMIT="${BW24_CPU_EXPERT_FREEZE_PROFILE_ADMIT:-1}" \
  "${mirror_env[@]}" \
  "$run_spec" "$model_dir"
