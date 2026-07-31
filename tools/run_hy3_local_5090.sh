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
# Serial read-then-compute is the measured local winner: overlapped full-rate O_DIRECT
# DMA inflates the compute loops themselves ~2.7x (2026-07-22 bisect chain,
# local-5090-next3/pipeline-bisect-receipt.md), so MEMRA_CPU_EXPERT_PIPELINE stays off
# unless explicitly requested for experiments.
proc_list=${MEMRA_CPU_PROC_AFFINITY:-0-7}
cpu_list=${MEMRA_CPU_AFFINITY:-0-7}
spec_env=()
if [[ ${MEMRA_SPEC_K:-1} != all ]]; then
  spec_env=(MEMRA_SPEC_K="${MEMRA_SPEC_K:-1}")
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
  mirror_env=(MEMRA_CPU_EXPERT_MIRROR_MAP="$mirror_map")
fi

exec taskset --cpu-list "$proc_list" env \
  GOMP_CPU_AFFINITY="$cpu_list" \
  "${spec_env[@]}" \
  MEMRA_SPEC_HOST_EMBD="${MEMRA_SPEC_HOST_EMBD:-1}" \
  MEMRA_CHAT="${MEMRA_CHAT:-1}" \
  MEMRA_MOE_CACHE="${MEMRA_MOE_CACHE:-1}" \
  MEMRA_MOE_SIZE_AWARE="${MEMRA_MOE_SIZE_AWARE:-1}" \
  MEMRA_MOE_LFU="${MEMRA_MOE_LFU:-1}" \
  MEMRA_MOE_LFU_DECAY="${MEMRA_MOE_LFU_DECAY:-1.0}" \
  MEMRA_MOE_VRAM_FRAC="${MEMRA_MOE_VRAM_FRAC:-0.90}" \
  MEMRA_MOE_HARD_VRAM_FRAC="${MEMRA_MOE_HARD_VRAM_FRAC:-0.90}" \
  MEMRA_SPILL_IO="${MEMRA_SPILL_IO:-direct}" \
  MEMRA_SPILL_PREAD_DEPTH="${MEMRA_SPILL_PREAD_DEPTH:-32}" \
  MEMRA_SPILL_WORKER_EXPERT_WINDOW="${MEMRA_SPILL_WORKER_EXPERT_WINDOW:-8}" \
  MEMRA_CPU_EXPERT_LIB="$cpu_expert_lib" \
  MEMRA_CPU_EXPERT_THREADS="${MEMRA_CPU_EXPERT_THREADS:-8}" \
  MEMRA_CPU_EXPERT_IO_THREADS="${MEMRA_CPU_EXPERT_IO_THREADS:-8}" \
  MEMRA_CPU_EXPERT_CACHE_GB="${MEMRA_CPU_EXPERT_CACHE_GB:-20}" \
  MEMRA_CPU_EXPERT_RESERVE_GB="${MEMRA_CPU_EXPERT_RESERVE_GB:-4}" \
  MEMRA_CPU_EXPERT_IO="${MEMRA_CPU_EXPERT_IO:-direct}" \
  MEMRA_CPU_EXPERT_FREEZE_CACHE="${MEMRA_CPU_EXPERT_FREEZE_CACHE:-1}" \
  MEMRA_CPU_EXPERT_FREEZE_WARMUP_TOKENS="${MEMRA_CPU_EXPERT_FREEZE_WARMUP_TOKENS:-128}" \
  MEMRA_CPU_EXPERT_FREEZE_WARMUP_SPEC_K="${MEMRA_CPU_EXPERT_FREEZE_WARMUP_SPEC_K:-3}" \
  MEMRA_CPU_EXPERT_FREEZE_PROFILE_ADMIT="${MEMRA_CPU_EXPERT_FREEZE_PROFILE_ADMIT:-1}" \
  "${mirror_env[@]}" \
  "$run_spec" "$model_dir"
