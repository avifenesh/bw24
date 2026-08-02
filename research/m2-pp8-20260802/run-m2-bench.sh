#!/usr/bin/env bash
# M2 ppN scaling/overhead bench — 8xH100 darklanes-bench box.
# INTERLEAVED AT THE INVOCATION LEVEL (H100 law: every perf claim interleaved x5 on-box;
# per-config in-process reps=1, five outer rounds). Bench discipline: GPU0 stays free —
# the baseline runs behind CUDA_VISIBLE_DEVICES=1 and the N=2/4 configs use devices 1..;
# N=8 necessarily takes all eight (noted in the receipt).
# Raw JSONL rows live in the per-run logs (tee) — the logs ARE the receipt.
set -uo pipefail
cd ~/memra
OUT=~/receipts/m2-pp8
mkdir -p "$OUT"
Q9=/opt/dlami/nvme/models/Qwen3.5-9B-Q8_0.gguf
LOCK="flock /tmp/gpu-box.lock"
BIN=target/release
P=32; G=128

nvidia-smi --query-gpu=index,name,memory.used,temperature.gpu --format=csv > "$OUT/gpu-state-pre-bench.txt" 2>&1

run() { # run <logname> <env...> -- <cmd...>   (appends: one rep per outer round)
    local log="$OUT/$1"; shift
    local envs=()
    while [ "$1" != "--" ]; do envs+=("$1"); shift; done
    shift
    echo "=== $log: env[${envs[*]:-}] $*"
    $LOCK env "${envs[@]}" "$@" 2>&1 | tee -a "$log" || echo "FAIL: $log"
}

for round in 1 2 3 4 5; do
    echo "######## interleave round $round ########"
    # single-GPU baseline (GPU1 via CUDA_VISIBLE_DEVICES; unsharded load, door closed)
    run bench-q9-base.log        CUDA_VISIBLE_DEVICES=1 -- $BIN/ppn-bench "$Q9" $P $G 1
    # N=2 on devices 1,2 — sharded and bring-up placement
    run bench-q9-n2-shard.log    MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=1,2 -- $BIN/ppn-bench "$Q9" $P $G 1
    run bench-q9-n2-noshard.log  MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=1,2 MEMRA_PP_SHARD=0 -- $BIN/ppn-bench "$Q9" $P $G 1
    # N=4 on devices 1..4
    run bench-q9-n4-shard.log    MEMRA_PP_STAGES=4 MEMRA_PP_DEVICES=1,2,3,4 -- $BIN/ppn-bench "$Q9" $P $G 1
    run bench-q9-n4-noshard.log  MEMRA_PP_STAGES=4 MEMRA_PP_DEVICES=1,2,3,4 MEMRA_PP_SHARD=0 -- $BIN/ppn-bench "$Q9" $P $G 1
    # N=8 needs all eight GPUs (GPU0 unavoidable at this N)
    run bench-q9-n8-shard.log    MEMRA_PP_STAGES=8 MEMRA_PP_DEVICES=0,1,2,3,4,5,6,7 -- $BIN/ppn-bench "$Q9" $P $G 1
    run bench-q9-n8-noshard.log  MEMRA_PP_STAGES=8 MEMRA_PP_DEVICES=0,1,2,3,4,5,6,7 MEMRA_PP_SHARD=0 -- $BIN/ppn-bench "$Q9" $P $G 1
done

echo
echo "==== per-config rows (all rounds) ===="
for f in $OUT/bench-q9-*.log; do
    echo "--- $(basename $f)"
    grep '"arm"' "$f"
done
