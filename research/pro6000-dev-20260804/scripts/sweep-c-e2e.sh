#!/usr/bin/env bash
# pro6000-dev: E2E decode sweep — NVFP4 m=1 mmvq family arms on the daily artifact.
# tg128 @ d512 + d6257, N=3 interleaved per arm, 1Hz clocks logged.
# Arms: auto (default mr2+dual), mr1, nodual, mr1+nodual, rp0 (GGUF layout).
set -u
export LD_LIBRARY_PATH=/usr/local/cuda-13.1/compat
cd /root/bw24
R=/root/receipts-dev/e2e
mkdir -p "$R"
M=/root/models/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
P512=research/e2e/prompts/pp512.txt
PLONG=research/e2e/prompts/p3-agentic-long.txt
gpustate() { nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw --format=csv,noheader; }
log() { echo "[$(date -u +%H:%M:%SZ)] $*" >> "$R/driver.log"; }
nvidia-smi --query-gpu=timestamp,power.draw,clocks.sm,temperature.gpu --format=csv -l 1 > "$R/gpu-1hz.csv" 2>&1 &
SMPID=$!
trap 'kill $SMPID 2>/dev/null' EXIT

run_arm() { # $1=arm name, $2=prompt, $3=depth tag, $4=rep
  local arm=$1 p=$2 d=$3 r=$4
  local env=""
  case $arm in
    auto)       env="" ;;
    mr1)        env="MEMRA_NV_MR=1" ;;
    nodual)     env="MEMRA_NV_DUAL=0" ;;
    mr1nodual)  env="MEMRA_NV_MR=1 MEMRA_NV_DUAL=0" ;;
    rp0)        env="MEMRA_RP=0" ;;
  esac
  log "$d $arm r$r pre: $(gpustate)"
  eval "$env MEMRA_NGEN=128 MEMRA_PROMPT_FILE=$p timeout 900 target/release/run-gen $M" > "$R/$d-$arm-r$r.log" 2>&1
  log "$d $arm r$r post rc=$?: $(gpustate) | $(grep -oE 'generated 128 tokens in [0-9.]+s = [0-9.]+ tok/s' "$R/$d-$arm-r$r.log" | head -1) | $(grep -oE 'argmax=[0-9]+ +decode argmax=[0-9]+ +logit maxdiff=[0-9.e-]+ +(MATCH|MISMATCH)' "$R/$d-$arm-r$r.log" | head -1)"
}

for r in 1 2 3; do
  for arm in auto mr1 nodual mr1nodual rp0; do
    run_arm $arm $P512 d512 $r
  done
done
for r in 1 2 3; do
  for arm in auto mr1 nodual mr1nodual rp0; do
    run_arm $arm $PLONG dlong $r
  done
done
log "SWEEP_C_DONE"
echo SWEEP_C_DONE
