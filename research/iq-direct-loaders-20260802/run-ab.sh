#!/bin/bash
# iq-direct-loaders: 5090 perf A/B. INTERLEAVED process rounds (rep loop OUTSIDE, arms
# round-robin per rep — the sk-bm128 clock-drift protocol), each pp value = the run-gen
# in-process median of 5 reps (+1 warmup).
#   q35 : board-2048 pp-only, MEMRA_MOE_F16G=2, x5 rounds. Arms: old (MEMRA_F16G_DIRECT=kq —
#         k-quant direct kept, IQ classes on the workspace path = the sk-tail 3728.7 config)
#         vs new (naked, IQ direct). The mission's mode-2 headline cell.
#   kat : pp2048 (board-2048 pp-only) + gen512 (pp512 prefill, NGEN=128) x3 rounds, arms:
#         naked (auto-kquant default — experts on int8 MMQ) vs f16g2 (MEMRA_MOE_F16G=2,
#         experts on sk + IQ direct). The "does mode-2+direct beat the MMQ arm" probe.
#   q35flip : q35 naked vs MEMRA_MOE_F16G=2 board-2048 pp-only, interleaved — the
#         auto-kquant stale-verdict re-sweep ("IQ banks keep their measured-faster MMQ
#         tiles" was priced pre-IQ-direct).
# usage: run-ab.sh <q35|kat|q35flip|all> [nreps]
set -u
PHASE=${1:-all}
W=/home/avifenesh/projects/bw24-iq-direct
R=$W/research/iq-direct-loaders-20260802
PF512=$W/research/e2e/prompts/pp512.txt
PF2048=$W/research/e2e/prompts/board-2048.txt
OUT=$R/ab.jsonl
Q35=/data/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
KAT=/data/ai-ml/hf-models/kat-coder-v25-dev-gguf/Kwaipilot_KAT-Coder-V2.5-Dev-IQ4_XS.gguf

TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
GIT_SHA=$(git -C "$W" rev-parse --short HEAD)
PROFILE=$(cat /sys/firmware/acpi/platform_profile 2>/dev/null || echo unknown)
gpu-full-power on >/dev/null 2>&1 || true

busy_procs() {
  local n=0 pid
  while IFS=, read -r pid _; do
    pid=$(echo "$pid" | tr -d ' '); [ -n "$pid" ] || continue
    tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q -- "--embedding" || n=$((n+1))
  done < <(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null)
  echo $n
}
wait_idle() {
  local n=0
  while true; do
    local busy; busy=$(busy_procs)
    [ "$busy" -eq 0 ] && break
    sleep 5; n=$((n+1)); [ $n -gt 240 ] && { echo "wait_idle timeout (busy=$busy)"; break; }
  done
}
row() { # cell arm metric value rep
  printf '{"ts":"%s","git":"%s","cell":"%s","arm":"%s","metric":"%s","value":%s,"rep":%s,"profile":"%s","temp_c":%s}\n' \
    "$TS" "$GIT_SHA" "$1" "$2" "$3" "$4" "$5" "$PROFILE" \
    "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)" >> "$OUT"
  echo "  [$1/$2 rep$5] $3 = $4"
}

ppq35() { # arm rep
  local arm=$1 rep=$2 log="$R/q35-ab-r$2-$1.log"
  local -a env_extra=()
  [ "$arm" = old ] && env_extra+=(MEMRA_F16G_DIRECT=kq)
  wait_idle
  env "${env_extra[@]+"${env_extra[@]}"}" MEMRA_MOE_F16G=2 MEMRA_PP_ONLY=1 MEMRA_PP_REPS=5 \
    MEMRA_PP_WARMUP=1 MEMRA_PROMPT_FILE="$PF2048" \
    flock /tmp/gpu5090.lock timeout 1800 "$W/target/release/run-gen" "$Q35" > "$log" 2>&1
  local med
  med=$(grep -oE "pp-only MEDIAN: [0-9]+ tok in [0-9.]+s = [0-9.]+ tok/s" "$log" | grep -oE "= [0-9.]+" | grep -oE "[0-9.]+")
  row q35-f16g2-board2048 "$arm" pp2048_toks "${med:-null}" "$rep"
}
katarm() { # arm rep  (arm: naked | f16g2)
  local arm=$1 rep=$2
  local -a env_extra=()
  [ "$arm" = f16g2 ] && env_extra+=(MEMRA_MOE_F16G=2)
  local log="$R/kat-ab-r$rep-$arm-pp2048.log"
  wait_idle
  env "${env_extra[@]+"${env_extra[@]}"}" MEMRA_PP_ONLY=1 MEMRA_PP_REPS=5 MEMRA_PP_WARMUP=1 \
    MEMRA_PROMPT_FILE="$PF2048" \
    flock /tmp/gpu5090.lock timeout 1800 "$W/target/release/run-gen" "$KAT" > "$log" 2>&1
  local med
  med=$(grep -oE "pp-only MEDIAN: [0-9]+ tok in [0-9.]+s = [0-9.]+ tok/s" "$log" | grep -oE "= [0-9.]+" | grep -oE "[0-9.]+")
  row kat-board2048 "$arm" pp2048_toks "${med:-null}" "$rep"
  log="$R/kat-ab-r$rep-$arm-gen512.log"
  wait_idle
  env "${env_extra[@]+"${env_extra[@]}"}" MEMRA_NGEN=128 MEMRA_PROMPT_FILE="$PF512" \
    flock /tmp/gpu5090.lock timeout 1800 "$W/target/release/run-gen" "$KAT" > "$log" 2>&1
  local pp tg gate thash
  pp=$(grep -oE "prefill [0-9]+ tok in [0-9.]+s = [0-9.]+ tok/s" "$log" | grep -oE "= [0-9.]+" | grep -oE "[0-9.]+" | tail -1)
  tg=$(grep -oE "generated [0-9]+ tokens in [0-9.]+s = [0-9.]+ tok/s" "$log" | grep -oE "= [0-9.]+" | grep -oE "[0-9.]+" | tail -1)
  gate=$(grep -c "MATCH" "$log")
  thash=$(grep -A1 "^generated" "$log" | grep "tokens:" | sha256sum | cut -c1-16)
  row kat-gen512 "$arm" pp512_prefill_toks "${pp:-null}" "$rep"
  row kat-gen512 "$arm" decode_toks "${tg:-null}" "$rep"
  row kat-gen512 "$arm" match_lines "${gate:-0}" "$rep"
  echo "  [kat-gen512/$arm rep$rep] tokens_sha=$thash" | tee -a "$R/token-hashes.log"
}

exec > >(tee -a "$R/ab-console.log") 2>&1
echo "=== IQ-DIRECT AB phase=$PHASE $TS git=$GIT_SHA profile=$PROFILE ==="

if [ "$PHASE" = q35 ] || [ "$PHASE" = all ]; then
  N=${2:-5}
  for rep in $(seq 1 "$N"); do
    for arm in old new; do ppq35 "$arm" "$rep"; done
  done
fi

if [ "$PHASE" = kat ] || [ "$PHASE" = all ]; then
  N=${2:-3}
  for rep in $(seq 1 "$N"); do
    for arm in naked f16g2; do katarm "$arm" "$rep"; done
  done
fi

if [ "$PHASE" = q35flip ]; then
  N=${2:-3}
  for rep in $(seq 1 "$N"); do
    for arm in naked f16g2; do
      log="$R/q35-flip-r$rep-$arm.log"
      declare -a env_extra=()
      [ "$arm" = f16g2 ] && env_extra=(MEMRA_MOE_F16G=2)
      wait_idle
      env "${env_extra[@]+"${env_extra[@]}"}" MEMRA_PP_ONLY=1 MEMRA_PP_REPS=5 MEMRA_PP_WARMUP=1 \
        MEMRA_PROMPT_FILE="$PF2048" \
        flock /tmp/gpu5090.lock timeout 1800 "$W/target/release/run-gen" "$Q35" > "$log" 2>&1
      med=$(grep -oE "pp-only MEDIAN: [0-9]+ tok in [0-9.]+s = [0-9.]+ tok/s" "$log" | grep -oE "= [0-9.]+" | grep -oE "[0-9.]+")
      row q35-flip-board2048 "$arm" pp2048_toks "${med:-null}" "$rep"
    done
  done
fi

echo "AB-DONE phase=$PHASE $(date -u +%FT%TZ)"
