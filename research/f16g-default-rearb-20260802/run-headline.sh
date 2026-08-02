#!/bin/bash
# f16g-default-rearb: 5090 headline + per-model arm A/B for the sm_120a naked default flip
# moe_f16g_mode 3 (AUTO-KQUANT) -> 2 (all f16g-admitted layers on sk visitor + direct tiles).
#
# ARMS ARE WHOLE BINARIES (the real user-facing delta, naked both sides):
#   mode3 = bin-preflip/run-gen  (git e42cc8e1, restructure/public-split merge head — old default)
#   mode2 = target/release/run-gen (this lane's flip commit — candidate default)
# INTERLEAVED process rounds (rep loop OUTSIDE, arms round-robin per rep — the sk-bm128
# clock-drift protocol); each pp value = the run-gen in-process median of 5 reps (+1 warmup).
#
#   q35  : board-2048 pp-only + board-2048 e2e (NGEN=128: prefill + decode + e2e wall), x5.
#          The mission headline (stale mode-3 ruling vs mode-2+direct).
#   o35b : board-2048 pp-only + gen512 (pp512, NGEN=128), x3. Q4_K_M bank — every expert
#          layer is MMA-rejected, so mode 3 already admits f16g everywhere: expect FLAT +
#          token sha IDENTICAL across arms (dispatch unchanged by construction).
#   kat  : board-2048 pp-only + gen512, x3. Pure-IQ4_XS bank — the other flipped model
#          (mode 3 = all-MMQ, mode 2 = all-sk+direct). Decode must stay flat (t>=16 floor).
#   g26  : board-2048 pp-only + gen512, x3. gemma-MoE (gelu) — the dispatch site is
#          env-explicit (moe_f16g_gemma_on), naked stays CLOSED under both defaults:
#          expect FLAT + sha IDENTICAL across arms.
# usage: run-headline.sh <q35|o35b|kat|g26|all> [nreps]
set -u
PHASE=${1:-all}
W=/home/avifenesh/projects/wt-f16g-rearb
R=$W/research/f16g-default-rearb-20260802
PF512=$W/research/e2e/prompts/pp512.txt
PF2048=$W/research/e2e/prompts/board-2048.txt
OUT=$R/headline.jsonl
Q35=/data/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
O35B=/data/ai-ml/hf-models/ornith-1.0-35b-gguf/ornith-1.0-35b-Q4_K_M.gguf
KAT=/data/ai-ml/hf-models/kat-coder-v25-dev-gguf/Kwaipilot_KAT-Coder-V2.5-Dev-IQ4_XS.gguf
G26=/data/ai-ml/hf-models/gemma4-26b-a4b-qat-gguf/gemma-4-26B_q4_0-it.gguf

TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
GIT_SHA=$(git -C "$W" rev-parse --short HEAD)
OLD_SHA=$(cat "$R/bin-preflip/GIT_SHA")
PROFILE=$(cat /sys/firmware/acpi/platform_profile 2>/dev/null || echo unknown)
gpu-full-power on >/dev/null 2>&1 || true

binfor() { case "$1" in mode3) echo "$R/bin-preflip/run-gen" ;; mode2) echo "$W/target/release/run-gen" ;; esac; }

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
  printf '{"ts":"%s","git":"%s","git_mode3_bin":"%s","cell":"%s","arm":"%s","metric":"%s","value":%s,"rep":%s,"profile":"%s","temp_c":%s}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$GIT_SHA" "$OLD_SHA" "$1" "$2" "$3" "$4" "$5" "$PROFILE" \
    "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)" >> "$OUT"
  echo "  [$1/$2 rep$5] $3 = $4"
}

pp2048() { # tag arm model rep
  local tag=$1 arm=$2 model=$3 rep=$4
  local log="$R/$tag-pp2048-r$rep-$arm.log" bin; bin=$(binfor "$arm")
  wait_idle
  env MEMRA_PP_ONLY=1 MEMRA_PP_REPS=5 MEMRA_PP_WARMUP=1 MEMRA_PROMPT_FILE="$PF2048" \
    flock /tmp/gpu5090.lock timeout 1800 "$bin" "$model" > "$log" 2>&1
  local med
  med=$(grep -oE "pp-only MEDIAN: [0-9]+ tok in [0-9.]+s = [0-9.]+ tok/s" "$log" | grep -oE "= [0-9.]+" | grep -oE "[0-9.]+")
  row "$tag-board2048" "$arm" pp2048_toks "${med:-null}" "$rep"
}
gen_e2e() { # tag arm model rep promptfile
  local tag=$1 arm=$2 model=$3 rep=$4 pf=$5
  local log="$R/$tag-gen-r$rep-$arm.log" bin; bin=$(binfor "$arm")
  wait_idle
  env MEMRA_NGEN=128 MEMRA_PROMPT_FILE="$pf" \
    flock /tmp/gpu5090.lock timeout 1800 "$bin" "$model" > "$log" 2>&1
  local pp pps tg tgs gate thash e2e
  pp=$(grep -oE "prefill [0-9]+ tok in [0-9.]+s = [0-9.]+ tok/s" "$log" | grep -oE "= [0-9.]+" | grep -oE "[0-9.]+" | tail -1)
  pps=$(grep -oE "prefill [0-9]+ tok in [0-9.]+s" "$log" | grep -oE "in [0-9.]+s" | grep -oE "[0-9.]+" | tail -1)
  tg=$(grep -oE "generated [0-9]+ tokens in [0-9.]+s = [0-9.]+ tok/s" "$log" | grep -oE "= [0-9.]+" | grep -oE "[0-9.]+" | tail -1)
  tgs=$(grep -oE "generated [0-9]+ tokens in [0-9.]+s" "$log" | grep -oE "in [0-9.]+s" | grep -oE "[0-9.]+" | tail -1)
  gate=$(grep -c "MATCH" "$log")
  thash=$(grep -A1 "^generated" "$log" | grep "tokens:" | sha256sum | cut -c1-16)
  e2e=$(python3 -c "print(f'{${pps:-0}+${tgs:-0}:.4f}')" 2>/dev/null || echo null)
  row "$tag-gen" "$arm" prefill_toks "${pp:-null}" "$rep"
  row "$tag-gen" "$arm" decode_toks "${tg:-null}" "$rep"
  row "$tag-gen" "$arm" e2e_s "${e2e:-null}" "$rep"
  row "$tag-gen" "$arm" match_lines "${gate:-0}" "$rep"
  echo "  [$tag-gen/$arm rep$rep] tokens_sha=$thash" | tee -a "$R/token-hashes.log"
}

exec > >(tee -a "$R/headline-console.log") 2>&1
echo "=== F16G-REARB HEADLINE phase=$PHASE $TS git=$GIT_SHA (mode3 bin=$OLD_SHA) profile=$PROFILE ==="

if [ "$PHASE" = q35 ] || [ "$PHASE" = all ]; then
  N=${2:-5}
  for rep in $(seq 1 "$N"); do
    for arm in mode3 mode2; do
      pp2048 q35 "$arm" "$Q35" "$rep"
      gen_e2e q35 "$arm" "$Q35" "$rep" "$PF2048"
    done
  done
fi

if [ "$PHASE" = o35b ] || [ "$PHASE" = all ]; then
  N=${2:-3}
  for rep in $(seq 1 "$N"); do
    for arm in mode3 mode2; do
      pp2048 o35b "$arm" "$O35B" "$rep"
      gen_e2e o35b "$arm" "$O35B" "$rep" "$PF512"
    done
  done
fi

if [ "$PHASE" = kat ] || [ "$PHASE" = all ]; then
  N=${2:-3}
  for rep in $(seq 1 "$N"); do
    for arm in mode3 mode2; do
      pp2048 kat "$arm" "$KAT" "$rep"
      gen_e2e kat "$arm" "$KAT" "$rep" "$PF512"
    done
  done
fi

if [ "$PHASE" = g26 ] || [ "$PHASE" = all ]; then
  N=${2:-3}
  for rep in $(seq 1 "$N"); do
    for arm in mode3 mode2; do
      pp2048 g26 "$arm" "$G26" "$rep"
      gen_e2e g26 "$arm" "$G26" "$rep" "$PF512"
    done
  done
fi

echo "HEADLINE-DONE phase=$PHASE $(date -u +%FT%TZ)"
