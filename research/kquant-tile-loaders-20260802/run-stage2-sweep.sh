#!/bin/bash
# kquant-tile-loaders stage 2: KAT-Coder IQ4_XS dense-trunk MMQ (m>=16 prefill admission).
# Arms per rep (interleaved, rep loop OUTSIDE): dp4a (MEMRA_PP_IQMMQ=0 — the post-kat-anomaly
# default, trunk prefill on the per-column dp4a grid), mmq (naked — the new dense int8-MMA MMQ).
# Measurements per arm-rep:
#   gen512 = pp512.txt MEMRA_NGEN=128 run-gen — board-shape prefill+decode + argmax + token sha.
#   pp2048 = board-2048.txt MEMRA_PP_ONLY (median of 5 in-process reps, 1 warmup).
# NOTE: the MMQ arm is a NUMERIC CLASS change at m>=16 (MMA fold order) — token sha may shift
# vs the dp4a arm; the gates are argmax MATCH + spec self-consistency, not sha equality.
# The dp4a arm sha anchor is 9102ffd0b8241a65 (research/kat-anomaly-20260802, naked post-flip).
# usage: run-stage2-sweep.sh [nreps]
set -u
N=${1:-3}
W=/home/avifenesh/projects/bw24-kquant-tile-loaders
R=$W/research/kquant-tile-loaders-20260802
PF512=$W/research/e2e/prompts/pp512.txt
PF2048=$W/research/e2e/prompts/board-2048.txt
OUT=$R/stage2-sweep.jsonl
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
row() { # arm metric value rep
  printf '{"ts":"%s","git":"%s","cell":"kat-iq4xs-mmq","arm":"%s","metric":"%s","value":%s,"rep":%s,"profile":"%s","temp_c":%s}\n' \
    "$TS" "$GIT_SHA" "$1" "$2" "$3" "$4" "$PROFILE" \
    "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)" >> "$OUT"
  echo "  [$1 rep$4] $2 = $3"
}

arm_env() { # arm -> env assignments on stdout (one per line)
  case "$1" in
    dp4a) echo MEMRA_PP_IQMMQ=0 ;;   # pre-lane default (trunk prefill dp4a)
    mmq)  ;;                          # naked = new dense IQ4_XS MMQ at m>=16
  esac
}

run_gen512() { # arm rep
  local arm=$1 rep=$2 log="$R/s2-gen512-$1-rep$2.log" vram="$R/s2-gen512-$1-rep$2.vram"
  local -a env_extra=(); local ln
  while IFS= read -r ln; do [ -n "$ln" ] && env_extra+=("$ln"); done < <(arm_env "$arm")
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$vram"; sleep 1; done ) &
  local sampler=$!
  env "${env_extra[@]}" MEMRA_NGEN=128 MEMRA_PROMPT_FILE="$PF512" \
    flock /tmp/gpu5090.lock timeout 1800 "$W/target/release/run-gen" "$KAT" > "$log" 2>&1
  local rc=$?
  kill $sampler 2>/dev/null; wait $sampler 2>/dev/null
  local pp tg gate peak thash dec
  pp=$(grep -oE "prefill [0-9]+ tok in [0-9.]+s = [0-9.]+ tok/s" "$log" | grep -oE "= [0-9.]+" | grep -oE "[0-9.]+" | tail -1)
  tg=$(grep -oE "generated [0-9]+ tokens in [0-9.]+s = [0-9.]+ tok/s" "$log" | grep -oE "= [0-9.]+" | grep -oE "[0-9.]+" | tail -1)
  gate=$(grep -c "MATCH" "$log")
  peak=$(sort -n "$vram" | tail -1)
  dec=$(grep -oE "resident-experts decision: .* -> (RESIDENT|SLRU cache)" "$log" | grep -oE "(RESIDENT|SLRU cache)$" | head -1 | tr ' ' '_')
  thash=$(grep -A1 "^generated" "$log" | grep "tokens:" | sha256sum | cut -c1-16)
  if [ -z "${tg:-}" ]; then row "$arm" gen512_ERROR "$rc" "$rep"; tail -5 "$log"; return; fi
  row "$arm" gen512_prefill_toks "${pp:-0}" "$rep"
  row "$arm" gen512_decode_toks "$tg" "$rep"
  row "$arm" gen512_argmax_match "$gate" "$rep"
  row "$arm" gen512_peak_vram_mib "${peak:-0}" "$rep"
  echo "  [kat $arm rep$rep] decision=$dec tokens_sha=$thash (dp4a anchor 9102ffd0b8241a65)" | tee -a "$R/token-hashes.log"
}

run_pp2048() { # arm rep
  local arm=$1 rep=$2 log="$R/s2-pp2048-$1-rep$2.log"
  local -a env_extra=(); local ln
  while IFS= read -r ln; do [ -n "$ln" ] && env_extra+=("$ln"); done < <(arm_env "$arm")
  env "${env_extra[@]}" MEMRA_PP_ONLY=1 MEMRA_PP_REPS=5 MEMRA_PP_WARMUP=1 \
    MEMRA_PROMPT_FILE="$PF2048" \
    flock /tmp/gpu5090.lock timeout 1800 "$W/target/release/run-gen" "$KAT" > "$log" 2>&1
  local rc=$?
  local med
  med=$(grep -oE "pp-only MEDIAN: [0-9]+ tok in [0-9.]+s = [0-9.]+ tok/s" "$log" | grep -oE "= [0-9.]+" | grep -oE "[0-9.]+")
  if [ -z "${med:-}" ]; then row "$arm" pp2048_ERROR "$rc" "$rep"; tail -5 "$log"; return; fi
  row "$arm" pp2048_toks "$med" "$rep"
}

echo "=== KAT IQ4XS-MMQ STAGE2 SWEEP x$N $TS git=$GIT_SHA profile=$PROFILE ===" | tee -a "$R/sweep-console.log"
{
  for rep in $(seq 1 "$N"); do
    for arm in dp4a mmq; do
      wait_idle; run_gen512 "$arm" "$rep"
      wait_idle; run_pp2048 "$arm" "$rep"
    done
  done
  echo "STAGE2-SWEEP-DONE $(date -u +%FT%TZ)"
} 2>&1 | tee -a "$R/sweep-console.log"
