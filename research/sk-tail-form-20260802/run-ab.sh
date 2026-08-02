#!/bin/bash
# sk-tail-form: 5090 perf A/B. INTERLEAVED process rounds (rep loop OUTSIDE, arms round-robin
# per rep — the sk-bm128 clock-drift protocol), each pp value = the run-gen in-process median
# of 5 reps (+1 warmup).
#   q35  : board-2048 pp-only, MEMRA_MOE_F16G=2, x5 rounds. Arms: old (MEMRA_F16G_TAIL=0,
#          round-51 2-stage tail) vs new (naked, deep tail). The mission's headline cell.
#   o35b : pp2048 (board-2048 pp-only) + pp512 (gen512 prefill, NGEN=128) x3 rounds, naked
#          dispatch (direct kq tail) — small-m tail matters most at short prompts.
# usage: run-ab.sh <q35|o35b|all> [nreps]
set -u
PHASE=${1:-all}
W=/home/avifenesh/projects/wt-sk-tail-form
R=$W/research/sk-tail-form-20260802
PF512=$W/research/e2e/prompts/pp512.txt
PF2048=$W/research/e2e/prompts/board-2048.txt
OUT=$R/ab.jsonl
Q35=/data/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
O35B=/data/ai-ml/hf-models/ornith-1.0-35b-gguf/ornith-1.0-35b-Q4_K_M.gguf

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
arm_env() {
  case "$1" in
    old) echo MEMRA_F16G_TAIL=0 ;;
    new) ;;
  esac
}

ppq35() { # arm rep
  local arm=$1 rep=$2 log="$R/q35-ab-r$2-$1.log"
  local -a env_extra=(); local ln
  while IFS= read -r ln; do [ -n "$ln" ] && env_extra+=("$ln"); done < <(arm_env "$arm")
  wait_idle
  env "${env_extra[@]}" MEMRA_MOE_F16G=2 MEMRA_PP_ONLY=1 MEMRA_PP_REPS=5 MEMRA_PP_WARMUP=1 \
    MEMRA_PROMPT_FILE="$PF2048" \
    flock /tmp/gpu5090.lock timeout 1800 "$W/target/release/run-gen" "$Q35" > "$log" 2>&1
  local med
  med=$(grep -oE "pp-only MEDIAN: [0-9]+ tok in [0-9.]+s = [0-9.]+ tok/s" "$log" | grep -oE "= [0-9.]+" | grep -oE "[0-9.]+")
  row q35-f16g2-board2048 "$arm" pp2048_toks "${med:-null}" "$rep"
}
ppo35b() { # arm rep
  local arm=$1 rep=$2 log="$R/o35b-ab-r$2-$1-pp2048.log"
  local -a env_extra=(); local ln
  while IFS= read -r ln; do [ -n "$ln" ] && env_extra+=("$ln"); done < <(arm_env "$arm")
  wait_idle
  env "${env_extra[@]}" MEMRA_PP_ONLY=1 MEMRA_PP_REPS=5 MEMRA_PP_WARMUP=1 \
    MEMRA_PROMPT_FILE="$PF2048" \
    flock /tmp/gpu5090.lock timeout 1800 "$W/target/release/run-gen" "$O35B" > "$log" 2>&1
  local med
  med=$(grep -oE "pp-only MEDIAN: [0-9]+ tok in [0-9.]+s = [0-9.]+ tok/s" "$log" | grep -oE "= [0-9.]+" | grep -oE "[0-9.]+")
  row o35b-board2048 "$arm" pp2048_toks "${med:-null}" "$rep"
}
gen512o35b() { # arm rep
  local arm=$1 rep=$2 log="$R/o35b-ab-r$2-$1-gen512.log"
  local -a env_extra=(); local ln
  while IFS= read -r ln; do [ -n "$ln" ] && env_extra+=("$ln"); done < <(arm_env "$arm")
  wait_idle
  env "${env_extra[@]}" MEMRA_NGEN=128 MEMRA_PROMPT_FILE="$PF512" \
    flock /tmp/gpu5090.lock timeout 1800 "$W/target/release/run-gen" "$O35B" > "$log" 2>&1
  local pp tg gate thash
  pp=$(grep -oE "prefill [0-9]+ tok in [0-9.]+s = [0-9.]+ tok/s" "$log" | grep -oE "= [0-9.]+" | grep -oE "[0-9.]+" | tail -1)
  tg=$(grep -oE "generated [0-9]+ tokens in [0-9.]+s = [0-9.]+ tok/s" "$log" | grep -oE "= [0-9.]+" | grep -oE "[0-9.]+" | tail -1)
  gate=$(grep -c "MATCH" "$log")
  thash=$(grep -A1 "^generated" "$log" | grep "tokens:" | sha256sum | cut -c1-16)
  row o35b-gen512 "$arm" pp512_prefill_toks "${pp:-null}" "$rep"
  row o35b-gen512 "$arm" decode_toks "${tg:-null}" "$rep"
  row o35b-gen512 "$arm" match_lines "${gate:-0}" "$rep"
  echo "  [o35b-gen512/$arm rep$rep] tokens_sha=$thash (anchor c0c12c3b350dc7f5)" | tee -a "$R/token-hashes.log"
}

exec > >(tee -a "$R/ab-console.log") 2>&1
echo "=== SK-TAIL AB phase=$PHASE $TS git=$GIT_SHA profile=$PROFILE ==="

if [ "$PHASE" = q35 ] || [ "$PHASE" = all ]; then
  N=${2:-5}
  for rep in $(seq 1 "$N"); do
    for arm in old new; do ppq35 "$arm" "$rep"; done
  done
fi

if [ "$PHASE" = o35b ] || [ "$PHASE" = all ]; then
  N=${2:-3}
  for rep in $(seq 1 "$N"); do
    for arm in old new; do
      ppo35b "$arm" "$rep"
      gen512o35b "$arm" "$rep"
    done
  done
fi

echo "AB-DONE phase=$PHASE $(date -u +%FT%TZ)"
