#!/bin/bash
# sk-tail-form: gate battery (5090). Arms: old = MEMRA_F16G_TAIL=0 (round-51 2-stage 32x64x32
# tail), new = naked (deep 32x64x64 3-stage tail, lane default).
#   q35-ab    : MEMRA_MOE_F16G=2 gen512 argmax + token sha, BOTH arms (sha must be identical —
#               the tail is byte-identical by construction)
#   q35-guard : naked gen512 + pp2048 x3 (sha anchor 86dc5f7105a3716b from
#               research/q4k-expert-prefill-20260802 — naked q35 admits f16g only on its 5
#               k-quant straggler layers, which DO ride the tail forms)
#   o35b-ab   : naked gen512 argmax + sha, BOTH arms (anchor c0c12c3b350dc7f5 — the direct
#               kq tail path)
#   spec      : run-spec K=1..8 self-consistency — q35 (MEMRA_MOE_F16G=2, owntrim draft,
#               p2, NGEN=64 — the sk-bm128 protocol) + o35b (owntrim draft, p2, NGEN=128)
# usage: run-gates.sh <q35-ab|q35-guard|o35b-ab|spec|all>
set -u
PHASE=${1:-all}
W=/home/avifenesh/projects/wt-sk-tail-form
R=$W/research/sk-tail-form-20260802
PF512=$W/research/e2e/prompts/pp512.txt
PF2048=$W/research/e2e/prompts/board-2048.txt
PFP2=$W/research/e2e/prompts/p2-code-medium.txt
OUT=$R/gates.jsonl
Q35=/data/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
Q35_DRAFT=/data/ai-ml/hf-models/qwen36-35b-moe/draft-35b-owntrim-nvfp4head-q4blk.gguf
O35B=/data/ai-ml/hf-models/ornith-1.0-35b-gguf/ornith-1.0-35b-Q4_K_M.gguf
O35B_DRAFT=/data/ai-ml/hf-models/ornith-1.0-35b-gguf/draft-ornith35b-owntrim-nvfp4head-q4blk.gguf

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
arm_env() { # arm -> env assignments on stdout (one per line)
  case "$1" in
    old) echo MEMRA_F16G_TAIL=0 ;;   # round-51 2-stage tail rollback
    new) ;;                           # naked = deep tail (lane default)
  esac
}

gen512() { # cell arm model rep extra_env...
  local cell=$1 arm=$2 model=$3 rep=$4; shift 4
  local log="$R/$cell-$arm-gen512-rep$rep.log"
  local -a env_extra=(); local ln
  while IFS= read -r ln; do [ -n "$ln" ] && env_extra+=("$ln"); done < <(arm_env "$arm")
  wait_idle
  env "${env_extra[@]}" "$@" MEMRA_NGEN=128 MEMRA_PROMPT_FILE="$PF512" \
    flock /tmp/gpu5090.lock timeout 1800 "$W/target/release/run-gen" "$model" > "$log" 2>&1
  local pp tg gate thash
  pp=$(grep -oE "prefill [0-9]+ tok in [0-9.]+s = [0-9.]+ tok/s" "$log" | grep -oE "= [0-9.]+" | grep -oE "[0-9.]+" | tail -1)
  tg=$(grep -oE "generated [0-9]+ tokens in [0-9.]+s = [0-9.]+ tok/s" "$log" | grep -oE "= [0-9.]+" | grep -oE "[0-9.]+" | tail -1)
  gate=$(grep -c "MATCH" "$log")
  thash=$(grep -A1 "^generated" "$log" | grep "tokens:" | sha256sum | cut -c1-16)
  row "$cell" "$arm" gen512_prefill_toks "${pp:-null}" "$rep"
  row "$cell" "$arm" gen512_decode_toks "${tg:-null}" "$rep"
  row "$cell" "$arm" gen512_match_lines "${gate:-0}" "$rep"
  echo "  [$cell/$arm rep$rep] tokens_sha=$thash" | tee -a "$R/token-hashes.log"
}
pp2048() { # cell arm model rep extra_env...
  local cell=$1 arm=$2 model=$3 rep=$4; shift 4
  local log="$R/$cell-$arm-pp2048-rep$rep.log"
  local -a env_extra=(); local ln
  while IFS= read -r ln; do [ -n "$ln" ] && env_extra+=("$ln"); done < <(arm_env "$arm")
  wait_idle
  env "${env_extra[@]}" "$@" MEMRA_PP_ONLY=1 MEMRA_PP_REPS=5 MEMRA_PP_WARMUP=1 \
    MEMRA_PROMPT_FILE="$PF2048" \
    flock /tmp/gpu5090.lock timeout 1800 "$W/target/release/run-gen" "$model" > "$log" 2>&1
  local med
  med=$(grep -oE "pp-only MEDIAN: [0-9]+ tok in [0-9.]+s = [0-9.]+ tok/s" "$log" | grep -oE "= [0-9.]+" | grep -oE "[0-9.]+")
  row "$cell" "$arm" pp2048_toks "${med:-null}" "$rep"
}

exec > >(tee -a "$R/gates-console.log") 2>&1
echo "=== SK-TAIL GATES phase=$PHASE $TS git=$GIT_SHA profile=$PROFILE ==="

if [ "$PHASE" = q35-ab ] || [ "$PHASE" = all ]; then
  for arm in old new; do
    gen512 q35-f16g2 "$arm" "$Q35" 1 MEMRA_MOE_F16G=2
  done
  echo "q35-f16g2: shas above MUST be identical across arms (byte-identity)"
fi

if [ "$PHASE" = o35b-ab ] || [ "$PHASE" = all ]; then
  for arm in old new; do
    gen512 o35b "$arm" "$O35B" 1
  done
  echo "o35b sha anchor: c0c12c3b350dc7f5 (both arms)"
fi

if [ "$PHASE" = q35-guard ] || [ "$PHASE" = all ]; then
  # GUARD_REPS: override rep ids for re-batches (keeps earlier raw logs intact)
  for rep in ${GUARD_REPS:-1 2 3}; do
    pp2048 q35-guard new "$Q35" "$rep"
    gen512 q35-guard new "$Q35" "$rep"
  done
  echo "q35 guard sha anchor: 86dc5f7105a3716b (must match every rep above)"
fi

if [ "$PHASE" = spec ] || [ "$PHASE" = all ]; then
  # SPEC_ONLY=q35|o35b restricts (re-run seam); SPEC_TAG suffixes the log (keeps failed raws)
  STAG=${SPEC_TAG:-}
  if [ "${SPEC_ONLY:-both}" != o35b ]; then
    wait_idle
    MEMRA_MOE_F16G=2 MEMRA_MTP_DRAFT="$Q35_DRAFT" MEMRA_NGEN=64 MEMRA_PROMPT="$(cat "$PFP2")" \
      flock /tmp/gpu5090.lock timeout 3600 "$W/target/release/run-spec" "$Q35" \
      > "$R/q35-spec-k1-8$STAG.log" 2>&1
    echo "q35 run-spec rc=$? PASS-lines=$(grep -c "self-consistency: PASS" "$R/q35-spec-k1-8$STAG.log") FAIL-lines=$(grep -ci fail "$R/q35-spec-k1-8$STAG.log")"
    grep -E "generate_spec K=|SELF-CONSISTENCY" "$R/q35-spec-k1-8$STAG.log" | tail -12
  fi
  if [ "${SPEC_ONLY:-both}" != q35 ]; then
    wait_idle
    MEMRA_MTP_DRAFT="$O35B_DRAFT" MEMRA_NGEN=128 MEMRA_PROMPT="$(cat "$PFP2")" \
      flock /tmp/gpu5090.lock timeout 3600 "$W/target/release/run-spec" "$O35B" \
      > "$R/o35b-spec-k1-8$STAG.log" 2>&1
    echo "o35b run-spec rc=$? PASS-lines=$(grep -c "self-consistency: PASS" "$R/o35b-spec-k1-8$STAG.log") FAIL-lines=$(grep -ci fail "$R/o35b-spec-k1-8$STAG.log")"
    grep -E "generate_spec K=|SELF-CONSISTENCY" "$R/o35b-spec-k1-8$STAG.log" | tail -12
  fi
fi

echo "GATES-DONE phase=$PHASE $(date -u +%FT%TZ)"
