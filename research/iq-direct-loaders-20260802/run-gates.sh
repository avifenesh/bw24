#!/bin/bash
# iq-direct-loaders: gate battery (5090). Arms: old = MEMRA_F16G_DIRECT=kq (k-quant direct
# loaders kept, IQ4_XS/IQ3_S back on the dequant-workspace path — the pre-lane shipped
# config), new = naked (IQ direct loaders, lane default).
#   kc        : kernel-check (f16g-kq-direct now gates iq4_xs/iq3_s synth + real q35 weights
#               bitwise, on top of the q4_K/q6_K + f16g-sk arms)
#   q35-ab    : MEMRA_MOE_F16G=2 gen512 argmax + token sha, BOTH arms (sha must be identical —
#               the loaders are byte-identical by construction; mode-2 anchor e94b6553fde7b9a0
#               from research/sk-tail-form-20260802)
#   q35-guard : naked gen512 + pp2048 x3 (sha anchor 86dc5f7105a3716b — naked q35 admits f16g
#               only on its 5 k-quant straggler layers, whose IQ3_S gate/up projections now
#               ride the IQ direct loaders)
#   o35b      : naked gen512 x2 (anchor c0c12c3b350dc7f5 — the Q4_K direct path went through
#               the launcher template refactor; its stream must not move a bit)
#   kat-ab    : naked gen512 (anchor e5d59ecedc57aa7d, the kquant-lane mmq sha — naked KAT is
#               dispatch-unchanged by construction) + MEMRA_MOE_F16G=2 gen512 BOTH arms (sha
#               identical across arms = end-to-end bit-identity on a pure-IQ4_XS bank)
#   spec      : run-spec K=1..8 self-consistency q35 (MEMRA_MOE_F16G=2, owntrim draft, p2,
#               NGEN=64 — covers the K=1..4 mission gate)
# usage: run-gates.sh <kc|q35-ab|q35-guard|o35b|kat-ab|spec|all>
set -u
PHASE=${1:-all}
W=/home/avifenesh/projects/bw24-iq-direct
R=$W/research/iq-direct-loaders-20260802
PF512=$W/research/e2e/prompts/pp512.txt
PF2048=$W/research/e2e/prompts/board-2048.txt
PFP2=$W/research/e2e/prompts/p2-code-medium.txt
OUT=$R/gates.jsonl
Q35=/data/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
Q35_DRAFT=/data/ai-ml/hf-models/qwen36-35b-moe/draft-35b-owntrim-nvfp4head-q4blk.gguf
O35B=/data/ai-ml/hf-models/ornith-1.0-35b-gguf/ornith-1.0-35b-Q4_K_M.gguf
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
arm_env() { # arm -> env assignments on stdout (one per line)
  case "$1" in
    old) echo MEMRA_F16G_DIRECT=kq ;;   # IQ classes back on the workspace path
    new) ;;                              # naked = IQ direct loaders (lane default)
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
echo "=== IQ-DIRECT GATES phase=$PHASE $TS git=$GIT_SHA profile=$PROFILE ==="

if [ "$PHASE" = kc ] || [ "$PHASE" = all ]; then
  wait_idle
  flock /tmp/gpu5090.lock timeout 3600 "$W/target/release/kernel-check" "$Q35" \
    > "$R/kernel-check-r1.log" 2>&1
  rc=$?
  fails=$(grep -c FAIL "$R/kernel-check-r1.log")
  echo "kernel-check rc=$rc FAIL-lines=$fails (log $R/kernel-check-r1.log)"
  grep -E "f16g-kq-direct" "$R/kernel-check-r1.log" | sed 's/^/  /'
fi

if [ "$PHASE" = q35-ab ] || [ "$PHASE" = all ]; then
  for arm in old new; do
    gen512 q35-f16g2 "$arm" "$Q35" 1 MEMRA_MOE_F16G=2
  done
  echo "q35-f16g2: shas above MUST be identical across arms (anchor e94b6553fde7b9a0)"
fi

if [ "$PHASE" = q35-guard ] || [ "$PHASE" = all ]; then
  for rep in ${GUARD_REPS:-1 2 3}; do
    pp2048 q35-guard new "$Q35" "$rep"
    gen512 q35-guard new "$Q35" "$rep"
  done
  echo "q35 guard sha anchor: 86dc5f7105a3716b (must match every rep above)"
fi

if [ "$PHASE" = o35b ] || [ "$PHASE" = all ]; then
  for rep in 1 2; do
    gen512 o35b new "$O35B" "$rep"
  done
  echo "o35b sha anchor: c0c12c3b350dc7f5 (Q4_K path through the launcher refactor)"
fi

if [ "$PHASE" = kat-ab ] || [ "$PHASE" = all ]; then
  gen512 kat-naked new "$KAT" 1
  echo "kat naked sha anchor: e5d59ecedc57aa7d (dispatch-unchanged by construction)"
  for arm in old new; do
    gen512 kat-f16g2 "$arm" "$KAT" 1 MEMRA_MOE_F16G=2
  done
  echo "kat-f16g2: shas above MUST be identical across arms (byte-identity, pure-IQ4_XS bank)"
fi

if [ "$PHASE" = spec ] || [ "$PHASE" = all ]; then
  STAG=${SPEC_TAG:-}
  wait_idle
  MEMRA_MOE_F16G=2 MEMRA_MTP_DRAFT="$Q35_DRAFT" MEMRA_NGEN=64 MEMRA_PROMPT="$(cat "$PFP2")" \
    flock /tmp/gpu5090.lock timeout 3600 "$W/target/release/run-spec" "$Q35" \
    > "$R/q35-spec-k1-8$STAG.log" 2>&1
  echo "q35 run-spec rc=$? PASS-lines=$(grep -c "self-consistency: PASS" "$R/q35-spec-k1-8$STAG.log") FAIL-lines=$(grep -ci fail "$R/q35-spec-k1-8$STAG.log")"
  grep -E "generate_spec K=|SELF-CONSISTENCY" "$R/q35-spec-k1-8$STAG.log" | tail -12
fi

echo "GATES-DONE phase=$PHASE $(date -u +%FT%TZ)"
