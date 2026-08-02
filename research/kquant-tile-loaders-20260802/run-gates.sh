#!/bin/bash
# kquant-tile-loaders: gate battery + supported-model guard.
#   o35b-spec : Ornith-35B run-spec K=1..8 self-consistency with the adopted own-trim drafter
#   kat-spec  : KAT-Coder run-spec K=1..8 self-consistency with its drafter (post IQ4_XS MMQ)
#   q35       : ctrl guard — naked gen512 + pp2048 x3 (dispatch must be UNCHANGED: token sha
#               anchor 86dc5f7105a3716b from research/q4k-expert-prefill-20260802; the only
#               f16g-admitted q35 layers are its 5 k-quant stragglers, and direct is
#               bit-identical to workspace)
# usage: run-gates.sh <o35b-spec|kat-spec|q35|all>
set -u
PHASE=${1:-all}
W=/home/avifenesh/projects/bw24-kquant-tile-loaders
R=$W/research/kquant-tile-loaders-20260802
PF512=$W/research/e2e/prompts/pp512.txt
PF2048=$W/research/e2e/prompts/board-2048.txt
PFP2=$W/research/e2e/prompts/p2-code-medium.txt
OUT=$R/gates.jsonl
O35B=/data/ai-ml/hf-models/ornith-1.0-35b-gguf/ornith-1.0-35b-Q4_K_M.gguf
O35B_DRAFT=/data/ai-ml/hf-models/ornith-1.0-35b-gguf/draft-ornith35b-owntrim-nvfp4head-q4blk.gguf
KAT=/data/ai-ml/hf-models/kat-coder-v25-dev-gguf/Kwaipilot_KAT-Coder-V2.5-Dev-IQ4_XS.gguf
KAT_DRAFT=/data/ai-ml/hf-models/kat-coder-v25-dev-gguf/draft-katcoder-owntrim-nvfp4head-q4blk.gguf
Q35=/data/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf

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

pp2048() { # cell arm model rep
  local cell=$1 arm=$2 model=$3 rep=$4 log="$R/$1-$2-pp2048-rep$4.log"
  wait_idle
  MEMRA_PP_ONLY=1 MEMRA_PP_REPS=5 MEMRA_PP_WARMUP=1 MEMRA_PROMPT_FILE="$PF2048" \
    flock /tmp/gpu5090.lock timeout 1800 "$W/target/release/run-gen" "$model" > "$log" 2>&1
  local med
  med=$(grep -oE "pp-only MEDIAN: [0-9]+ tok in [0-9.]+s = [0-9.]+ tok/s" "$log" | grep -oE "= [0-9.]+" | grep -oE "[0-9.]+")
  row "$cell" "$arm" pp2048_toks "${med:-null}" "$rep"
}
gen512() { # cell arm model rep
  local cell=$1 arm=$2 model=$3 rep=$4 log="$R/$1-$2-gen512-rep$4.log"
  wait_idle
  MEMRA_NGEN=128 MEMRA_PROMPT_FILE="$PF512" \
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

exec > >(tee -a "$R/gates-console.log") 2>&1
echo "=== GATES phase=$PHASE $TS git=$GIT_SHA profile=$PROFILE ==="

if [ "$PHASE" = o35b-spec ] || [ "$PHASE" = all ]; then
  wait_idle
  MEMRA_MTP_DRAFT="$O35B_DRAFT" MEMRA_NGEN=128 MEMRA_PROMPT="$(cat "$PFP2")" \
    flock /tmp/gpu5090.lock timeout 3600 "$W/target/release/run-spec" "$O35B" \
    > "$R/o35b-spec-k1-8.log" 2>&1
  echo "o35b run-spec rc=$? PASS-lines=$(grep -c "self-consistency: PASS" "$R/o35b-spec-k1-8.log") FAIL-lines=$(grep -ci fail "$R/o35b-spec-k1-8.log")"
  grep -E "generate_spec K=|SELF-CONSISTENCY" "$R/o35b-spec-k1-8.log" | tail -12
fi

if [ "$PHASE" = kat-spec ] || [ "$PHASE" = all ]; then
  wait_idle
  MEMRA_MTP_DRAFT="$KAT_DRAFT" MEMRA_NGEN=128 MEMRA_PROMPT="$(cat "$PFP2")" \
    flock /tmp/gpu5090.lock timeout 3600 "$W/target/release/run-spec" "$KAT" \
    > "$R/kat-spec-k1-8.log" 2>&1
  echo "kat run-spec rc=$? PASS-lines=$(grep -c "self-consistency: PASS" "$R/kat-spec-k1-8.log") FAIL-lines=$(grep -ci fail "$R/kat-spec-k1-8.log")"
  grep -E "generate_spec K=|SELF-CONSISTENCY" "$R/kat-spec-k1-8.log" | tail -12
fi

if [ "$PHASE" = q35 ] || [ "$PHASE" = all ]; then
  for rep in 1 2 3; do
    pp2048 q35-guard naked "$Q35" "$rep"
    gen512 q35-guard naked "$Q35" "$rep"
  done
  echo "q35 sha anchor: 86dc5f7105a3716b (must match every rep above)"
fi

echo "GATES-DONE phase=$PHASE $(date -u +%FT%TZ)"
