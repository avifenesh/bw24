#!/bin/bash
# q4k-expert-prefill: gate battery + supported-model guard for the AUTO-KQUANT default flip.
#   kc    : kernel-check (full)
#   o35b  : post-flip naked verification x2 (gen512 gates + pp2048 — default engages; the
#           interleaved perf claim is the door sweep, same session)
#   q35   : ctrl guard — PRE-binary (run-gen-preflip) vs POST-binary naked, interleaved x3
#           process pairs: board-2048 pp-only (med of 5 in-process) + gen512 argmax
#   spec  : o35b run-spec K=1..8 self-consistency with the adopted own-trim drafter (p2 prompt)
# usage: run-gates.sh <kc|o35b|q35|spec|all>
set -u
PHASE=${1:-all}
W=/home/avifenesh/projects/wt-q4k-expert-prefill
R=$W/research/q4k-expert-prefill-20260802
PF512=$W/research/e2e/prompts/pp512.txt
PF2048=$W/research/e2e/prompts/board-2048.txt
PFP2=$W/research/e2e/prompts/p2-code-medium.txt
OUT=$R/gates.jsonl
O35B=/data/ai-ml/hf-models/ornith-1.0-35b-gguf/ornith-1.0-35b-Q4_K_M.gguf
O35B_DRAFT=/data/ai-ml/hf-models/ornith-1.0-35b-gguf/draft-ornith35b-owntrim-nvfp4head-q4blk.gguf
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

pp2048() { # cell arm bin model rep
  local cell=$1 arm=$2 bin=$3 model=$4 rep=$5 log="$R/$1-$2-pp2048-rep$5.log"
  wait_idle
  MEMRA_PP_ONLY=1 MEMRA_PP_REPS=5 MEMRA_PP_WARMUP=1 MEMRA_PROMPT_FILE="$PF2048" \
    flock /tmp/gpu5090.lock timeout 1800 "$bin" "$model" > "$log" 2>&1
  local med
  med=$(grep -oE "pp-only MEDIAN: [0-9]+ tok in [0-9.]+s = [0-9.]+ tok/s" "$log" | grep -oE "= [0-9.]+" | grep -oE "[0-9.]+")
  row "$cell" "$arm" pp2048_toks "${med:-null}" "$rep"
}
gen512() { # cell arm bin model rep
  local cell=$1 arm=$2 bin=$3 model=$4 rep=$5 log="$R/$1-$2-gen512-rep$5.log"
  wait_idle
  MEMRA_NGEN=128 MEMRA_PROMPT_FILE="$PF512" \
    flock /tmp/gpu5090.lock timeout 1800 "$bin" "$model" > "$log" 2>&1
  local pp tg gate prime thash
  pp=$(grep -oE "prefill [0-9]+ tok in [0-9.]+s = [0-9.]+ tok/s" "$log" | grep -oE "= [0-9.]+" | grep -oE "[0-9.]+" | tail -1)
  tg=$(grep -oE "generated [0-9]+ tokens in [0-9.]+s = [0-9.]+ tok/s" "$log" | grep -oE "= [0-9.]+" | grep -oE "[0-9.]+" | tail -1)
  gate=$(grep -c "MATCH" "$log")
  prime=$(grep -oE "batched-prime argmax=[0-9]+ +tokenwise argmax=[0-9]+ +logit maxdiff=[0-9.e+-]+ +[A-Z-]+" "$log" | awk '{print $NF}')
  thash=$(grep -A1 "^generated" "$log" | grep "tokens:" | sha256sum | cut -c1-16)
  row "$cell" "$arm" gen512_prefill_toks "${pp:-null}" "$rep"
  row "$cell" "$arm" gen512_decode_toks "${tg:-null}" "$rep"
  row "$cell" "$arm" gen512_match_lines "${gate:-0}" "$rep"
  echo "  [$cell/$arm rep$rep] prime-gate=${prime:-?} tokens_sha=$thash" | tee -a "$R/token-hashes.log"
}

exec > >(tee -a "$R/gates-console.log") 2>&1
echo "=== GATES phase=$PHASE $TS git=$GIT_SHA profile=$PROFILE ==="

if [ "$PHASE" = kc ] || [ "$PHASE" = all ]; then
  wait_idle
  flock /tmp/gpu5090.lock timeout 3600 "$W/target/release/kernel-check" > "$R/kernel-check-post.log" 2>&1
  echo "kernel-check rc=$? FAIL-lines=$(grep -c FAIL "$R/kernel-check-post.log") tail:"
  tail -3 "$R/kernel-check-post.log"
fi

if [ "$PHASE" = o35b ] || [ "$PHASE" = all ]; then
  for rep in 1 2; do
    pp2048 o35b-post naked "$W/target/release/run-gen" "$O35B" "$rep"
    gen512 o35b-post naked "$W/target/release/run-gen" "$O35B" "$rep"
  done
fi

if [ "$PHASE" = q35 ] || [ "$PHASE" = all ]; then
  for rep in 1 2 3; do
    pp2048 q35-guard pre  "$W/target/release/run-gen-preflip" "$Q35" "$rep"
    pp2048 q35-guard post "$W/target/release/run-gen"         "$Q35" "$rep"
    gen512 q35-guard pre  "$W/target/release/run-gen-preflip" "$Q35" "$rep"
    gen512 q35-guard post "$W/target/release/run-gen"         "$Q35" "$rep"
  done
fi

if [ "$PHASE" = spec ] || [ "$PHASE" = all ]; then
  wait_idle
  MEMRA_MTP_DRAFT="$O35B_DRAFT" MEMRA_NGEN=128 MEMRA_PROMPT="$(cat "$PFP2")" \
    flock /tmp/gpu5090.lock timeout 3600 "$W/target/release/run-spec" "$O35B" \
    > "$R/o35b-post-spec-k1-8.log" 2>&1
  echo "run-spec rc=$? PASS-lines=$(grep -c "self-consistency: PASS" "$R/o35b-post-spec-k1-8.log") FAIL-lines=$(grep -ci fail "$R/o35b-post-spec-k1-8.log")"
  grep -E "generate_spec K=|SELF-CONSISTENCY" "$R/o35b-post-spec-k1-8.log" | tail -12
fi

echo "GATES-DONE phase=$PHASE $(date -u +%FT%TZ)"
