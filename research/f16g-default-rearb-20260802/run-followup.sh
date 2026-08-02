#!/bin/bash
# f16g-default-rearb follow-ups:
#   strict : decode-batch --mode strict per the ACTUAL protocol (equalized composition
#            MEMRA_MMVQ=0 MEMRA_NO_FUSE_NORMQ=1, worst draws MEMRA_GATE_SEED q9j=0 / q35=16
#            — gate1-recal-20260802 + validate-h100.sh). The battery's first strict runs
#            were invoked naked (no equalization) = out-of-protocol; their FAILs are the
#            documented accepted FP-composition gap, logs kept as *-MISFIRE.
#   g26gen : g26 gen A/B on the board-2048 prompt (the g26-decode-20260801 known-green gate
#            prompt) both arms x2 — the pp512 prompt turned out to be a PRE-EXISTING
#            never-gated near-tie MISMATCH on the merge head (bit-identical in both arms).
# usage: run-followup.sh <strict|g26gen|all>
set -u
PHASE=${1:-all}
W=/home/avifenesh/projects/wt-f16g-rearb
R=$W/research/f16g-default-rearb-20260802
BIN=$W/target/release
PF2048=$W/research/e2e/prompts/board-2048.txt
OUT=$R/gates.jsonl
Q35=/data/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
G26=/data/ai-ml/hf-models/gemma4-26b-a4b-qat-gguf/gemma-4-26B_q4_0-it.gguf
Q9J=/home/avifenesh/models/qwen3.5-9b-judge-q8_0.gguf

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
row() { # cell metric value rep
  printf '{"ts":"%s","git":"%s","cell":"%s","metric":"%s","value":%s,"rep":%s,"profile":"%s","temp_c":%s}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$GIT_SHA" "$1" "$2" "$3" "$4" "$PROFILE" \
    "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)" >> "$OUT"
  echo "  [$1 rep$4] $2 = $3"
}

exec > >(tee -a "$R/followup-console.log") 2>&1
echo "=== F16G-REARB FOLLOWUP phase=$PHASE $(date -u +%FT%TZ) git=$GIT_SHA profile=$PROFILE ==="

if [ "$PHASE" = strict ] || [ "$PHASE" = all ]; then
  wait_idle
  env MEMRA_GATE_SEED=16 MEMRA_MMVQ=0 MEMRA_NO_FUSE_NORMQ=1 \
    flock /tmp/gpu5090.lock timeout 3600 "$BIN/decode-batch-gate" "$Q35" --batch 4 --mode strict \
    > "$R/battery-decode-batch-q35-strict-equalized.log" 2>&1
  rc=$?; row decode-batch-q35-strict-equalized rc "$rc" 1
  tail -2 "$R/battery-decode-batch-q35-strict-equalized.log" | sed 's/^/  /'
  wait_idle
  env MEMRA_GATE_SEED=0 MEMRA_MMVQ=0 MEMRA_NO_FUSE_NORMQ=1 \
    flock /tmp/gpu5090.lock timeout 3600 "$BIN/decode-batch-gate" "$Q9J" --batch 4 --mode strict \
    > "$R/battery-decode-batch-q9j-strict-equalized.log" 2>&1
  rc=$?; row decode-batch-q9j-strict-equalized rc "$rc" 1
  tail -2 "$R/battery-decode-batch-q9j-strict-equalized.log" | sed 's/^/  /'
fi

if [ "$PHASE" = g26gen ] || [ "$PHASE" = all ]; then
  for rep in 1 2; do
    for arm in mode3 mode2; do
      bin="$W/target/release/run-gen"; [ "$arm" = mode3 ] && bin="$R/bin-preflip/run-gen"
      log="$R/g26-b2048gen-r$rep-$arm.log"
      wait_idle
      env MEMRA_NGEN=128 MEMRA_PROMPT_FILE="$PF2048" \
        flock /tmp/gpu5090.lock timeout 1800 "$bin" "$G26" > "$log" 2>&1
      pp=$(grep -oE "prefill [0-9]+ tok in [0-9.]+s = [0-9.]+ tok/s" "$log" | grep -oE "= [0-9.]+" | grep -oE "[0-9.]+" | tail -1)
      tg=$(grep -oE "generated [0-9]+ tokens in [0-9.]+s = [0-9.]+ tok/s" "$log" | grep -oE "= [0-9.]+" | grep -oE "[0-9.]+" | tail -1)
      gate=$(grep -c "MATCH" "$log")
      thash=$(grep -A1 "^generated" "$log" | grep "tokens:" | sha256sum | cut -c1-16)
      row "g26-b2048gen-$arm" prefill_toks "${pp:-null}" "$rep"
      row "g26-b2048gen-$arm" decode_toks "${tg:-null}" "$rep"
      row "g26-b2048gen-$arm" match_lines "${gate:-0}" "$rep"
      echo "  [g26-b2048gen/$arm rep$rep] tokens_sha=$thash" | tee -a "$R/token-hashes.log"
    done
  done
  echo "g26 b2048 shas MUST be identical across arms (gemma door env-explicit, naked closed)"
fi

echo "FOLLOWUP-DONE phase=$PHASE $(date -u +%FT%TZ)"
