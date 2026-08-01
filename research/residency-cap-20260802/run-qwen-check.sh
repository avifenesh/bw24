#!/bin/bash
# residency-cap: Qwen3.6-35B-A3B UD-IQ4_XS residency check on the 5090 — does the supported
# MoE model spill under the residency policy? Board-shape run-gen (pp512.txt, NGEN=128) x3,
# decision line + decode + argmax per rep. Board row guard: plain decode 178.2 (512ctx).
# usage: run-qwen-check.sh <tag> [nreps]   (tag e.g. "pre" = current default, "post" = patched)
set -u
TAG=${1:?tag required (pre|post)}
N=${2:-3}
W=/home/avifenesh/projects/wt-residency-cap
R=$W/research/residency-cap-20260802
PF=$W/research/e2e/prompts/pp512.txt
OUT=$R/qwen-check.jsonl
Q35B=/data/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf

TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
GIT_SHA=$(git -C "$W" rev-parse --short HEAD)
PROFILE=$(cat /sys/firmware/acpi/platform_profile 2>/dev/null || echo unknown)

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
    local busy; busy=$(busy_procs); [ "$busy" -eq 0 ] && break
    sleep 5; n=$((n+1)); [ $n -gt 120 ] && { echo "wait_idle timeout (busy=$busy)"; break; }
  done
}
row() { # metric value rep
  printf '{"ts":"%s","git":"%s","cell":"q36-35b-residency-%s","arm":"naked","metric":"%s","value":%s,"rep":%s,"profile":"%s","temp_c":%s}\n' \
    "$TS" "$GIT_SHA" "$TAG" "$1" "$2" "$3" "$PROFILE" \
    "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)" >> "$OUT"
  echo "  [qwen-$TAG rep$3] $1 = $2"
}

for rep in $(seq 1 "$N"); do
  wait_idle
  log="$R/qwen-$TAG-rep$rep.log"; vram="$R/qwen-$TAG-rep$rep.vram"
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$vram"; sleep 1; done ) &
  sampler=$!
  MEMRA_NGEN=128 MEMRA_PROMPT_FILE="$PF" \
    flock /tmp/gpu5090.lock timeout 900 "$W/target/release/run-gen" "$Q35B" > "$log" 2>&1
  rc=$?
  kill $sampler 2>/dev/null; wait $sampler 2>/dev/null
  pp=$(grep -oE "prefill [0-9]+ tok in [0-9.]+s = [0-9.]+ tok/s" "$log" | grep -oE "= [0-9.]+" | grep -oE "[0-9.]+" | tail -1)
  tg=$(grep -oE "generated [0-9]+ tokens in [0-9.]+s = [0-9.]+ tok/s" "$log" | grep -oE "= [0-9.]+" | grep -oE "[0-9.]+" | tail -1)
  gate=$(grep -c "MATCH" "$log")
  peak=$(sort -n "$vram" | tail -1)
  dec=$(grep -oE "resident-experts decision: .* -> (RESIDENT|SLRU cache)" "$log" | grep -oE "(RESIDENT|SLRU cache)$" | head -1 | tr ' ' '_')
  grep "resident-experts decision" "$log" | tee -a "$R/qwen-decision-lines.log"
  if [ -z "${tg:-}" ]; then row ERROR "$rc" "$rep"; tail -5 "$log"; continue; fi
  row prefill_toks "${pp:-0}" "$rep"
  row decode_toks "$tg" "$rep"
  row argmax_match "$gate" "$rep"
  row peak_vram_mib "${peak:-0}" "$rep"
  echo "  [qwen-$TAG rep$rep] decision=$dec"
done
echo "QWEN-CHECK-$TAG-DONE $(date -u +%FT%TZ)"
