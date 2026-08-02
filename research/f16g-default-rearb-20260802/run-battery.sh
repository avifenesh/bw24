#!/bin/bash
# f16g-default-rearb: FULL exactness battery for mode 2 as the sm_120a naked default.
# Everything runs NAKED on the flip build (naked = the candidate default) unless stated.
#   kc     : kernel-check with q35 real weights (f16g-kq-direct iq4_xs/iq3_s subcases live)
#   gen    : run-gen naked gen512 (pp512, NGEN=128) argmax + token shas —
#            q35 x3 (mode-2-class anchor e94b6553fde7b9a0 expected: the iq-direct F16G=2
#            stream IS the new naked stream), o35b x2 (anchor c0c12c3b350dc7f5 — dispatch
#            unchanged), kat x2 (mode-2-class anchor 9102ffd0b8241a65), g26 x1 (must equal
#            this lane's pre-flip mode3 arm sha — gemma door stays env-explicit).
#   dbatch : decode-batch gates 1/2/3, q35 + q9j (9B dense judge), config + strict.
#            The in-binary MEMRA_MOE_F16G=0 pin is default-flip-invariant ("0" closes the
#            door under every mode semantics) — this proves it.
#   pbatch : prime-batch-gate --carried, q35 + q9j + o35b + kat (cross-request concat prime
#            under the new naked prime config).
#   serve  : memra-server greedy c1-vs-c16 byte identity (16 prompts, 96 max_tokens,
#            temp 0 seed 0), q35 + o35b, fresh refs (tools/check-batch-exact.py).
#   spec   : run-spec K=1..8 self-consistency, q35 + o35b, own-trim drafters, p2 prompt,
#            NGEN=64.
#   graph  : decode-dc-gate + graph-decode-gate + graph-session-gate, q35 + o35b
#            (graph capture paths under the new prime config — SLRU-graph precedent).
# usage: run-battery.sh <kc|gen|dbatch|pbatch|serve|spec|graph|all>
set -u
PHASE=${1:-all}
W=/home/avifenesh/projects/wt-f16g-rearb
R=$W/research/f16g-default-rearb-20260802
BIN=$W/target/release
PF512=$W/research/e2e/prompts/pp512.txt
PFP2=$W/research/e2e/prompts/p2-code-medium.txt
OUT=$R/gates.jsonl
Q35=/data/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
Q35_DRAFT=/data/ai-ml/hf-models/qwen36-35b-moe/draft-35b-owntrim-nvfp4head-q4blk.gguf
O35B=/data/ai-ml/hf-models/ornith-1.0-35b-gguf/ornith-1.0-35b-Q4_K_M.gguf
O35B_DRAFT=/data/ai-ml/hf-models/ornith-1.0-35b-gguf/draft-ornith35b-owntrim-nvfp4head-q4blk.gguf
KAT=/data/ai-ml/hf-models/kat-coder-v25-dev-gguf/Kwaipilot_KAT-Coder-V2.5-Dev-IQ4_XS.gguf
G26=/data/ai-ml/hf-models/gemma4-26b-a4b-qat-gguf/gemma-4-26B_q4_0-it.gguf
Q9J=/home/avifenesh/models/qwen3.5-9b-judge-q8_0.gguf

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
row() { # cell metric value rep
  printf '{"ts":"%s","git":"%s","cell":"%s","metric":"%s","value":%s,"rep":%s,"profile":"%s","temp_c":%s}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$GIT_SHA" "$1" "$2" "$3" "$4" "$PROFILE" \
    "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)" >> "$OUT"
  echo "  [$1 rep$4] $2 = $3"
}

gen512() { # tag model rep
  local tag=$1 model=$2 rep=$3
  local log="$R/battery-gen-$tag-rep$rep.log"
  wait_idle
  env MEMRA_NGEN=128 MEMRA_PROMPT_FILE="$PF512" \
    flock /tmp/gpu5090.lock timeout 1800 "$BIN/run-gen" "$model" > "$log" 2>&1
  local rc=$? gate thash
  gate=$(grep -c "MATCH" "$log")
  thash=$(grep -A1 "^generated" "$log" | grep "tokens:" | sha256sum | cut -c1-16)
  row "gen-$tag" match_lines "${gate:-0}" "$rep"
  row "gen-$tag" rc "$rc" "$rep"
  echo "  [gen-$tag rep$rep] tokens_sha=$thash" | tee -a "$R/token-hashes.log"
}

exec > >(tee -a "$R/battery-console.log") 2>&1
echo "=== F16G-REARB BATTERY phase=$PHASE $TS git=$GIT_SHA profile=$PROFILE ==="

if [ "$PHASE" = kc ] || [ "$PHASE" = all ]; then
  wait_idle
  flock /tmp/gpu5090.lock timeout 3600 "$BIN/kernel-check" "$Q35" > "$R/battery-kernel-check.log" 2>&1
  rc=$?
  fails=$(grep -c FAIL "$R/battery-kernel-check.log")
  oks=$(grep -c " OK" "$R/battery-kernel-check.log")
  echo "kernel-check rc=$rc FAIL-lines=$fails OK-lines=$oks"
  row kernel-check rc "$rc" 1; row kernel-check fail_lines "$fails" 1
fi

if [ "$PHASE" = gen ] || [ "$PHASE" = all ]; then
  for rep in 1 2 3; do gen512 q35 "$Q35" "$rep"; done
  for rep in 1 2;   do gen512 o35b "$O35B" "$rep"; done
  for rep in 1 2;   do gen512 kat "$KAT" "$rep"; done
  gen512 g26 "$G26" 1
  echo "anchors: q35 mode-2-class e94b6553fde7b9a0 | o35b c0c12c3b350dc7f5 (unchanged) |"
  echo "         kat mode-2-class 9102ffd0b8241a65 | g26 == headline mode3-arm sha"
fi

if [ "$PHASE" = dbatch ] || [ "$PHASE" = all ]; then
  for tag in q35 q9j; do
    m=$Q35; [ "$tag" = q9j ] && m=$Q9J
    wait_idle
    flock /tmp/gpu5090.lock timeout 3600 "$BIN/decode-batch-gate" "$m" > "$R/battery-decode-batch-$tag.log" 2>&1
    rc=$?; row "decode-batch-$tag" rc "$rc" 1
    grep -E "gate 1|gate 2|gate 3|ALL GREEN|FAIL" "$R/battery-decode-batch-$tag.log" | tail -5 | sed 's/^/  /'
    wait_idle
    flock /tmp/gpu5090.lock timeout 3600 "$BIN/decode-batch-gate" "$m" --mode strict > "$R/battery-decode-batch-$tag-strict.log" 2>&1
    rc=$?; row "decode-batch-$tag-strict" rc "$rc" 1
    grep -E "ALL GREEN|FAIL" "$R/battery-decode-batch-$tag-strict.log" | tail -3 | sed 's/^/  /'
  done
fi

if [ "$PHASE" = pbatch ] || [ "$PHASE" = all ]; then
  for tag in q35 q9j o35b kat; do
    case "$tag" in q35) m=$Q35;; q9j) m=$Q9J;; o35b) m=$O35B;; kat) m=$KAT;; esac
    wait_idle
    flock /tmp/gpu5090.lock timeout 3600 "$BIN/prime-batch-gate" "$m" --carried > "$R/battery-prime-batch-$tag.log" 2>&1
    rc=$?; row "prime-batch-$tag" rc "$rc" 1
    grep -E "ALL GREEN|FAIL|DIVERGED" "$R/battery-prime-batch-$tag.log" | tail -3 | sed 's/^/  /'
  done
fi

if [ "$PHASE" = serve ] || [ "$PHASE" = all ]; then
  for tag in q35 o35b; do
    m=$Q35; [ "$tag" = o35b ] && m=$O35B
    wait_idle
    flock /tmp/gpu5090.lock bash -c '
      W='"$W"'; R='"$R"'; tag='"$tag"'; m='"$m"'
      cd "$W"
      MEMRA_ADDR="127.0.0.1:8123" MEMRA_MODELS="m=$m" \
        ./target/release/memra-server > "$R/server-$tag.log" 2>&1 &
      pid=$!
      up=0
      for _ in $(seq 1 180); do
        sleep 2
        curl -s -m 2 "http://127.0.0.1:8123/v1/models" >/dev/null 2>&1 && { up=1; break; }
        kill -0 "$pid" 2>/dev/null || break
      done
      if [ "$up" -ne 1 ]; then echo "SERVER FAILED $tag (see server-$tag.log)"; kill "$pid" 2>/dev/null; exit 1; fi
      python3 tools/check-batch-exact.py --base "http://127.0.0.1:8123" --model m \
        --n 16 --max-tokens 96 --label "$tag-naked-mode2" \
        --out "$R/greedy-hash-$tag.jsonl" --ref "$R/greedy-refs-$tag.json" 2>&1 | tee "$R/greedy-hash-$tag.log"
      kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
      sleep 3
    '
    rc=$?; row "serve-c1c16-$tag" rc "$rc" 1
    grep -E "identical|moved|MISMATCH" "$R/greedy-hash-$tag.log" | tail -3 | sed 's/^/  /'
  done
fi

if [ "$PHASE" = spec ] || [ "$PHASE" = all ]; then
  wait_idle
  env MEMRA_MTP_DRAFT="$Q35_DRAFT" MEMRA_NGEN=64 MEMRA_PROMPT="$(cat "$PFP2")" \
    flock /tmp/gpu5090.lock timeout 3600 "$BIN/run-spec" "$Q35" > "$R/battery-spec-q35.log" 2>&1
  rc=$?
  echo "q35 run-spec rc=$rc PASS=$(grep -c "self-consistency: PASS" "$R/battery-spec-q35.log") FAIL=$(grep -ci fail "$R/battery-spec-q35.log")"
  row spec-q35 rc "$rc" 1
  row spec-q35 pass_lines "$(grep -c "self-consistency: PASS" "$R/battery-spec-q35.log")" 1
  wait_idle
  env MEMRA_MTP_DRAFT="$O35B_DRAFT" MEMRA_NGEN=64 MEMRA_PROMPT="$(cat "$PFP2")" \
    flock /tmp/gpu5090.lock timeout 3600 "$BIN/run-spec" "$O35B" > "$R/battery-spec-o35b.log" 2>&1
  rc=$?
  echo "o35b run-spec rc=$rc PASS=$(grep -c "self-consistency: PASS" "$R/battery-spec-o35b.log") FAIL=$(grep -ci fail "$R/battery-spec-o35b.log")"
  row spec-o35b rc "$rc" 1
  row spec-o35b pass_lines "$(grep -c "self-consistency: PASS" "$R/battery-spec-o35b.log")" 1
fi

if [ "$PHASE" = graph ] || [ "$PHASE" = all ]; then
  for tag in q35 o35b; do
    m=$Q35; [ "$tag" = o35b ] && m=$O35B
    for g in decode-dc-gate graph-decode-gate graph-session-gate; do
      wait_idle
      flock /tmp/gpu5090.lock timeout 3600 "$BIN/$g" "$m" > "$R/battery-$g-$tag.log" 2>&1
      rc=$?; row "$g-$tag" rc "$rc" 1
      tail -2 "$R/battery-$g-$tag.log" | sed 's/^/  /'
    done
  done
fi

echo "BATTERY-DONE phase=$PHASE $(date -u +%FT%TZ)"
