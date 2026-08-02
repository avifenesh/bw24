#!/usr/bin/env bash
# agentworld-iq4xs: bar cells — plain-vs-llama AND best-vs-best in one interleaved session
# (o9b-cell shape, research/ornith-bar-20260802/run-9b-cell.sh; llama per-class best on
# this NextN-less GGUF is PLAIN — its draftless spec doors are structurally broken on the
# qwen35 M-RoPE arch, screen receipts research/ornith-bar-20260802/llama-spec-doors-screen.md).
# memra arm: run-spec at the per-class swept K (plain + spec interleaved in-process, one
# invocation = one interleaved pair). llama arm: llama-completion, swept-best board flags,
# greedy --ignore-eos, 256 new tokens. Rep loop OUTSIDE the class loop, N=3 pairs, every
# GPU run under flock /tmp/gpu5090.lock; co-resident llama-server --embedding allowlisted.
# usage: run-bar-cell.sh [nreps]   (per-class K baked below after the acc sweep)
set -u
N_PAIRS=${1:-3}
W=/home/avifenesh/projects/bw24-aw-iq4xs
R=$W/research/agentworld-iq4xs-20260802
LBIN=/home/avifenesh/projects/llama.cpp/build/bin
PDIR=$W/research/e2e/prompts
OUT=$R/aw-cell.jsonl
MODEL=/data/ai-ml/hf-models/agentworld-35b-gguf/Qwen-AgentWorld-35B-A3B-UD-IQ4_XS.gguf
DRAFT=/data/ai-ml/hf-models/agentworld-35b-gguf/draft-agentworld-owntrim-nvfp4head-q4blk.gguf
CLASSES="p1-code-short p2-code-medium p3-agentic-long"
K_p1=${K_p1:-2}; K_p2=${K_p2:-2}; K_p3=${K_p3:-2}   # per-class swept K (edit after acc sweep)
LFLAGS="-ngl 999 -fa on -ctk q8_0 -ctv q5_1 -c 8192 --temp 0"

TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
GIT_SHA=$(git -C "$W" rev-parse --short HEAD)
PROFILE=$(cat /sys/firmware/acpi/platform_profile 2>/dev/null || echo unknown)
LLAMA_VER=$("$LBIN/llama-completion" --version 2>&1 | head -1)
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
    sleep 5; n=$((n+1)); [ $n -gt 120 ] && { echo "wait_idle timeout (busy=$busy)"; break; }
  done
}
row() { # arm class metric value rep
  printf '{"ts":"%s","git":"%s","llama_build":"%s","cell":"aw-iq4xs-bar","arm":"%s","class":"%s","metric":"%s","value":%s,"rep":%s,"profile":"%s","temp_c":%s}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$GIT_SHA" "$LLAMA_VER" "$1" "$2" "$3" "$4" "$5" "$PROFILE" \
    "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)" >> "$OUT"
  echo "  [$1/$2 rep$5] $3 = $4"
}

llama_plain() { # class rep
  local cls=$1 rep=$2 log="$R/aw-llama-plain-$1-rep$2.log"
  flock /tmp/gpu5090.lock timeout 900 "$LBIN/llama-completion" -m "$MODEL" -f "$PDIR/$cls.txt" \
    -n 256 --ignore-eos --no-warmup $LFLAGS > "$log" 2>&1
  local pp_ms pp_n tg_ms tg_n
  pp_ms=$(grep "prompt eval time" "$log" | grep -oE "= *[0-9.]+ ms" | grep -oE "[0-9.]+" | head -1)
  pp_n=$(grep "prompt eval time" "$log" | grep -oE "/ *[0-9]+ tokens" | grep -oE "[0-9]+" | head -1)
  tg_ms=$(grep -E "common_perf_print: +eval time" "$log" | grep -oE "= *[0-9.]+ ms" | grep -oE "[0-9.]+" | head -1)
  tg_n=$(grep -E "common_perf_print: +eval time" "$log" | grep -oE "/ *[0-9]+ runs" | grep -oE "[0-9]+" | head -1)
  if [ -z "${pp_ms:-}" ] || [ -z "${tg_ms:-}" ]; then row llama-plain "$cls" ERROR 0 "$rep"; return; fi
  row llama-plain "$cls" prefill_toks "$(echo "$pp_n $pp_ms" | awk '{printf "%.1f", $1/($2/1000)}')" "$rep"
  row llama-plain "$cls" prefill_s "$(echo "$pp_ms" | awk '{printf "%.4f", $1/1000}')" "$rep"
  row llama-plain "$cls" decode_toks "$(echo "$tg_n $tg_ms" | awk '{printf "%.2f", $1/($2/1000)}')" "$rep"
  row llama-plain "$cls" n_prompt "$pp_n" "$rep"
  row llama-plain "$cls" n_gen "$tg_n" "$rep"
}

memra_arm() { # class rep K
  local cls=$1 rep=$2 K=$3 log="$R/aw-memra-spec-$cls-rep$rep.log"
  MEMRA_MTP_DRAFT="$DRAFT" MEMRA_SPEC_K=$K MEMRA_NGEN=256 MEMRA_PROMPT="$(cat "$PDIR/$cls.txt")" \
    flock /tmp/gpu5090.lock timeout 1800 "$W/target/release/run-spec" "$MODEL" > "$log" 2>&1
  local np plain spec prime_p prime_s acc
  np=$(grep -aoE "\-> [0-9]+ tokens" "$log" | grep -oE "[0-9]+" | head -1)
  plain=$(grep -aoE "\[generate\] +[0-9]+ tok in [0-9.]+s = [0-9.]+ tok/s" "$log" | grep -oE "= [0-9.]+" | grep -oE "[0-9.]+")
  prime_p=$(grep -a "\[generate\]" "$log" | grep -oE "prime [0-9.]+s" | grep -oE "[0-9.]+")
  spec=$(grep -aoE "\[generate_spec K=$K\] +[0-9]+ tok in [0-9.]+s = [0-9.]+ tok/s" "$log" | grep -oE "= [0-9.]+" | grep -oE "[0-9.]+")
  prime_s=$(grep -a "\[generate_spec" "$log" | grep -oE "prime [0-9.]+s" | grep -oE "[0-9.]+")
  acc=$(grep -aoE "acceptance: [0-9]+/[0-9]+ = [0-9.]+%" "$log" | grep -oE "= [0-9.]+" | grep -oE "[0-9.]+")
  if [ -z "${spec:-}" ]; then row memra-spec "$cls" ERROR 0 "$rep"; return; fi
  grep -qa "SELF-CONSISTENCY PASS" "$log" && row memra-spec "$cls" self_consistency 1 "$rep" \
                                          || row memra-spec "$cls" self_consistency 0 "$rep"
  row memra-spec "$cls" spec_k "$K" "$rep"
  row memra-spec "$cls" n_prompt "${np:-0}" "$rep"
  row memra-spec "$cls" plain_decode_toks "${plain:-0}" "$rep"
  row memra-spec "$cls" decode_toks "$spec" "$rep"
  row memra-spec "$cls" accept_pct "${acc:-0}" "$rep"
  row memra-spec "$cls" prime_plain_s "${prime_p:-0}" "$rep"
  row memra-spec "$cls" prime_spec_s "${prime_s:-0}" "$rep"
  [ -n "${np:-}" ] && [ -n "${prime_s:-}" ] && \
    row memra-spec "$cls" prefill_toks "$(echo "$np $prime_s" | awk '{printf "%.1f", $1/$2}')" "$rep"
}

echo "=== AGENTWORLD IQ4_XS BAR CELL $TS git=$GIT_SHA profile=$PROFILE llama=[$LLAMA_VER] K=($K_p1,$K_p2,$K_p3) ===" | tee -a "$R/aw-cell-console.log"
for rep in $(seq 1 "$N_PAIRS"); do
  for cls in $CLASSES; do
    case $cls in p1*) K=$K_p1;; p2*) K=$K_p2;; *) K=$K_p3;; esac
    wait_idle; llama_plain "$cls" "$rep"
    wait_idle; memra_arm "$cls" "$rep" "$K"
  done
done 2>&1 | tee -a "$R/aw-cell-console.log"
echo "CELL-DONE $(date -u +%FT%TZ)" | tee -a "$R/aw-cell-console.log"
