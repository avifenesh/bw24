#!/bin/bash
# ornith-bar: Ornith-9B BEST-vs-BEST cell (the deployment decider).
# memra at its best config = the ADOPTED own-gen trimmed drafter (research/ornith-drafters-20260801,
# K=3, serving-legal defaults otherwise) via run-spec (plain + spec interleaved in-process).
# llama.cpp at ITS best config on the same gguf: swept-best plain (-fa on -ctk q8_0 -ctv q5_1
# -ngl 999, the board convention) PLUS a fair best-effort screen of its draftless speculative
# doors (llama-lookup n-gram, llama-lookahead) — llama has no Ornith draft artifact.
# Protocol: board — interleaved same-session pairs (llama arm then memra arm), N=3 reps,
# rep loop OUTSIDE the class loop; every GPU run under flock /tmp/gpu5090.lock; the co-resident
# llama-server --embedding is allowlisted and untouched.
# usage: run-9b-cell.sh <screen|cell>
set -u
PHASE=${1:-cell}
W=/home/avifenesh/projects/wt-ornith-bar
R=$W/research/ornith-bar-20260802
LBIN=/home/avifenesh/projects/llama.cpp/build/bin
PDIR=$W/research/e2e/prompts
OUT=$R/o9b-cell.jsonl
O9B=/data/ai-ml/hf-models/ornith-1.0-9b-gguf/ornith-1.0-9b-Q8_0.gguf
DRAFT=/data/ai-ml/hf-models/ornith-1.0-9b-gguf/draft-ornith9b-owntrim-nvfp4head-q4blk.gguf
CLASSES="p1-code-short p2-code-medium p3-agentic-long"
N_PAIRS=3
LFLAGS="-ngl 999 -fa on -ctk q8_0 -ctv q5_1 -c 8192 --temp 0"

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
    sleep 5; n=$((n+1)); [ $n -gt 120 ] && { echo "wait_idle timeout (busy=$busy)"; break; }
  done
}
row() { # arm class metric value rep
  printf '{"ts":"%s","git":"%s","cell":"o9b-best","arm":"%s","class":"%s","metric":"%s","value":%s,"rep":%s,"profile":"%s","temp_c":%s}\n' \
    "$TS" "$GIT_SHA" "$1" "$2" "$3" "$4" "$5" "$PROFILE" \
    "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)" >> "$OUT"
  echo "  [$1/$2 rep$5] $3 = $4"
}

# --- llama plain arm: llama-completion greedy, --ignore-eos, 256 new tokens, swept-best flags
# (this fork's llama-cli refuses -no-cnv and points at llama-completion)
llama_plain() { # class rep
  local cls=$1 rep=$2 log="$R/o9b-llama-plain-$1-rep$2.log"
  flock /tmp/gpu5090.lock timeout 900 "$LBIN/llama-completion" -m "$O9B" -f "$PDIR/$cls.txt" \
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
  row llama-plain "$cls" decode_s "$(echo "$tg_ms" | awk '{printf "%.4f", $1/1000}')" "$rep"
  row llama-plain "$cls" n_prompt "$pp_n" "$rep"
  row llama-plain "$cls" n_gen "$tg_n" "$rep"
  local tot_ms; tot_ms=$(grep "total time" "$log" | grep -oE "= *[0-9.]+ ms" | grep -oE "[0-9.]+" | head -1)
  [ -n "${tot_ms:-}" ] && row llama-plain "$cls" total_s "$(echo "$tot_ms" | awk '{printf "%.4f", $1/1000}')" "$rep"
}

# --- llama draftless spec arms (screen): n-gram lookup + lookahead
llama_spec() { # tool class rep extra-args...
  local tool=$1 cls=$2 rep=$3; shift 3
  [ "$tool" = lookup ] && [ -n "${LOOKUP_DRAFT_MAX:-}" ] && set -- "$@" --spec-draft-n-max "$LOOKUP_DRAFT_MAX"
  local log="$R/o9b-llama-$tool-$cls-rep$rep.log"
  # -b 8192: the lookup/lookahead examples single-shot the whole prompt into one llama_decode
  # (no chunking) — n_batch must cover the longest prompt class (p3 = ~6.3k tokens).
  flock /tmp/gpu5090.lock timeout 900 "$LBIN/llama-$tool" -m "$O9B" -f "$PDIR/$cls.txt" \
    -n 256 -b 8192 $LFLAGS "$@" > "$log" 2>&1
  local enc dec encs decs
  enc=$(grep "encoded" "$log" | grep -oE "speed: *[0-9.]+" | grep -oE "[0-9.]+" | head -1)
  encs=$(grep "encoded" "$log" | grep -oE "in *[0-9.]+ seconds" | grep -oE "[0-9.]+" | head -1)
  dec=$(grep "decoded" "$log" | grep -oE "speed: *[0-9.]+" | grep -oE "[0-9.]+" | head -1)
  decs=$(grep "decoded" "$log" | grep -oE "in *[0-9.]+ seconds" | grep -oE "[0-9.]+" | head -1)
  if [ -z "${dec:-}" ]; then row "llama-$tool" "$cls" ERROR 0 "$rep"; return; fi
  row "llama-$tool" "$cls" prefill_toks "${enc:-0}" "$rep"
  row "llama-$tool" "$cls" prefill_s "${encs:-0}" "$rep"
  row "llama-$tool" "$cls" decode_toks "$dec" "$rep"
  row "llama-$tool" "$cls" decode_s "${decs:-0}" "$rep"
  local acc; acc=$(grep -oE "accept += +[0-9.]+%" "$log" | grep -oE "[0-9.]+" | head -1)
  [ -n "${acc:-}" ] && row "llama-$tool" "$cls" accept_pct "$acc" "$rep"
}

# --- memra best arm: run-spec with the adopted drafter, K=3 (plain + spec same invocation)
memra_arm() { # class rep
  local cls=$1 rep=$2 log="$R/o9b-memra-spec-$cls-rep$rep.log"
  MEMRA_MTP_DRAFT="$DRAFT" MEMRA_SPEC_K=3 MEMRA_NGEN=256 MEMRA_PROMPT="$(cat "$PDIR/$cls.txt")" \
    flock /tmp/gpu5090.lock timeout 1800 "$W/target/release/run-spec" "$O9B" > "$log" 2>&1
  local np plain spec prime_p prime_s acc
  np=$(grep -oE "\-> [0-9]+ tokens" "$log" | grep -oE "[0-9]+" | head -1)
  plain=$(grep -oE "\[generate\] +[0-9]+ tok in [0-9.]+s = [0-9.]+ tok/s" "$log" | grep -oE "= [0-9.]+" | grep -oE "[0-9.]+")
  prime_p=$(grep "\[generate\]" "$log" | grep -oE "prime [0-9.]+s" | grep -oE "[0-9.]+")
  spec=$(grep -oE "\[generate_spec K=3\] +[0-9]+ tok in [0-9.]+s = [0-9.]+ tok/s" "$log" | grep -oE "= [0-9.]+" | grep -oE "[0-9.]+")
  prime_s=$(grep "\[generate_spec" "$log" | grep -oE "prime [0-9.]+s" | grep -oE "[0-9.]+")
  acc=$(grep -oE "acceptance: [0-9]+/[0-9]+ = [0-9.]+%" "$log" | grep -oE "= [0-9.]+" | grep -oE "[0-9.]+")
  if [ -z "${spec:-}" ]; then row memra-spec "$cls" ERROR 0 "$rep"; return; fi
  grep -q "SELF-CONSISTENCY PASS" "$log" && row memra-spec "$cls" self_consistency 1 "$rep" \
                                         || row memra-spec "$cls" self_consistency 0 "$rep"
  row memra-spec "$cls" n_prompt "${np:-0}" "$rep"
  row memra-spec "$cls" plain_decode_toks "${plain:-0}" "$rep"
  row memra-spec "$cls" decode_toks "$spec" "$rep"
  row memra-spec "$cls" accept_pct "${acc:-0}" "$rep"
  row memra-spec "$cls" prime_plain_s "${prime_p:-0}" "$rep"
  row memra-spec "$cls" prime_spec_s "${prime_s:-0}" "$rep"
  [ -n "${np:-}" ] && [ -n "${prime_s:-}" ] && \
    row memra-spec "$cls" prefill_toks "$(echo "$np $prime_s" | awk '{printf "%.1f", $1/$2}')" "$rep"
}

echo "=== ORNITH-9B BEST-vs-BEST ($PHASE) $TS git=$GIT_SHA profile=$PROFILE ===" | tee -a "$R/o9b-cell-console.log"

if [ "$PHASE" = screen ]; then
  # N=1 screen of llama's draftless speculative doors vs one plain reading, per class.
  for cls in $CLASSES; do
    wait_idle; llama_plain "$cls" 0
    wait_idle; llama_spec lookup    "$cls" 0
    wait_idle; llama_spec lookahead "$cls" 0
  done 2>&1 | tee -a "$R/o9b-cell-console.log"
  echo "SCREEN-DONE $(date -u +%FT%TZ)" | tee -a "$R/o9b-cell-console.log"
  exit 0
fi

# cell: interleaved llama-best/memra-best pairs x N_PAIRS, rep loop outside class loop.
# LLAMA_BEST_ARM per class is decided from the screen (default plain; edit after screen).
LLAMA_BEST_p1=${LLAMA_BEST_p1:-plain}
LLAMA_BEST_p2=${LLAMA_BEST_p2:-plain}
LLAMA_BEST_p3=${LLAMA_BEST_p3:-plain}
for rep in $(seq 1 $N_PAIRS); do
  for cls in $CLASSES; do
    case $cls in p1*) best=$LLAMA_BEST_p1;; p2*) best=$LLAMA_BEST_p2;; *) best=$LLAMA_BEST_p3;; esac
    wait_idle
    if [ "$best" = plain ]; then llama_plain "$cls" "$rep"; else llama_spec "$best" "$cls" "$rep"; fi
    wait_idle; memra_arm "$cls" "$rep"
  done
done 2>&1 | tee -a "$R/o9b-cell-console.log"
echo "CELL-DONE $(date -u +%FT%TZ)" | tee -a "$R/o9b-cell-console.log"
