#!/usr/bin/env bash
# H100 full-board bench (task #22, 2026-07-30): every model bw24 serves on sm_90a,
# interleaved same-session vs llama.cpp (same fork, same GGUF artifacts — the 5090
# board protocol). Runs ON the darklanes-bench box.
#
#   bash tools/full-board-bench-h100.sh [cell-filter] [out.jsonl]
#
# Cells: {q9,q35,g12,g26,g31,e4b}-plain-{short,d1736}. q27 has no 90a-compatible
# artifact (NVFP4 is sm_120a-only) — honestly absent. Spec cells are a follow-up:
# the 9B MTP drafts are NVFP4-head (120a) and gemma drafters are unproven on 90a.
# llama flags mirror the 5090 board's per-model bests (swept THERE — noted as such;
# an H100 llama flag sweep is future work, do not lower bars without one).
set -u
cd "$(dirname "$0")/.."
FILTER="${1:-.}"
OUT="${2:-research/tune-data/h100board-20260730.jsonl}"
LOGD="${OUT%.jsonl}-logs"; mkdir -p "$LOGD" "$(dirname "$OUT")"
N_PAIRS="${N_PAIRS:-3}"
LB=$HOME/llama.cpp/build/bin/llama-bench
BW=target/release
M=$HOME/models
GDIR=research/gemma4-bringup
TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
GIT_SHA=$(git -C . rev-parse --short HEAD 2>/dev/null || echo rsync-tree)

wait_idle() {
  for _ in $(seq 60); do
    local busy
    busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
    [ "$busy" -eq 0 ] && return
    sleep 5
  done
}

row() { # cell arm toks
  printf '{"ts":"%s","git":"%s","rig":"h100-darklanes","cell":"%s","arm":"%s","toks":%s}\n' \
    "$TS" "$GIT_SHA" "$1" "$2" "$3" >> "$OUT"
  echo "  [$1/$2] $3 tok/s"
}

llama_tg() { # cell model depth extra...
  local cell=$1 model=$2 depth=$3; shift 3
  "$LB" -m "$model" -ngl 999 -p 512 -n 128 -d "$depth" -r 3 "$@" 2>>"$LOGD/$cell-llama.log" \
    | tee -a "$LOGD/$cell-llama.log" \
    | grep -E "tg128" | grep -oE '[0-9.]+ ±' | grep -oE '^[0-9.]+' | tail -1
}

bw24_plain() { # cell model promptfile ngen
  local cell=$1 model=$2 pf=$3 ngen=$4
  local log="$LOGD/$cell-bw24.log"
  if echo "$pf" | grep -q 'ids'; then
    # shellcheck disable=SC2046
    BW24_NGEN="$ngen" timeout 900 $BW/run-gen "$model" $(cat "$pf") 2>&1 | tee -a "$log" \
      | grep -oE "= [0-9.]+ tok/s" | tail -1 | grep -oE "[0-9.]+"
  else
    BW24_NGEN="$ngen" BW24_PROMPT_FILE="$pf" timeout 900 $BW/run-gen "$model" 2>&1 | tee -a "$log" \
      | grep -oE "= [0-9.]+ tok/s" | tail -1 | grep -oE "[0-9.]+"
  fi
}

plain_cell() { # cell model bwprompt ngen depth llama_extra...
  local cell=$1 model=$2 pf=$3 ngen=$4 depth=$5; shift 5
  echo "$cell" | grep -qE "$FILTER" || return 0
  [ -f "$model" ] || { echo "== $cell SKIP (no artifact $model)"; return 0; }
  echo "== $cell (plain, interleaved x$N_PAIRS) =="
  for _ in $(seq 1 $N_PAIRS); do
    wait_idle; t=$(llama_tg "$cell" "$model" "$depth" "$@"); row "$cell" llama "${t:-0}"
    wait_idle; t=$(bw24_plain "$cell" "$model" "$pf" "$ngen"); row "$cell" bw24 "${t:-0}"
  done
}

echo "=== H100 FULL BOARD $TS git=$GIT_SHA filter=$FILTER ==="
SHORT="$GDIR/e4b-chat-watercycle-ids.txt"
DEPTH="$GDIR/depth-prompt-1736-ids.txt"

plain_cell q9-plain-short  "$M/Qwen3.5-9B-Q8_0.gguf"            "$SHORT" 128 0    -fa 1 -ctk q8_0 -ctv q5_1
plain_cell q9-plain-d1736  "$M/Qwen3.5-9B-Q8_0.gguf"            "$DEPTH" 128 1736 -fa 1 -ctk q8_0 -ctv q5_1
plain_cell q35-plain-short "$M/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"  "$SHORT" 128 0    -fa 1 -ctk q8_0 -ctv q5_1
plain_cell q35-plain-d1736 "$M/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"  "$DEPTH" 128 1736 -fa 1 -ctk q8_0 -ctv q5_1
plain_cell g12-plain-short "$M/gemma-4-12b-it-qat-q4_0.gguf"    "$SHORT" 128 0    -fa 1
plain_cell g12-plain-d1736 "$M/gemma-4-12b-it-qat-q4_0.gguf"    "$DEPTH" 128 1736 -fa 1
plain_cell g26-plain-short "$M/gemma-4-26B_q4_0-it.gguf"        "$SHORT" 128 0    -fa 1
plain_cell g26-plain-d1736 "$M/gemma-4-26B_q4_0-it.gguf"        "$DEPTH" 128 1736 -fa 1
plain_cell g31-plain-short "$M/gemma-4-31B_q4_0-it.gguf"        "$SHORT" 128 0    -fa 1
plain_cell g31-plain-d1736 "$M/gemma-4-31B_q4_0-it.gguf"        "$DEPTH" 128 1736 -fa 1
plain_cell e4b-plain-short "$M/gemma-4-E4B_q4_0-it.gguf"        "$SHORT" 128 0    -fa 1
plain_cell e4b-plain-d1736 "$M/gemma-4-E4B_q4_0-it.gguf"        "$DEPTH" 128 1736 -fa 1

echo "=== H100 FULL BOARD DONE $(date -u +%H:%M:%SZ) — rows in $OUT ==="
