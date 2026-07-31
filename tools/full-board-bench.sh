#!/usr/bin/env bash
# FULL-BOARD interleaved bench — every supported model, memra vs llama.cpp, both arms
# alternated inside one session/thermal window (post-unified-merge picture, 2026-07-30).
#
# Protocol: SERIAL (one engine on the GPU at a time), idle-gate between arms, N_PAIRS
# alternations per cell, tee'd raw logs + JSONL rows. llama at its documented swept-best
# per COMPETITOR-SETUP.md (qwen: -fa 1 -ctk q8_0 -ctv q5_1 + MTP server; gemma: -fa 1
# f16 KV + MTP server at per-model n-max optimum). memra at naked defaults + the board's
# documented spec configs (MEMRA_MTP_DRAFT owntrim for qwen, gemma-gate manifest configs).
#
# Usage: tools/full-board-bench.sh [cell-regex]
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH=/usr/local/cuda-13.1/bin:$PATH
export GGML_CUDA_GRAPH_OPT=1
gpu-full-power on >/dev/null 2>&1 || true

M=/data/ai-ml/hf-models
LB=/home/avifenesh/projects/llama.cpp/build/bin/llama-bench
LS=/home/avifenesh/projects/llama.cpp/build/bin/llama-server
BW=./target/release
PDIR=research/e2e/prompts
GDIR=research/gemma4-bringup
LOGD=research/tune-data/fullboard-logs
OUT=research/tune-data/fullboard-20260730.jsonl
FILTER="${1:-.}"
N_PAIRS="${N_PAIRS:-2}"
PORT=8099
mkdir -p "$LOGD"

TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
GIT_SHA=$(git rev-parse --short HEAD)
PROFILE=$(cat /sys/firmware/acpi/platform_profile 2>/dev/null || echo unknown)

busy_procs() {  # count GPU compute apps, ignoring the allowlisted --embedding co-resident
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
    local busy
    busy=$(busy_procs)
    local clk
    clk=$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ "$busy" -eq 0 ] && [ "${clk:-2000}" -lt 1200 ] && break
    sleep 5; n=$((n+1)); [ $n -gt 120 ] && { echo "wait_idle: 10min timeout (busy=$busy clk=$clk)"; break; }
  done
}

row() {  # cell arm toks extra
  printf '{"ts":"%s","git":"%s","cell":"%s","arm":"%s","toks":%s%s,"profile":"%s"}\n' \
    "$TS" "$GIT_SHA" "$1" "$2" "$3" "${4:-}" "$PROFILE" >> "$OUT"
  echo "  [$1/$2] $3 tok/s ${4:-}"
}

# ---------- arm runners ----------
llama_bench_tg() {  # cell model depth extra_flags...
  local cell=$1 model=$2 depth=$3; shift 3
  local log="$LOGD/$cell-llama.log"
  "$LB" -m "$model" -ngl 999 -p 512 -n 128 -d "$depth" -r 3 "$@" 2>>"$log" | tee -a "$log" \
    | grep -E "tg128" | grep -oE '[0-9.]+ ±' | grep -oE '^[0-9.]+' | tail -1
}

memra_plain() {  # cell model promptfile ngen  (pf ending .txt+ids = token ids as args;
                # anything else = TEXT via MEMRA_PROMPT_FILE — knife-edge synthetic-id
                # prompts flake the argmax gate on address-lottery ULPs, ledger 2026-07-30)
  local cell=$1 model=$2 pf=$3 ngen=$4
  local log="$LOGD/$cell-memra.log"
  if echo "$pf" | grep -q 'ids'; then
    # shellcheck disable=SC2046
    MEMRA_NGEN="$ngen" timeout 600 $BW/run-gen "$model" $(cat "$pf") 2>&1 | tee -a "$log" \
      | grep -oE "= [0-9.]+ tok/s" | tail -1 | grep -oE "[0-9.]+"
  else
    MEMRA_NGEN="$ngen" MEMRA_PROMPT_FILE="$pf" timeout 600 $BW/run-gen "$model" 2>&1 | tee -a "$log" \
      | grep -oE "= [0-9.]+ tok/s" | tail -1 | grep -oE "[0-9.]+"
  fi
}

llama_server_start() {  # model draft nmax pmin kvflags...
  local model=$1 draft=$2 nmax=$3 pmin=$4; shift 4
  local args=(-m "$model" -ngl 999 -fa on -c 16384 --parallel 1 --temp 0
              --host 127.0.0.1 --port $PORT "$@")
  if [ "$draft" = "self" ]; then
    # embedded NextN head (35B class): spec flags, no -md
    args+=(--spec-type draft-mtp --spec-draft-n-max "$nmax" --spec-draft-p-min "$pmin")
  elif [ -n "$draft" ]; then
    args+=(-md "$draft" --spec-type draft-mtp --spec-draft-n-max "$nmax"
           --spec-draft-p-min "$pmin" -ngld 999)
  fi
  "$LS" "${args[@]}" > "$LOGD/llama-server-$(date +%s).log" 2>&1 &
  SPID=$!
  for _ in $(seq 240); do curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && return 0; sleep 2; done
  echo "SERVER FAILED to come up"; kill $SPID 2>/dev/null; return 1
}
llama_server_stop() { kill "${SPID:-0}" 2>/dev/null; wait "${SPID:-0}" 2>/dev/null || true; wait_idle; }

llama_completion_tps() {  # promptfile ids|text ngen
  python3 - "$1" "$2" "$3" << 'PY'
import json,sys,urllib.request
src, kind, ngen = sys.argv[1], sys.argv[2], int(sys.argv[3])
raw = open(src).read()
prompt = [int(t) for t in raw.split()] if kind == "ids" else raw
req = urllib.request.Request('http://127.0.0.1:8099/completion',
  data=json.dumps({'prompt': prompt, 'n_predict': ngen, 'temperature': 0,
                   'cache_prompt': False, 'ignore_eos': True}).encode(),
  headers={'Content-Type':'application/json'})
r = json.loads(urllib.request.urlopen(req, timeout=900).read())
t = r['timings']
d = r.get('timings', {})
extra = ''
if 'draft_n' in d and d.get('draft_n'):
    acc = d.get('draft_n_accepted', 0) / max(1, d['draft_n'])
    extra = f" draft_accept={acc:.3f}"
print(f"{t['predicted_per_second']:.2f}{extra}")
PY
}

memra_spec_qwen() {  # cell model trim k promptfile ngen
  local cell=$1 model=$2 trim=$3 k=$4 pf=$5 ngen=$6
  local log="$LOGD/$cell-memra.log"
  MEMRA_MTP_DRAFT="$trim" MEMRA_SPEC_K="$k" MEMRA_NGEN="$ngen" MEMRA_PROMPT="$(cat "$pf")" \
    timeout 900 $BW/run-spec "$model" 2>&1 | tee -a "$log" \
    | grep -oE "[0-9.]+ tok/s" | tail -1 | grep -oE "[0-9.]+"
}

memra_spec_gemma() {  # cell model draft k ranks promptfile ngen
  local cell=$1 model=$2 draft=$3 k=$4 ranks=$5 pf=$6 ngen=$7
  local log="$LOGD/$cell-memra.log"
  local envs=(MEMRA_SPEC_ONLY=1 "MEMRA_SPEC=$k" "MEMRA_DRAFT=$draft" "MEMRA_NGEN=$ngen")
  [ -n "$ranks" ] && envs+=("MEMRA_GEMMA_DRAFT_RANKS=$ranks")
  # shellcheck disable=SC2046
  env "${envs[@]}" timeout 900 $BW/gemma-gate "$model" $(cat "$pf") 2>&1 | tee -a "$log" \
    | grep -oE "spec: [0-9.]+" | grep -oE "[0-9.]+" | tail -1
}

# ---------- cell drivers (interleaved: llama arm then memra arm, x N_PAIRS) ----------
plain_cell() {  # cell model bwprompt ngen depth llama_extra...
  local cell=$1 model=$2 pf=$3 ngen=$4 depth=$5; shift 5
  echo "$cell" | grep -qE "$FILTER" || return 0
  echo "== $cell (plain, interleaved x$N_PAIRS) =="
  for _ in $(seq 1 $N_PAIRS); do
    wait_idle; t=$(llama_bench_tg "$cell" "$model" "$depth" "$@"); row "$cell" llama "${t:-0}"
    wait_idle; t=$(memra_plain "$cell" "$model" "$pf" "$ngen"); row "$cell" memra "${t:-0}"
  done
}

spec_cell_qwen() {  # cell model llama_draft nmax pmin trim k promptfile ngen
  local cell=$1 model=$2 ldraft=$3 nmax=$4 pmin=$5 trim=$6 k=$7 pf=$8 ngen=$9
  echo "$cell" | grep -qE "$FILTER" || return 0
  echo "== $cell (spec, interleaved x$N_PAIRS) =="
  for _ in $(seq 1 $N_PAIRS); do
    wait_idle
    if llama_server_start "$model" "$ldraft" "$nmax" "$pmin" -ctk q8_0 -ctv q5_1; then
      out=$(llama_completion_tps "$pf" text "$ngen"); llama_server_stop
      row "$cell" llama "${out%% *}" "$(echo "$out" | grep -oE 'draft_accept=[0-9.]+' | sed 's/^/,"accept_note":"/;s/$/"/' )"
    fi
    wait_idle; t=$(memra_spec_qwen "$cell" "$model" "$trim" "$k" "$pf" "$ngen"); row "$cell" memra "${t:-0}"
  done
}

spec_cell_gemma() {  # cell model llama_draft nmax bwdraft k ranks idsfile ngen
  local cell=$1 model=$2 ldraft=$3 nmax=$4 bwdraft=$5 k=$6 ranks=$7 pf=$8 ngen=$9
  echo "$cell" | grep -qE "$FILTER" || return 0
  echo "== $cell (spec, interleaved x$N_PAIRS) =="
  for _ in $(seq 1 $N_PAIRS); do
    wait_idle
    if llama_server_start "$model" "$ldraft" "$nmax" 0.1; then   # gemma best: f16 KV (no ctk/ctv)
      out=$(llama_completion_tps "$pf" ids "$ngen"); llama_server_stop
      row "$cell" llama "${out%% *}" "$(echo "$out" | grep -oE 'draft_accept=[0-9.]+' | sed 's/^/,"accept_note":"/;s/$/"/' )"
    fi
    wait_idle; t=$(memra_spec_gemma "$cell" "$model" "$bwdraft" "$k" "$ranks" "$pf" "$ngen"); row "$cell" memra "${t:-0}"
  done
}

# serial guard at entry (allowlisted --embedding co-resident excluded)
[ "$(busy_procs)" -eq 0 ] || { echo "ABORT: non-embedding GPU compute procs present"; exit 1; }

# 512-id deterministic prompt for qwen plain cells (bench.sh pattern)
QP=/tmp/qwen-512ids.txt
awk 'BEGIN{for(i=0;i<512;i++){printf "%d ", 100+(i*7)%900}}' > "$QP"

echo "=== FULL BOARD $TS git=$GIT_SHA profile=$PROFILE filter=$FILTER ==="

# ---------------- PLAIN CELLS ----------------
plain_cell q9-plain   "$M/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf"      "$PDIR/pp512.txt" 128 0 -fa 1 -ctk q8_0 -ctv q5_1
plain_cell q27-plain  "$M/qwen36-27b-nvfp4-mtp/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf"   "$PDIR/pp512.txt" 128 0 -fa 1 -ctk q8_0 -ctv q5_1
plain_cell q35-plain  "$M/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"            "$PDIR/pp512.txt" 128 0 -fa 1 -ctk q8_0 -ctv q5_1
plain_cell g12-plain-short "/data/ai-ml/models/gemma-4-12b-it-qat/gemma-4-12b-it-qat-q4_0.gguf" "$GDIR/e4b-chat-watercycle-ids.txt" 128 0 -fa 1
plain_cell g12-plain-d1736 "/data/ai-ml/models/gemma-4-12b-it-qat/gemma-4-12b-it-qat-q4_0.gguf" "$GDIR/depth-prompt-1736-ids.txt" 128 1736 -fa 1
plain_cell g26-plain-short "$M/gemma4-26b-a4b-qat-gguf/gemma-4-26B_q4_0-it.gguf"    "$GDIR/e4b-chat-watercycle-ids.txt" 128 0 -fa 1
plain_cell g26-plain-d1736 "$M/gemma4-26b-a4b-qat-gguf/gemma-4-26B_q4_0-it.gguf"    "$GDIR/depth-prompt-1736-ids.txt" 128 1736 -fa 1
plain_cell g31-plain-short "$M/gemma4-31b-qat-gguf/gemma-4-31B_q4_0-it.gguf"        "$GDIR/e4b-chat-watercycle-ids.txt" 128 0 -fa 1
plain_cell g31-plain-d1736 "$M/gemma4-31b-qat-gguf/gemma-4-31B_q4_0-it.gguf"        "$GDIR/depth-prompt-1736-ids.txt" 128 1736 -fa 1
plain_cell e4b-plain-short "$M/gemma4-e4b-qat-gguf/gemma-4-E4B_q4_0-it.gguf"        "$GDIR/e4b-chat-watercycle-ids.txt" 128 0 -fa 1

# ---------------- QWEN SPEC (3 prompt classes; llama MTP server arm) ----------------
for P in p1-code-short p2-code-medium p3-agentic-long; do
  spec_cell_qwen "q9-spec-$P"  "$M/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf" \
    "" 0 0 "$M/qwen35-9b-nvfp4-gguf/draft-9b-owntrim-nvfp4head-q4blk.gguf" 3 "$PDIR/$P.txt" 256
  spec_cell_qwen "q27-spec-$P" "$M/qwen36-27b-nvfp4-mtp/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf" \
    "$M/qwen36-27b-nvfp4-mtp/mtp-Qwen3.6-27B-NVFP4.gguf" 3 0.1 \
    "$M/qwen36-27b-nvfp4-mtp/draft-daily-owntrim-nvfp4head-q4blk.gguf" 3 "$PDIR/$P.txt" 256
  spec_cell_qwen "q35-spec-$P" "$M/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf" \
    "self" 3 0.1 \
    "$M/qwen36-35b-moe/draft-35b-owntrim-nvfp4head-q4blk.gguf" 2 "$PDIR/$P.txt" 256
done

# ---------------- GEMMA SPEC (token-id parity via /completion ids arrays) ----------------
spec_cell_gemma g12-spec-chat  "/data/ai-ml/models/gemma-4-12b-it-qat/gemma-4-12b-it-qat-q4_0.gguf" \
  "$M/gemma4-12b-mtp-gguf-q8/gemma-4-12B-it-qat-assistant-MTP-Q8_0.gguf" 2 \
  "$M/gemma4-12b-mtp-gguf-q8/gemma-4-12B-it-qat-assistant-MTP-Q8_0.gguf" 4 \
  "$GDIR/gemma4-12b-owngen-ranks-32768.gguf.txt" "$GDIR/e4b-chat-watercycle-ids.txt" 128
spec_cell_gemma g12-spec-d1736 "/data/ai-ml/models/gemma-4-12b-it-qat/gemma-4-12b-it-qat-q4_0.gguf" \
  "$M/gemma4-12b-mtp-gguf-q8/gemma-4-12B-it-qat-assistant-MTP-Q8_0.gguf" 2 \
  "$M/gemma4-12b-mtp-gguf-q8/gemma-4-12B-it-qat-assistant-MTP-Q8_0.gguf" 4 \
  "$GDIR/gemma4-12b-owngen-ranks-32768.gguf.txt" "$GDIR/depth-prompt-1736-ids.txt" 128
spec_cell_gemma g26-spec-short "$M/gemma4-26b-a4b-qat-gguf/gemma-4-26B_q4_0-it.gguf" \
  "$M/gemma4-26b-a4b-qat-gguf/drafter/MTP/mtp-gemma-4-26B-A4B-it-Q4_0.gguf" 4 \
  "$M/gemma4-26b-a4b-qat-gguf/drafter/MTP/mtp-gemma-4-26B-A4B-it-Q4_0.gguf" 4 \
  "$GDIR/gemma4-26b-owngen-ranks-32768.gguf.txt" "$GDIR/e4b-chat-watercycle-ids.txt" 128
spec_cell_gemma g26-spec-d1736 "$M/gemma4-26b-a4b-qat-gguf/gemma-4-26B_q4_0-it.gguf" \
  "$M/gemma4-26b-a4b-qat-gguf/drafter/MTP/mtp-gemma-4-26B-A4B-it-Q4_0.gguf" 6 \
  "$M/gemma4-26b-a4b-qat-gguf/drafter/MTP/mtp-gemma-4-26B-A4B-it-Q4_0.gguf" 6 \
  "$GDIR/gemma4-26b-owngen-ranks-32768.gguf.txt" "$GDIR/depth-prompt-1736-ids.txt" 128
spec_cell_gemma g31-spec-chat  "$M/gemma4-31b-qat-gguf/gemma-4-31B_q4_0-it.gguf" \
  "$M/gemma4-31b-tooluse-gguf/gemma-4-31B-it-Q4_0-MTP.gguf" 4 \
  "$M/gemma4-31b-tooluse-gguf/gemma-4-31B-it-Q4_0-MTP.gguf" 3 \
  "$GDIR/gemma4-31b-owngen-ranks-32768.txt" "$GDIR/e4b-chat-watercycle-ids.txt" 128
spec_cell_gemma g31-spec-d1736 "$M/gemma4-31b-qat-gguf/gemma-4-31B_q4_0-it.gguf" \
  "$M/gemma4-31b-tooluse-gguf/gemma-4-31B-it-Q4_0-MTP.gguf" 6 \
  "$M/gemma4-31b-tooluse-gguf/gemma-4-31B-it-Q4_0-MTP.gguf" 6 \
  "$GDIR/gemma4-31b-owngen-ranks-32768.txt" "$GDIR/depth-prompt-1736-ids.txt" 128
spec_cell_gemma e4b-spec-short "$M/gemma4-e4b-qat-gguf/gemma-4-E4B_q4_0-it.gguf" \
  "" 0 \
  "$M/gemma4-e4b-qat-gguf/drafter/MTP/gemma-4-E4B-it-assistant.Q8_0.gguf" 6 \
  "" "$GDIR/e4b-chat-watercycle-ids.txt" 128

echo "=== FULL BOARD DONE $(date -u +%FT%TZ) — rows in $OUT ==="
