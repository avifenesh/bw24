#!/usr/bin/env bash
# H100 board vs vLLM (task #22, 2026-07-30) — runs ON the darklanes-bench box.
# Per model, the pinned showdown protocol (bench_vllm.py): single-stream p~2048/g512,
# N=5 + warmup, decode_tps median + prefill_tps. bw24 arm = run-gen, same shape,
# 5 invocations. Same-session blocks per model (H100 SXM clocks are stable; the
# 122.6% decode pin was measured this way).
#
# ARTIFACT NOTE (honest row semantics): vLLM serves what an H100 user deploys
# (w8a8 / FP8 / bf16 HF checkpoints — vLLM rejects these GGUFs); bw24 serves its
# GGUF artifacts. Rows carry the artifact name; this is a cross-artifact,
# same-model comparison BY DESIGN (the user-facing question), not same-bytes.
#
#   bash tools/h100-vllm-board.sh [model-filter] [out.jsonl]
set -u
cd "$(dirname "$0")/.."
FILTER="${1:-.}"
OUT="${2:-research/tune-data/h100board-vllm-20260730.jsonl}"
LOGD="${OUT%.jsonl}-logs"; mkdir -p "$LOGD" "$(dirname "$OUT")"
VP=$HOME/vllm-env/bin/python
BV=$HOME/bw24/bench_vllm.py
BW=target/release/run-gen
M=$HOME/models
TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
GIT_SHA=$(git -C . rev-parse --short HEAD 2>/dev/null || echo rsync-tree)
NGEN=512

# ~2048-token prompt for bw24 (per-model tokenizer variance is a few %; prefill
# normalizes by the actual token count run-gen prints; decode is shape-independent).
FOX=/tmp/fox2048.txt
python3 -c 'print("The quick brown fox jumps over the lazy dog. " * 215, end="")' > "$FOX"

wait_idle() {
  for _ in $(seq 90); do
    local busy
    busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
    [ "$busy" -eq 0 ] && return
    sleep 5
  done
}

row() { # model arm artifact decode prefill
  printf '{"ts":"%s","git":"%s","rig":"h100-darklanes","cell":"%s-p2048g512","arm":"%s","artifact":"%s","decode_tps":%s,"prefill_tps":%s}\n' \
    "$TS" "$GIT_SHA" "$1" "$2" "$3" "$4" "${5:-0}" >> "$OUT"
  echo "  [$1/$2] decode $4 tok/s prefill ${5:-?} tok/s ($3)"
}

vllm_arm() { # name model_id tokenizer(optional)
  local name=$1 mid=$2 tokz=${3:-}
  local extra=()
  [ -n "$tokz" ] && extra=(--tokenizer "$tokz")
  wait_idle
  timeout 3600 $VP "$BV" --model "$mid" "${extra[@]}" --runs 5 \
    --out "$LOGD/$name-vllm.json" > "$LOGD/$name-vllm.log" 2>&1
  local dec pre
  dec=$(python3 -c "import json;d=json.load(open('$LOGD/$name-vllm.json'));print(d['decode_tps_median'])" 2>/dev/null)
  pre=$(python3 -c "import json;d=json.load(open('$LOGD/$name-vllm.json'));rs=sorted(x['prefill_tps'] for x in d['runs']);print(rs[len(rs)//2])" 2>/dev/null)
  [ -n "${dec:-}" ] && row "$name" vllm "$mid" "$dec" "$pre" \
    || { echo "  [$name/vllm] FAILED — $(tail -2 "$LOGD/$name-vllm.log" | head -1)"; row "$name" vllm "$mid" 0 0; }
}

bw24_arm() { # name gguf
  local name=$1 gguf=$2
  local log="$LOGD/$name-bw24.log"
  local decs=() pres=()
  for r in 1 2 3 4 5; do
    wait_idle
    BW24_NGEN=$NGEN BW24_PROMPT_FILE="$FOX" timeout 1800 $BW "$gguf" >> "$log" 2>&1
    local d p
    d=$(grep -oE "= [0-9.]+ tok/s \((Stage|graph)" "$log" | tail -1 | grep -oE "[0-9.]+" | head -1)
    [ -z "$d" ] && d=$(grep -oE "generated $NGEN tokens in [0-9.]+s = [0-9.]+" "$log" | tail -1 | grep -oE "[0-9.]+$")
    p=$(grep -oE "prefill [0-9]+ tok in [0-9.]+s = [0-9.]+" "$log" | tail -1 | grep -oE "[0-9.]+$")
    decs+=("${d:-0}"); pres+=("${p:-0}")
  done
  local dec pre
  dec=$(printf '%s\n' "${decs[@]}" | sort -n | sed -n 3p)
  pre=$(printf '%s\n' "${pres[@]}" | sort -n | sed -n 3p)
  row "$name" bw24 "$(basename "$gguf")" "$dec" "$pre"
}

cell() { # name vllm_id bw24_gguf [tokenizer]
  echo "$1" | grep -qE "$FILTER" || return 0
  [ -f "$3" ] || { echo "== $1 SKIP (no gguf $3)"; return 0; }
  echo "== $1 (vllm vs bw24, p2048/g512, N=5 medians) =="
  vllm_arm "$1" "$2" "${4:-}"
  bw24_arm "$1" "$3"
}

echo "=== H100 vLLM BOARD $TS git=$GIT_SHA filter=$FILTER ==="
cell q9  RedHatAI/Qwen3.5-9B-quantized.w8a8 "$M/Qwen3.5-9B-Q8_0.gguf"
cell q35 Qwen/Qwen3.6-35B-A3B-FP8          "$M/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"
cell g12 unsloth/gemma-4-12b-it            "$M/gemma-4-12b-it-qat-q4_0.gguf"
cell g26 RedHatAI/gemma-4-26b-a4b-it-FP8-dynamic        "$M/gemma-4-26B_q4_0-it.gguf"
cell g31 RedHatAI/gemma-4-31b-it-FP8-dynamic            "$M/gemma-4-31B_q4_0-it.gguf"
cell e4b unsloth/gemma-4-E4B-it            "$M/gemma-4-E4B_q4_0-it.gguf"
echo "=== H100 vLLM BOARD DONE $(date -u +%H:%M:%SZ) — rows in $OUT ==="
