#!/usr/bin/env bash
# MTP-heal ACCEPTANCE BATTERY (§item 3, HANDOVER "BW24 DUAL-SHAPE").
#
# Measures MTP draft-head acceptance for ONE arm across a fixed prompt set + the 8-turn agent loop,
# N>=3 runs each, one JSONL row per (prompt,K,run). Run it twice — once for the bf16 full-precision
# CEILING, once for the NVFP4 daily GGUF — then feed the two JSONL files to acceptance_delta.py; the
# per-(prompt,K) delta is the quant hit on drafting.
#
# The two canonical arms (paths verified on this rig 2026-07-08):
#   CEILING (bf16, natural mtp.* head, full precision — SLOW, exact):
#     FULL_PREC=1 ARM=bf16-fullprec \
#       tools/acceptance_battery.sh /data/ai-ml/hf-models/qwen35-9b-hf out-bf16.jsonl
#   QUANT (NVFP4 GGUF, embedded nextn head):
#     ARM=nvfp4 \
#       tools/acceptance_battery.sh /data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf out-nvfp4.jsonl
#
# Env knobs:
#   ARM=<label>              row tag (default: bf16-fullprec if FULL_PREC=1 else nvfp4)
#   FULL_PREC=0|1            sets BW24_FULL_PREC (bf16 ST ceiling arm) — default 0
#   N=3                      runs per (prompt,K) for the static prompts
#   KS="1 2 3 4 6 8"         K values (one run-spec invocation per K, for clean per-K parsing)
#   NGEN=128                 tokens generated per run
#   PROMPTS="p1 p2 p3"       which fixed prompts (files research/e2e/prompts/<id>-*.txt)
#   RUN_AGENTLOOP=1          also run the 8-turn accumulative agent loop (default 1)
#   EXTRA_ENV="..."          extra env words passed to run-spec (e.g. "BW24_SPEC_HPOST=1")
#   RUNSPEC=./target/release/run-spec   TIMEOUT=1800   PYBIN=python3
set -euo pipefail

MODEL="${1:?usage: acceptance_battery.sh <model_path> <out.jsonl>}"
OUT="${2:?missing out.jsonl}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

FULL_PREC="${FULL_PREC:-0}"
ARM="${ARM:-$([ "$FULL_PREC" = 1 ] && echo bf16-fullprec || echo nvfp4)}"
N="${N:-3}"
KS="${KS:-1 2 3 4 6 8}"
NGEN="${NGEN:-128}"
PROMPTS="${PROMPTS:-p1 p2 p3}"
RUN_AGENTLOOP="${RUN_AGENTLOOP:-1}"
EXTRA_ENV="${EXTRA_ENV:-}"
RUNSPEC="${RUNSPEC:-./target/release/run-spec}"
TIMEOUT="${TIMEOUT:-1800}"
PYBIN="${PYBIN:-python3}"
PARSE="$ROOT/tools/acceptance_parse.py"

# shellcheck disable=SC2206
EXTRA_ENV_ARR=($EXTRA_ENV)
FP_FLAG=""; [ "$FULL_PREC" = "1" ] && FP_FLAG="--full-prec"

# resolve a prompt id (p1/p2/p3) to its file
prompt_file() {
  local id="$1"
  local f
  f=$(ls research/e2e/prompts/${id}-*.txt 2>/dev/null | head -1 || true)
  [ -n "$f" ] || { echo "no prompt file for id '$id'" >&2; return 1; }
  echo "$f"
}

echo "[battery] arm=$ARM model=$MODEL full_prec=$FULL_PREC N=$N KS=($KS) ngen=$NGEN -> $OUT"
[ "$FULL_PREC" = 1 ] && echo "[battery] FULL PRECISION mode — SLOW is expected (f32 oracle path)."
: > "$OUT"   # fresh file

for pid in $PROMPTS; do
  PF="$(prompt_file "$pid")"
  echo "[battery] prompt=$pid file=$PF"
  for k in $KS; do
    for run in $(seq 1 "$N"); do
      echo "[battery]   $ARM $pid K=$k run=$run/$N ..."
      OUTLOG="$(env "${EXTRA_ENV_ARR[@]}" \
          ${FULL_PREC:+BW24_FULL_PREC=$FULL_PREC} \
          BW24_NGEN="$NGEN" BW24_SPEC_K="$k" BW24_SPEC_STATS=1 \
          BW24_PROMPT_FILE="$PF" \
          timeout "$TIMEOUT" "$RUNSPEC" "$MODEL" 2>&1 || true)"
      printf '%s\n' "$OUTLOG" | "$PYBIN" "$PARSE" --out "$OUT" --arm "$ARM" \
          --prompt "$pid" --k "$k" --run "$run" --model "$MODEL" --ngen "$NGEN" $FP_FLAG \
          --extra "$EXTRA_ENV"
      ACC="$(printf '%s\n' "$OUTLOG" | grep -oE 'acceptance:[^%]*%' | head -1 || true)"
      echo "[battery]     -> ${ACC:-<no acceptance parsed>}"
    done
  done
done

if [ "$RUN_AGENTLOOP" = "1" ]; then
  echo "[battery] agent-loop (8 turns, accumulative) ..."
  FULL_PREC="$FULL_PREC" K="$(echo $KS | awk '{print $NF==8?3:$1}')" NGEN="256" \
    tools/agent_loop_acceptance.sh "$MODEL" "$OUT" "$ARM" "${EXTRA_ENV_ARR[@]}"
fi

echo "[battery] DONE arm=$ARM -> $OUT  ($(wc -l <"$OUT") rows)"
echo "[battery] delta table: tools/acceptance_delta.py out-bf16.jsonl out-nvfp4.jsonl"
