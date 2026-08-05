#!/usr/bin/env bash
# lane/chunkinv-flip full evaluation — grain-free quantize-then-attend as DEFAULT.
# All params baked as literals (background fan-out law). Each cell takes its OWN flock hold
# (short holds; 3 other lanes share the 5090). Raw log per cell; parse the log, never the pipe.
set -uo pipefail
cd "$(dirname "$0")/../.."
D=research/chunkinv-flip-20260805
L=$D/logs
mkdir -p "$L"
LOCK=/tmp/gpu5090.lock
M9=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
M27=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
MST=/data/ai-ml/hf-models/qwen35-9b-nvfp4-st-modelopt
P=./target/release/concat-prime-probe
NLLWIN=research/fp8st-20260804/mmq-v2/nll-window.txt
PP512=research/chunk-invariance-20260805/prompt-pp512.txt
PP6257=research/chunk-invariance-20260805/prompt-pp6257.txt
PROBE_TXT=tools/fast-gate/prompts/probe.txt
run() { # run <cell-name> <cmd...>: one flock hold, tee'd raw log
    local cell="$1"; shift
    echo "### cell $cell $(date -Is)"
    flock -w 7200 "$LOCK" "$@" > "$L/$cell.log" 2>&1
    local rc=$?
    echo "cell $cell exit=$rc"
    return $rc
}

echo "### run-eval $(date -Is) HEAD=$(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader

# ---- A. EXACTNESS CLASS: run-gen argmax gates (9B NVFP4, 27B NVFP4, ST arm) -------------
# The fix changes SHORT-prompt prime arithmetic (single-chunk prompts now read quantized KV).
# run-gen's own gates: prefill/decode argmax MATCH (prime-dispatch-blind, must stay MATCH)
# + batched-prime vs tokenwise (gap #46 — THE gate the fix moves; near-tie flips REPORT).
run A-runggen-9b  env MEMRA_NGEN=20 MEMRA_NMEASURE=0 MEMRA_PROMPT_FILE=$PROBE_TXT timeout 900 target/release/run-gen "$M9"
run A-runggen-27b env MEMRA_NGEN=20 MEMRA_NMEASURE=0 MEMRA_PROMPT_FILE=$PROBE_TXT timeout 900 target/release/run-gen "$M27"
run A-runggen-st  env MEMRA_NGEN=20 MEMRA_NMEASURE=0 MEMRA_PROMPT_FILE=$PROBE_TXT timeout 1800 target/release/run-gen "$MST"

# Golden-token comparison (q9/k27 fast-gate goldens): flips here are the CONTRACT CHANGE
# (near-tie class), quantified by cell C below — report, not fail.
run A-golden-q9  env MEMRA_NGEN=20 MEMRA_NMEASURE=0 MEMRA_PROMPT_FILE=$PROBE_TXT timeout 900 target/release/run-gen "$M9"
grep -oE "^tokens: \[[0-9, ]*\]" "$L/A-golden-q9.log" | head -1 > "$L/A-golden-q9.tokens"
run A-golden-k27 env MEMRA_NGEN=20 MEMRA_NMEASURE=0 MEMRA_PROMPT_FILE=$PROBE_TXT timeout 900 target/release/run-gen "$M27"
grep -oE "^tokens: \[[0-9, ]*\]" "$L/A-golden-k27.log" | head -1 > "$L/A-golden-k27.tokens"

# ---- B. QUALITY: NLL window through the SERVING PRIME, new vs legacy seam ---------------
# nllwin lm_heads prime_cache's own hidden stack — the pass the fix changes. Window = the
# frozen mmq-v2 held-out GSM8K prose. Chunk arms: monolithic-class (default 4096 > 1024)
# and chunked (256) so both the short-prompt and boundary regimes are scored.
run B-nll-new-mono   env                        $P "$M9" nllwin --prompt-a "@$NLLWIN" --window 1024
run B-nll-old-mono   env MEMRA_PRIME_F32CHUNK0=1 $P "$M9" nllwin --prompt-a "@$NLLWIN" --window 1024
run B-nll-new-c256   env                        $P "$M9" nllwin --prompt-a "@$NLLWIN" --window 1024 --chunk 256
run B-nll-old-c256   env MEMRA_PRIME_F32CHUNK0=1 $P "$M9" nllwin --prompt-a "@$NLLWIN" --window 1024 --chunk 256

# ---- C. CONTRACT QUANTIFICATION: teacher-forced disagreements, new vs legacy ------------
# tfcmp = the mmq-v2 flip protocol: per-position argmax disagreement count + legacy-margin
# percentile per flip (near-tie class = far below median).
run C-tfcmp-mono env $P "$M9" tfcmp --prompt-a "@$NLLWIN" --window 1024
run C-tfcmp-c256 env $P "$M9" tfcmp --prompt-a "@$NLLWIN" --window 1024 --chunk 256

# ---- D. PERF: prime-only pp512 + pp6257-class, interleaved N=5, new vs legacy -----------
# ppprime times prime_cache itself (fresh cache per rep, median of 3 in-process reps).
# Interleave arms per rep (clock-drift law). Quantized-KV attention may WIN (cheaper reads).
for rep in 1 2 3 4 5; do
  for ARM in new old; do
    EX=(); [ "$ARM" = old ] && EX=(MEMRA_PRIME_F32CHUNK0=1)
    run "D-pp512-$ARM-r$rep"  env "${EX[@]}" $P "$M9" ppprime --prompt-a "@$PP512"  --reps 3 --warmup 1
    run "D-pp6257-$ARM-r$rep" env "${EX[@]}" $P "$M9" ppprime --prompt-a "@$PP6257" --reps 3 --warmup 1
  done
done
for f in "$L"/D-pp*.log; do
  tag=$(basename "$f" .log)
  med=$(grep -oE "ppprime MEDIAN: [0-9]+ tok in [0-9.]+s = [0-9.]+ tok/s" "$f" | grep -oE "= [0-9.]+" | grep -oE "[0-9.]+")
  toks=$(grep -oE "ppprime MEDIAN: [0-9]+ tok" "$f" | grep -oE "[0-9]+")
  echo "{\"cell\":\"$tag\",\"prompt_tokens\":${toks:-null},\"tok_s\":${med:-null}}" >> "$L/D-perf.jsonl"
done

# ---- E. INVARIANCE + BATTERY: kernel-check, run-spec K=1..8, serve-smoke ----------------
run E-kernel-check timeout 1800 target/release/kernel-check
run E-runspec-9b   env MEMRA_NGEN=32 MEMRA_PROMPT_FILE=$PROBE_TXT timeout 3600 target/release/run-spec "$M9"
# serve-smoke manages its own server lifecycle; still one flock hold (GPU-resident server)
run E-serve-smoke  tools/serve-smoke.sh

echo "### run-eval done $(date -Is)"
