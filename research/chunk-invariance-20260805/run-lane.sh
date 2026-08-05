#!/usr/bin/env bash
# lane/chunk-invariance — ONE lock hold does everything (three lanes share the 5090, so
# batch every measurement per hold instead of taking the card four times).
#
# Phases:
#   A  BASELINE root-cause: chunkinv at 2048/64/32 on the two prompts the original finding
#      named (97-tok turn 1, 149-tok turn 2) + the --profile per-row razor that separates
#      "flat GEMM-m band" from "precision-class STEP at the first chunk boundary".
#   B  GEMM m-dependence razor (the leak-1 receipt): does a row's value move when only the
#      batch height m changes? No chunking involved — pure kernel property.
#   C  FIX arm: the same chunkinv sweep under MEMRA_PRIME_INVARIANT=1. Byte-identity here is
#      the whole claim.
#   D  CANARY: the gate must be able to FAIL. Run the fix arm with a deliberately
#      mismatched grain across arms — greedy streams must diverge, proving teeth.
#   E  PERF: prefill battery, INTERLEAVED (off,on,off,on,... — the H100 lane's law 1), at a
#      prompt long enough to actually chunk, so the invariance cost is measured not guessed.
set -uo pipefail
cd "$(dirname "$0")/../.."
D=research/chunk-invariance-20260805
L=$D/logs
mkdir -p "$L"
M=${M:-/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf}
P=./target/release/concat-prime-probe
echo "### lane/chunk-invariance $(date -Is) model=$(basename "$M")"
nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader

# ---- A: baseline root cause -------------------------------------------------------------
for T in 1 2; do
  echo "=== A baseline turn$T ==="
  $P "$M" chunkinv --prompt-a "@$D/prompt-turn$T.txt" --chunks 2048,64,32 --steps 48 \
      --profile --jsonl "$L/A-base-turn$T.jsonl" 2>&1 | tee "$L/A-base-turn$T.log" | tail -20
done

# ---- B: GEMM m-dependence (leak 1 receipt, no chunking) ---------------------------------
echo "=== B gemm m-razor (wq) ==="
$P "$M" gemm --weight wq --ta 32 --lmin 1 --lmax 48 2>&1 | tee "$L/B-gemm-wq.log" | tail -12
echo "=== B gemm m-razor (head) ==="
$P "$M" gemm --weight head --ta 32 --lmin 1 --lmax 48 2>&1 | tee "$L/B-gemm-head.log" | tail -6

# ---- C: the fix ------------------------------------------------------------------------
for T in 1 2; do
  echo "=== C invariant turn$T ==="
  MEMRA_PRIME_INVARIANT=1 MEMRA_PRIME_GRAIN=32 \
    $P "$M" chunkinv --prompt-a "@$D/prompt-turn$T.txt" --chunks 2048,64,32 --steps 48 \
      --profile --jsonl "$L/C-fix-turn$T.jsonl" 2>&1 | tee "$L/C-fix-turn$T.log" | tail -14
done

# ---- D: canary — the probe and the gate must both be able to FAIL -----------------------
# D1: the GRAIN is an explicit numeric knob, so two different grains MUST produce different
# text. This is the cross-RUN canary, so it compares the probe's own reference streams (a
# single-chunk chunkinv call has nothing to compare and always says INVARIANT — that shape
# was the first version of this phase and it was vacuous; keep the two-run diff).
echo "=== D1 canary: grain 32 vs 64 MUST differ (proves the probe measures something) ==="
for G in 32 64; do
  MEMRA_PRIME_INVARIANT=1 MEMRA_PRIME_GRAIN=$G \
    $P "$M" chunkinv --prompt-a "@$D/prompt-turn2.txt" --chunks 2048,32 --steps 24 \
      --jsonl "$L/D1-canary-g$G.jsonl" > "$L/D1-canary-g$G.log" 2>&1
  echo "  grain=$G ref_argmax=$(grep -oE 'argmax=[0-9]+' "$L/D1-canary-g$G.log" | head -1)"
done
python3 - "$L/D1-canary-g32.jsonl" "$L/D1-canary-g64.jsonl" <<'PY'
import json, sys
a = [json.loads(l) for l in open(sys.argv[1])]
b = [json.loads(l) for l in open(sys.argv[2])]
# both arms are internally invariant; the QUESTION is whether the grain changed the answer
same = a[0]["argmax_ref"] == b[0]["argmax_ref"] and a[0]["logit_maxdiff"] == b[0]["logit_maxdiff"]
print(f"  D1 grain32 ref_argmax={a[0]['argmax_ref']} grain64 ref_argmax={b[0]['argmax_ref']}")
print("  D1 verdict:", "grain is a live numeric knob (arms differ) — probe HAS teeth"
      if not same else "*** grain changed nothing — SUSPECT: probe may not be measuring ***")
PY
# D2: the GATE's own canary path — asserted expectation flipped, must report a diverged canary.
echo "=== D2 gate canary (tools/chunk-invariance-gate.sh --canary) ==="
tools/chunk-invariance-gate.sh --canary 2>&1 | tail -4
echo "=== D3 gate default (expect-variant, today's honest contract) ==="
tools/chunk-invariance-gate.sh 2>&1 | tail -3
echo "=== D4 gate under the door (expect-invariant) ==="
tools/chunk-invariance-gate.sh --expect-invariant 2>&1 | tail -3

# ---- E: perf, interleaved --------------------------------------------------------------
# THE HONEST QUESTION. The invariance door's cost is NOT "chunk 4096 vs grain 32" — that
# compares two different segmentations and would just re-measure the known chunk-size perf
# curve. The cost of INVARIANCE is: at the segmentation you would ship, does forcing it to be
# grain-pinned cost anything? So both arms run the SAME effective segmentation and the only
# difference is WHO chose it:
#   off: MEMRA_PRIME_CHUNK=<G>                      (chunk knob steers, today's default path)
#   on : MEMRA_PRIME_INVARIANT=1 MEMRA_PRIME_GRAIN=<G>, MEMRA_PRIME_CHUNK deliberately WRONG
#        (4096) — the door must ignore it and still segment at G
# Same boundaries, same m, same kernels => any delta is the door's own overhead, and the
# expectation is ~0. The REAL cost of the fix is a POLICY cost (the shipped default chunk must
# become the grain everywhere), which is a config decision, not a kernel tax — VERDICT.md
# states it that way and the number below is what proves the mechanism itself is free.
# run-gen's MEMRA_PP_ONLY is the timing harness (median of MEMRA_PP_REPS timed prime_cache
# calls, fresh cache per rep — the same pass PRIME_NANOS measures), not a wall-clock wrapper.
G=${G:-2048}
PPP=./target/release/run-gen
for LEN in 6257 512; do
  PF="$D/prompt-pp$LEN.txt"
  python3 - "$PF" "$LEN" <<'PY'
import sys
# deterministic ~LEN-token english-ish filler (token count checked from run-gen's own report)
p, n = sys.argv[1], int(sys.argv[2])
w = ("copies overlap with compute and pinned buffers bound host memory while bytes per token "
     "set the budget for every resident expert projection in the serving path ").split()
open(p, "w").write(" ".join(w[i % len(w)] for i in range(int(n * 0.78))) + "\n")
PY
  echo "=== E perf pp$LEN interleaved N=5 (off,on x5), grain=$G ==="
  for rep in 1 2 3 4 5; do
    for ARM in off on; do
      if [ "$ARM" = on ]; then
        EX=(MEMRA_PRIME_INVARIANT=1 "MEMRA_PRIME_GRAIN=$G" MEMRA_PRIME_CHUNK=4096)
      else
        EX=("MEMRA_PRIME_CHUNK=$G")
      fi
      LG="$L/E-pp$LEN-$ARM-r$rep.log"
      env "${EX[@]}" MEMRA_PP_ONLY=1 MEMRA_PP_REPS=3 MEMRA_PP_WARMUP=1 \
          MEMRA_PROMPT_FILE="$PF" timeout 900 $PPP "$M" > "$LG" 2>&1
      TOK=$(grep -oE "pp-only MEDIAN: [0-9]+ tok in [0-9.]+s = [0-9.]+ tok/s" "$LG" \
            | grep -oE "= [0-9.]+ tok/s" | grep -oE "[0-9.]+")
      NT=$(grep -oE "pp-only MEDIAN: [0-9]+ tok" "$LG" | grep -oE "[0-9]+")
      echo "{\"phase\":\"E\",\"pp\":$LEN,\"rep\":$rep,\"arm\":\"$ARM\",\"grain\":$G,\
\"prompt_tokens\":${NT:-null},\"tok_s\":${TOK:-null}}" | tee -a "$L/E-perf.jsonl"
    done
  done
done
echo "### done $(date -Is)"
