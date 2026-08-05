#!/usr/bin/env bash
# lane/chunk-invariance phase 2 — the questions phase 1 raised, batched into one lock hold.
set -uo pipefail
cd "$(dirname "$0")/../.."
D=research/chunk-invariance-20260805
L=$D/logs
M=${M:-/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf}
P=./target/release/concat-prime-probe
echo "### phase2 $(date -Is)"
nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader

# ---- F: WHICH leak? bisect the mechanism by disabling suspects one at a time ------------
# Phase 1 already refuted the documented cause: the prefill GEMM is m-INVARIANT (phase B),
# and the divergence begins EXACTLY at the first chunk boundary with a step from 0 to O(1)
# (phase A per-row profile), not as a flat band. That signature says the boundary crosses a
# numeric CLASS edge, not a reduction-order shift. The two candidate class edges at a
# boundary are (1) attention: chunk 0 reads f32 K/V via fa_prefill, later chunks read the
# QUANTIZED cache via fa_prefill_view_ws; (2) the GDN scan's carried state. Isolate:
#   F1 MEMRA_PRIME_DEQW=0  -> the other quantized-cache FA kernel. If divergence persists
#                             identically, the leak is the f32-vs-quantized CLASS, not that
#                             kernel's dequant-once workspace.
#   F2 MEMRA_GDN_CHUNKED=0 -> sequential GDN scan (removes the WY chunk segmentation as a
#                             variable). Divergence persisting => GDN is not the cause.
#   F3 MEMRA_KV_K/V pinned -> already default; record for the manifest.
for ARM in "F1 MEMRA_PRIME_DEQW=0" "F2 MEMRA_GDN_CHUNKED=0"; do
  TAG=${ARM%% *}; EX=${ARM#* }
  echo "=== $TAG ($EX) ==="
  env "$EX" $P "$M" chunkinv --prompt-a "@$D/prompt-turn2.txt" --chunks 2048,64,32 \
      --steps 48 --profile --jsonl "$L/$TAG.jsonl" > "$L/$TAG.log" 2>&1
  grep -E "profile chunk|verdict|^ +(64|32) \|" "$L/$TAG.log" | sed 's/^/  /'
done

# ---- G: does the door at the DEFAULT grain preserve today's shipped output? -------------
# The door pins segmentation to MEMRA_PRIME_GRAIN. If grain == the historical default chunk
# (4096) then for any prompt shorter than 4096 the door primes MONOLITHICALLY — exactly what
# today's default config does — so the door should be a NO-OP on the default path. That is
# what makes it safe to consider as a default. Assert it as bit-identity vs door-off.
echo "=== G door-at-default-grain vs default config (MUST be bit-identical) ==="
for T in 1 2; do
  $P "$M" chunkinv --prompt-a "@$D/prompt-turn$T.txt" --chunks 4096 --steps 32 \
      --jsonl "$L/G-off-turn$T.jsonl" > "$L/G-off-turn$T.log" 2>&1
  MEMRA_PRIME_INVARIANT=1 $P "$M" chunkinv --prompt-a "@$D/prompt-turn$T.txt" \
      --chunks 4096 --steps 32 --jsonl "$L/G-on-turn$T.jsonl" > "$L/G-on-turn$T.log" 2>&1
  A=$(grep -oE "argmax=[0-9]+ margin=[0-9.]+" "$L/G-off-turn$T.log" | head -1)
  B=$(grep -oE "argmax=[0-9]+ margin=[0-9.]+" "$L/G-on-turn$T.log" | head -1)
  echo "  turn$T off[$A] on[$B] -> $([ "$A" = "$B" ] && echo SAME || echo DIFFER)"
done

# ---- H: the door's REAL cost — forced fine grain on a long prompt -----------------------
# Phase E measured the door's mechanism overhead at a FIXED segmentation (flat, ~0.1%). The
# other half of the honest cost: if a rig needed a small chunk for transients, the door makes
# EVERY rig segment that small to stay invariant. Measure the pure segmentation cost curve so
# VERDICT.md can state the policy price, not guess it. Interleaved N=5 per grain.
PF="$D/prompt-pp6257.txt"
[ -f "$PF" ] || { echo "  (missing $PF — run run-lane.sh first)"; exit 1; }
echo "=== H segmentation cost curve, interleaved N=5 per grain ==="
for rep in 1 2 3 4 5; do
  for G in 4096 2048 512 64; do
    LG="$L/H-g$G-r$rep.log"
    env MEMRA_PRIME_INVARIANT=1 "MEMRA_PRIME_GRAIN=$G" MEMRA_PP_ONLY=1 MEMRA_PP_REPS=3 \
        MEMRA_PP_WARMUP=1 MEMRA_PROMPT_FILE="$PF" timeout 900 ./target/release/run-gen "$M" \
        > "$LG" 2>&1
    TOK=$(grep -oE "pp-only MEDIAN: [0-9]+ tok in [0-9.]+s = [0-9.]+ tok/s" "$LG" \
          | grep -oE "= [0-9.]+ tok/s" | grep -oE "[0-9.]+")
    echo "{\"phase\":\"H\",\"rep\":$rep,\"grain\":$G,\"tok_s\":${TOK:-null}}" \
      | tee -a "$L/H-graincost.jsonl"
  done
done
echo "### phase2 done $(date -Is)"
