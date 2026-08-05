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

# ---- D: canary — the gate must be able to fail ------------------------------------------
# Same door ON, but the GRAIN itself differs between the two runs. The grain is an explicit
# numeric knob, so this SHOULD diverge; if it does not, the probe is not measuring anything.
echo "=== D canary (grain 32 vs 64 under the door — MUST differ) ==="
for G in 32 64; do
  MEMRA_PRIME_INVARIANT=1 MEMRA_PRIME_GRAIN=$G \
    $P "$M" chunkinv --prompt-a "@$D/prompt-turn2.txt" --chunks 2048 --steps 24 \
      --jsonl "$L/D-canary-g$G.jsonl" 2>&1 | tail -4
done

# ---- E: perf, interleaved --------------------------------------------------------------
# pp-class prompt: concatenate the transcript turns until the token count clears the grain
# several times over, so the chunked path is genuinely exercised in both arms.
cat "$D"/prompt-turn*.txt "$D"/prompt-turn*.txt "$D"/prompt-turn*.txt > "$D/prompt-long.txt"
echo "=== E perf interleaved N=5 (off,on x5) ==="
for rep in 1 2 3 4 5; do
  for ARM in off on; do
    if [ "$ARM" = on ]; then EX="MEMRA_PRIME_INVARIANT=1 MEMRA_PRIME_GRAIN=32"; else EX=""; fi
    S=$(date +%s.%N)
    env $EX MEMRA_PRIME_CHUNK=64 $P "$M" chunkinv --prompt-a "@$D/prompt-long.txt" \
        --chunks 64 --steps 0 >/dev/null 2>&1
    E=$(date +%s.%N)
    echo "{\"phase\":\"E\",\"rep\":$rep,\"arm\":\"$ARM\",\"prime_s\":$(echo "$E-$S"|bc)}" \
      | tee -a "$L/E-perf.jsonl"
  done
done
echo "### done $(date -Is)"
