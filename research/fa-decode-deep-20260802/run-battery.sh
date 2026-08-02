#!/bin/bash
# fa-decode-deep: the exactness battery on the class models with the deep twins DEFAULT-ON.
# kernel-check ran separately (kernel-check-full.log, ALL GREEN incl. the deep bit pin).
# Here: run-gen argmax MATCH x3 models (deep region prompt), run-spec K=1..8
# self-consistency x3 models (own-trim drafters), decode-batch gates on q35
# (config + strict-equalized), graph-decode + graph-session gates (capture semantics).
# Every GPU run under flock /tmp/gpu5090.lock (shared rig, holds released between runs).
set -u
W=/home/avifenesh/projects/wt-fa-decode-deep
R=$W/research/fa-decode-deep-20260802
P=$W/research/depth-decode-20260802     # the depth prompts (same document prefixes)
declare -A GGUF=(
  [kat]=/data/ai-ml/hf-models/kat-coder-v25-dev-gguf/Kwaipilot_KAT-Coder-V2.5-Dev-IQ4_XS.gguf
  [q35]=/data/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
  [o35b]=/data/ai-ml/hf-models/ornith-1.0-35b-gguf/ornith-1.0-35b-Q4_K_M.gguf
)
declare -A DRAFT=(
  [kat]=/data/ai-ml/hf-models/kat-coder-v25-dev-gguf/draft-katcoder-owntrim-nvfp4head-q4blk.gguf
  [q35]=/data/ai-ml/hf-models/qwen36-35b-moe/draft-35b-owntrim-nvfp4head-q4blk.gguf
  [o35b]=/data/ai-ml/hf-models/ornith-1.0-35b-gguf/draft-ornith35b-owntrim-nvfp4head-q4blk.gguf
)
FAILS=0

echo "=== fa-deep battery $(date -u +%FT%TZ) git=$(git -C $W rev-parse --short HEAD) ==="

# 1. run-gen argmax gate, deep-region prompt (d4096 -> decode window 4096..4224, deep on)
for m in kat q35 o35b; do
  log=$R/gate-argmax-$m.log
  MEMRA_NGEN=128 MEMRA_PROMPT_FILE=$P/depth-4096-$m.txt \
    flock /tmp/gpu5090.lock timeout 1800 $W/target/release/run-gen "${GGUF[$m]}" > "$log" 2>&1
  n=$(grep -c "argmax.*MATCH" "$log")
  bad=$(grep -c "MISMATCH" "$log")
  if [ "$n" -ge 1 ] && [ "$bad" -eq 0 ]; then echo "argmax $m: MATCH ($n gate lines) OK"
  else echo "argmax $m: FAIL (match_lines=$n mismatch_lines=$bad, see $log)"; FAILS=$((FAILS+1)); fi
done

# 2. run-spec K=1..8 self-consistency (binary loops K=1..8; token-identical to plain greedy)
for m in kat q35 o35b; do
  log=$R/gate-spec-$m.log
  MEMRA_MTP_DRAFT="${DRAFT[$m]}" MEMRA_NGEN=64 MEMRA_PROMPT_FILE=$P/depth-2048-$m.txt \
    flock /tmp/gpu5090.lock timeout 3600 $W/target/release/run-spec "${GGUF[$m]}" > "$log" 2>&1
  ok=$(grep -cE "self-consistency.*(PASS|OK)|K=[0-9]+.*(PASS|OK)" "$log")
  bad=$(grep -ciE "FAIL|mismatch" "$log")
  if [ "$ok" -ge 8 ] && [ "$bad" -eq 0 ]; then echo "run-spec $m: K-battery PASS ($ok pass lines) OK"
  else echo "run-spec $m: check (pass_lines=$ok fail_lines=$bad, see $log)"; [ "$bad" -gt 0 ] && FAILS=$((FAILS+1)); fi
done

# 3. decode-batch gates on q35 (the batched tick shares the fa class; seqs twin unchanged
#    but its per-seq oracle loop rides the deep default) — config mode + strict equalized.
# (verdict = the binary's own "ALL GREEN" summary line; per-gate diagnostics legitimately
#  contain the word FAIL in threshold descriptions, so no free-text FAIL grep here)
log=$R/gate-decode-batch-config.log
flock /tmp/gpu5090.lock timeout 3600 $W/target/release/decode-batch-gate "${GGUF[q35]}" \
  --steps 32 --batch 8 --mode config > "$log" 2>&1
grep -q "ALL GREEN" "$log" \
  && echo "decode-batch config: GREEN OK" \
  || { echo "decode-batch config: FAIL (see $log)"; FAILS=$((FAILS+1)); }
log=$R/gate-decode-batch-strict.log
MEMRA_MMVQ=0 MEMRA_NO_FUSE_NORMQ=1 flock /tmp/gpu5090.lock timeout 3600 \
  $W/target/release/decode-batch-gate "${GGUF[q35]}" --steps 16 --batch 4 --mode strict > "$log" 2>&1
grep -q "ALL GREEN" "$log" \
  && echo "decode-batch strict: GREEN OK" \
  || { echo "decode-batch strict: FAIL (see $log)"; FAILS=$((FAILS+1)); }

# 4. graph gates (fa class boundaries are capture-relevant): graph-decode across buckets
#    incl. the deep region, graph-session step-lift identity.
for spec in "q35 6000 160" "kat 3000 160" "q35 500 96"; do
  set -- $spec; m=$1; pp=$2; nn=$3
  log=$R/gate-graph-decode-$m-p$pp.log
  flock /tmp/gpu5090.lock timeout 3600 $W/target/release/graph-decode-gate "${GGUF[$m]}" $pp $nn > "$log" 2>&1
  grep -qiE "PASS|IDENTICAL|MATCH" "$log" && ! grep -qiE "FAIL|MISMATCH" "$log" \
    && echo "graph-decode $m P=$pp N=$nn: PASS OK" \
    || { echo "graph-decode $m P=$pp N=$nn: FAIL (see $log)"; FAILS=$((FAILS+1)); }
done
log=$R/gate-graph-session.log
flock /tmp/gpu5090.lock timeout 3600 $W/target/release/graph-session-gate "${GGUF[q35]}" --steps 96 > "$log" 2>&1
grep -qiE "PASS|IDENTICAL|MATCH" "$log" && ! grep -qiE "FAIL|MISMATCH" "$log" \
  && echo "graph-session q35: PASS OK" \
  || { echo "graph-session q35: FAIL (see $log)"; FAILS=$((FAILS+1)); }

echo "=== battery done: FAILS=$FAILS ==="
exit $((FAILS > 0))
