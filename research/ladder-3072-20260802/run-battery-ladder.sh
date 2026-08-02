#!/bin/bash
# lane/ladder-3072: the NEW-NUMERIC-CONFIG battery for the sp8->sp64 rung move 3072 -> 512.
# The ladder changes WHICH split count runs (combine fold order changes per depth band), so:
# kernel-check full (incl. FA-DEEP pin, now with 511/512/513 rung-straddle depths),
# run-gen argmax x3 model classes, run-spec K=1..8 (q35), decode-batch config+strict
# (bucketed rows group by ladder value), graph-decode x3 (P=500 N=96 crosses the NEW rung
# inside the window; P=3000/6000 cover the sp64 band), graph-session.
set -u
W=/home/avifenesh/projects/wt-ladder-3072
R=$W/research/ladder-3072-20260802
P=$W/research/depth-decode-20260802
declare -A GGUF=(
  [kat]=/data/ai-ml/hf-models/kat-coder-v25-dev-gguf/Kwaipilot_KAT-Coder-V2.5-Dev-IQ4_XS.gguf
  [q35]=/data/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
  [o35b]=/data/ai-ml/hf-models/ornith-1.0-35b-gguf/ornith-1.0-35b-Q4_K_M.gguf
)
declare -A DRAFT=(
  [q35]=/data/ai-ml/hf-models/qwen36-35b-moe/draft-35b-owntrim-nvfp4head-q4blk.gguf
)
FAILS=0
echo "=== ladder-3072 battery $(date -u +%FT%TZ) git=$(git -C $W rev-parse --short HEAD) ==="

# 0. kernel-check FULL (incl. FA-DEEP pin at the new rung straddle)
log=$R/kernel-check-full.log
flock /tmp/gpu5090.lock timeout 3600 $W/target/release/kernel-check "${GGUF[q35]}" > "$log" 2>&1
fails_kc=$(grep -c "FAIL" "$log"); oks=$(grep -c " OK" "$log")
if [ "$fails_kc" -eq 0 ] && [ "$oks" -gt 100 ]; then echo "kernel-check: ALL GREEN ($oks OK) OK"
else echo "kernel-check: FAIL (fails=$fails_kc oks=$oks, see $log)"; FAILS=$((FAILS+1)); fi

# 1. run-gen argmax x3 classes — deep-region prompt (d4096) + a NEW-rung-region prompt (d1024)
for m in kat q35 o35b; do
  log=$R/gate-argmax-$m-d4096.log
  MEMRA_NGEN=128 MEMRA_PROMPT_FILE=$P/depth-4096-$m.txt \
    flock /tmp/gpu5090.lock timeout 1800 $W/target/release/run-gen "${GGUF[$m]}" > "$log" 2>&1
  n=$(grep -c "argmax.*MATCH" "$log"); bad=$(grep -c "MISMATCH" "$log")
  if [ "$n" -ge 1 ] && [ "$bad" -eq 0 ]; then echo "argmax $m d4096: MATCH ($n) OK"
  else echo "argmax $m d4096: FAIL (m=$n mm=$bad, $log)"; FAILS=$((FAILS+1)); fi
done
for m in kat q35; do   # d1024 = inside the NEW sp64 band (was sp8)
  log=$R/gate-argmax-$m-d1024.log
  MEMRA_NGEN=128 MEMRA_PROMPT_FILE=$R/depth-1024-$m.txt \
    flock /tmp/gpu5090.lock timeout 1800 $W/target/release/run-gen "${GGUF[$m]}" > "$log" 2>&1
  n=$(grep -c "argmax.*MATCH" "$log"); bad=$(grep -c "MISMATCH" "$log")
  if [ "$n" -ge 1 ] && [ "$bad" -eq 0 ]; then echo "argmax $m d1024: MATCH ($n) OK"
  else echo "argmax $m d1024: FAIL (m=$n mm=$bad, $log)"; FAILS=$((FAILS+1)); fi
done

# 2. run-spec K=1..8 self-consistency on q35 (own-trim drafter). Prompt d2048 (new sp64 band).
log=$R/gate-spec-q35.log
MEMRA_MTP_DRAFT="${DRAFT[q35]}" MEMRA_NGEN=64 MEMRA_PROMPT_FILE=$P/depth-2048-q35.txt \
  flock /tmp/gpu5090.lock timeout 3600 $W/target/release/run-spec "${GGUF[q35]}" > "$log" 2>&1
ok=$(grep -cE "self-consistency.*(PASS|OK)|K=[0-9]+.*(PASS|OK)" "$log")
bad=$(grep -ciE "FAIL|mismatch" "$log")
if [ "$ok" -ge 8 ] && [ "$bad" -eq 0 ]; then echo "run-spec q35: K-battery PASS ($ok) OK"
else echo "run-spec q35: check (pass=$ok fail=$bad, $log)"; [ "$bad" -gt 0 ] && FAILS=$((FAILS+1)); fi

# 3. decode-batch gates on q35 (bucketed rows group consecutive rows by ladder value —
#    the rung move changes the grouping boundaries) — config + strict-equalized.
log=$R/gate-decode-batch-config.log
flock /tmp/gpu5090.lock timeout 3600 $W/target/release/decode-batch-gate "${GGUF[q35]}" \
  --steps 32 --batch 8 --mode config > "$log" 2>&1
grep -q "ALL GREEN" "$log" && echo "decode-batch config: GREEN OK" \
  || { echo "decode-batch config: FAIL ($log)"; FAILS=$((FAILS+1)); }
log=$R/gate-decode-batch-strict.log
MEMRA_MMVQ=0 MEMRA_NO_FUSE_NORMQ=1 flock /tmp/gpu5090.lock timeout 3600 \
  $W/target/release/decode-batch-gate "${GGUF[q35]}" --steps 16 --batch 4 --mode strict > "$log" 2>&1
grep -q "ALL GREEN" "$log" && echo "decode-batch strict: GREEN OK" \
  || { echo "decode-batch strict: FAIL ($log)"; FAILS=$((FAILS+1)); }

# 4. graph gates: dc/replay paths key n_splits on bucket_max — the rung move changes the
#    capture split count per bucket. P=500 N=96 crosses the NEW 512 rung inside a segment
#    (the exact analogue of the old kat P=3000 crossing 3072); P=3000/6000 = sp64 band.
for spec in "q35 6000 160" "kat 3000 160" "q35 500 96" "kat 400 160"; do
  set -- $spec; m=$1; pp=$2; nn=$3
  log=$R/gate-graph-decode-$m-p$pp.log
  flock /tmp/gpu5090.lock timeout 3600 $W/target/release/graph-decode-gate "${GGUF[$m]}" $pp $nn > "$log" 2>&1
  grep -qiE "PASS|IDENTICAL|MATCH" "$log" && ! grep -qiE "FAIL|MISMATCH" "$log" \
    && echo "graph-decode $m P=$pp N=$nn: PASS OK" \
    || { echo "graph-decode $m P=$pp N=$nn: FAIL ($log)"; FAILS=$((FAILS+1)); }
done
log=$R/gate-graph-session.log
flock /tmp/gpu5090.lock timeout 3600 $W/target/release/graph-session-gate "${GGUF[q35]}" --steps 96 > "$log" 2>&1
grep -qiE "PASS|IDENTICAL|MATCH" "$log" && ! grep -qiE "FAIL|MISMATCH" "$log" \
  && echo "graph-session q35: PASS OK" \
  || { echo "graph-session q35: FAIL ($log)"; FAILS=$((FAILS+1)); }

echo "=== ladder battery done: FAILS=$FAILS ==="
exit $((FAILS > 0))
