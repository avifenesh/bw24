#!/usr/bin/env bash
# fp8-ship item B, phase 1 — OFFICIAL Qwen3.6-27B-FP8 checkpoint load gates on the vast 2x5090.
# First official-artifact gates for the block-128 loader:
#   default arm  = CPU dequant (f8_deq_f32 + host f32_to_q8_0 re-encode)
#   blkgpu arm   = MEMRA_FP8_BLK_GPU=1 (ARM B' device dequant, cu/fp8_blk_dequant.cu)
# Contract under test: BYTE-IDENTICAL residents => identical argmax gates, identical greedy
# token streams, bit-identical prefill logit vectors. Load wall = whole-process wall
# (same protocol as research/fp8st-20260803/armb/loadtime.log: engine init + tokenizer +
# gates + 32-tok greedy gen included in BOTH arms), interleaved pairs, N=3 on pp512.
set -uo pipefail
cd /root/memra
OUT=research/fp8ship-20260804/official
mkdir -p "$OUT"
CKPT=/root/models/qwen36-27b-fp8-official
BIN=target/release/run-gen
P512=research/e2e/prompts/pp512.txt
P1=research/e2e/prompts/p1-code-short.txt
LOCK=/tmp/memra-bench.lock
DLOG=$OUT/phase1-driver.log
log(){ echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$DLOG"; }
snap(){ nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw,memory.used --format=csv,noheader -i 0; }

run_arm(){ # arm rep tag promptfile [extra env...]
  local arm=$1 rep=$2 tag=$3 pf=$4; shift 4
  local logf=$OUT/load-$tag-$arm-r$rep.log
  log "$arm $tag rep$rep pre: $(snap)"
  local t0 t1 rc wall
  t0=$(date +%s.%N)
  flock "$LOCK" env CUDA_VISIBLE_DEVICES=0 MEMRA_NGEN=32 MEMRA_PROMPT_FILE="$pf" \
    MEMRA_PREFILL_LOGITS=$OUT/logits-$tag-$arm-r$rep.bin "$@" \
    timeout 3600 $BIN "$CKPT" > "$logf" 2>&1
  rc=$?
  t1=$(date +%s.%N)
  wall=$(echo "$t1 $t0" | awk '{printf "%.3f", $1-$2}')
  echo "$tag $arm rep$rep wall_s $wall rc=$rc" >> $OUT/loadtime.log
  log "$arm $tag rep$rep post: wall=${wall}s rc=$rc | $(grep -aE 'argmax=.*(MATCH|MISMATCH|FLIP)' "$logf" | head -2 | tr '\n' ' ; ') | oom=$(grep -ac 'out of memory' "$logf")"
}

log "== PHASE 1: official block-128 FP8 ckpt ($CKPT) load gates, default vs MEMRA_FP8_BLK_GPU=1, interleaved pairs, GPU0 =="
for rep in 1 2 3; do
  run_arm default "$rep" pp512 "$P512"
  run_arm blkgpu  "$rep" pp512 "$P512" MEMRA_FP8_BLK_GPU=1
done
run_arm default 1 p1 "$P1"
run_arm blkgpu  1 p1 "$P1" MEMRA_FP8_BLK_GPU=1

log "== BIT-IDENTITY: default vs blkgpu (token streams + argmax lines + prefill logit vectors) =="
check_pair(){ # tag rep
  local tag=$1 rep=$2
  local a=$OUT/load-$tag-default-r$rep.log b=$OUT/load-$tag-blkgpu-r$rep.log
  grep -aE '^(prompt tokens:|tokens:|OUTPUT TEXT:)|argmax=' "$a" > /tmp/ident-a.cmp
  grep -aE '^(prompt tokens:|tokens:|OUTPUT TEXT:)|argmax=' "$b" > /tmp/ident-b.cmp
  if diff -q /tmp/ident-a.cmp /tmp/ident-b.cmp >/dev/null; then
    log "IDENT $tag r$rep: token stream + argmax lines IDENTICAL"
  else
    log "DIVERGE $tag r$rep: token/argmax mismatch:"
    diff /tmp/ident-a.cmp /tmp/ident-b.cmp | head -20 | tee -a "$DLOG"
  fi
  if cmp -s $OUT/logits-$tag-default-r$rep.bin $OUT/logits-$tag-blkgpu-r$rep.bin; then
    log "IDENT $tag r$rep: prefill logit vectors BIT-IDENTICAL ($(stat -c%s $OUT/logits-$tag-default-r$rep.bin) bytes)"
  else
    log "DIVERGE $tag r$rep: prefill logits differ: $(cmp $OUT/logits-$tag-default-r$rep.bin $OUT/logits-$tag-blkgpu-r$rep.bin 2>&1 | head -1)"
  fi
}
for rep in 1 2 3; do check_pair pp512 "$rep"; done
check_pair p1 1
log "== within-arm rep stability (default r1 vs r2 vs r3 token lines) =="
for arm in default blkgpu; do
  if diff <(grep -a '^tokens:' $OUT/load-pp512-$arm-r1.log) <(grep -a '^tokens:' $OUT/load-pp512-$arm-r2.log) >/dev/null \
  && diff <(grep -a '^tokens:' $OUT/load-pp512-$arm-r2.log) <(grep -a '^tokens:' $OUT/load-pp512-$arm-r3.log) >/dev/null; then
    log "STABLE $arm: identical token stream across 3 reps"
  else
    log "UNSTABLE $arm: token streams differ across reps"
  fi
done
log "PHASE 1 DONE"
