#!/usr/bin/env bash
# lane/fp8-blk128-decode -- POST-FLIP battery. The default for the native-resident per-block MMQ
# prefill route just changed from OFF to ON, so every claim measured with MEMRA_FP8_MMQ=1 must be
# re-measured NAKED on the flipped binary. A flip whose kernel silently does not dispatch looks
# EXACTLY like a flip that works (same numbers as before = the pre-flip default), which is why every
# arm below prints the dispatch ledger and why the seam arm is measured too.
#
# FOUR ARMS, all on the 27B block-128 checkpoint, ONE lock hold, ONE md5-pinned binary:
#   N naked            -> native e4m3 residency + per-block MMQ prefill (THE NEW DEFAULT)
#   S seam             -> MEMRA_FP8_MMQ=0: native residency kept, prefill reverts to dequant-per-call
#   A slab             -> MEMRA_ST_E4M3_BLK=0: the pre-lane floor (Q8_0 slab, Q8_0 MMQ)
#   R rollback         -> MEMRA_ST_E4M3=0: the shared seam, BOTH e4m3 classes back to the slab
# N must equal the old C-blkmmq arm, S must equal the old B-blkdeq arm, A must equal the old A-slab
# arm. Anything else means the flip changed something it was not supposed to.
set -u
cd /home/avifenesh/projects/wt-fp8blk
R=research/fp8blk-20260805
CK=/data/ai-ml/hf-models/qwen36-27b-blk128fp8
P=research/e2e/prompts/pp512.txt
md5sum target/release/run-gen target/release/kernel-check target/release/memra-server > "$R/BINARY-md5-postflip.txt"
G() { nvidia-smi --query-gpu=memory.used,temperature.gpu,clocks.sm --format=csv,noheader; }
nvidia-smi --query-compute-apps=pid,used_memory --format=csv > "$R/postflip-gpustate.txt"
: > "$R/postflip-driver.log"
say() { echo "$*" | tee -a "$R/postflip-driver.log"; }

# ---- phase 1: pp512, 4 arms INTERLEAVED inside each rep (one clock window per rep)
for r in 1 2 3; do
  say "== rep $r  gpu $(G)"
  for arm in N S A; do
    case $arm in
      N) ENVV=(MEMRA_NOOP=1) ;;
      S) ENVV=(MEMRA_FP8_MMQ=0) ;;
      A) ENVV=(MEMRA_ST_E4M3_BLK=0) ;;
    esac
    env "${ENVV[@]}" MEMRA_FP8_MMQ_STATS=1 MEMRA_PP_ONLY=1 MEMRA_PP_REPS=3 MEMRA_PROMPT_FILE="$P" \
        timeout 2400 target/release/run-gen "$CK" > "$R/postflip-pp-$arm-r$r.log" 2>&1
    say "pp $arm r$r rc=$? | $(grep -a 'pp-only MEDIAN' "$R/postflip-pp-$arm-r$r.log" | head -1) | $(grep -a 'fp8-mmq dispatches' "$R/postflip-pp-$arm-r$r.log" | head -1)"
  done
done

# ---- phase 2: decode + residency census, naked vs the shared rollback seam. Decode must be
# UNCHANGED by this flip (the route is prefill-only) -- that is the regression this phase guards.
for r in 1 2 3; do
  for arm in N R; do
    case $arm in
      N) ENVV=(MEMRA_NOOP=1) ;;
      R) ENVV=(MEMRA_ST_E4M3=0) ;;
    esac
    env "${ENVV[@]}" MEMRA_FP8_MMQ_STATS=1 MEMRA_NGEN=128 MEMRA_PROMPT_FILE="$P" MEMRA_RESIDENCY_CENSUS=1 \
        timeout 2400 target/release/run-gen "$CK" > "$R/postflip-dec-$arm-r$r.log" 2>&1
    # NOTE the pattern: the ST-dir branch prints "generated N tokens in Xs = Y tok/s (ST greedy
    # decode)", NOT the GGUF branch's "(gen-only; ...)" wording. The first run of this script grepped
    # 'gen-only' and printed an EMPTY field for every decode arm while the logs held the numbers — a
    # blank summary line that looks like a failed run and is not one.
    say "dec $arm r$r rc=$? | $(grep -a -E 'tok/s \(ST greedy decode\)|gen-only' "$R/postflip-dec-$arm-r$r.log" | head -1)"
  done
done
say "census naked: $(grep -a -E 'Q8_0:|F8_E4M3_BLK:' "$R/postflip-dec-N-r1.log" | tr '\n' ' ')"
say "census rollback: $(grep -a -E 'Q8_0:|F8_E4M3_BLK:' "$R/postflip-dec-R-r1.log" | tr '\n' ' ')"

# ---- phase 3: prefill-path exactness on the NAKED default (the flipped arm is now what ships, so
# the shipped path is what must carry the exactness receipt). A-slab is re-run as the denominator on
# this same binary rather than reused from the pre-flip run.
for arm in N A; do
  case $arm in
    N) ENVV=(MEMRA_NOOP=1) ;;
    A) ENVV=(MEMRA_ST_E4M3_BLK=0) ;;
  esac
  env "${ENVV[@]}" MEMRA_FP8_MMQ_STATS=1 MEMRA_PP_ONLY=1 MEMRA_PP_REPS=1 MEMRA_PP_WARMUP=1 \
      MEMRA_PP_NLL=1 MEMRA_PP_LOGITS="$R/postflip-ppx-$arm.f32" MEMRA_PROMPT_FILE="$P" \
      timeout 2400 target/release/run-gen "$CK" > "$R/postflip-ppx-$arm.log" 2>&1
  say "ppx $arm rc=$? | $(grep -a 'fp8-mmq dispatches' "$R/postflip-ppx-$arm.log" | head -1)"
  say "$(grep -a 'prefill-path EXACTNESS' "$R/postflip-ppx-$arm.log" | head -1)"
done
say "== cross-arm prime-logit drift (A = floor denominator, same binary) =="
# HARNESS BUG, found and fixed on the first run of this script: `python3 - "$R" | tee -a log <<'PY'`
# attaches the heredoc to the LAST command in the pipeline (tee), not to python. Result: tee appended
# the python SOURCE to the driver log and python read EOF on stdin, so the comparator silently never
# ran and the log looked like it had output. Redirect to a file, then tee the file. (Same shape as
# the standing rule against `run-* 2>&1 | parser`: never let the pipe be the thing that eats the
# evidence.)
python3 - "$R" > "$R/postflip-ppx-cmp.txt" 2>&1 <<'PY'
import math, pathlib, struct, sys
r = pathlib.Path(sys.argv[1])
def rd(p):
    b = p.read_bytes(); return list(struct.unpack("<%df" % (len(b)//4), b))
ref = rd(r / "postflip-ppx-A.f32")
rms_ref = math.sqrt(sum(v*v for v in ref)/len(ref))
order = sorted(range(len(ref)), key=lambda i: -ref[i])[:10]
print(f"n={len(ref)}  rms(A-slab)={rms_ref:.4f}  argmax(A)={order[0]}")
for arm in ("A", "N"):
    g = rd(r / f"postflip-ppx-{arm}.f32")
    mx = max(abs(a-b) for a, b in zip(ref, g))
    rms = math.sqrt(sum((a-b)**2 for a, b in zip(ref, g))/len(g))
    go = sorted(range(len(g)), key=lambda i: -g[i])[:10]
    bits = sum(1 for a, b in zip(ref, g) if a != b)
    print(f"{arm}: max_abs={mx:.6e}  rms_abs={rms:.6e}  rms_rel={rms/rms_ref:.3e}  "
          f"bitdiff={bits}/{len(g)}  argmax={go[0]}  top10_same={go == order}")
PY
cat "$R/postflip-ppx-cmp.txt" | tee -a "$R/postflip-driver.log"

# ---- phase 4: run-spec K=1..8 self-consistency on the NAKED default (primes via prime_cache, so it
# exercises the flipped route on every draft+verify step).
for K in 1 2 3 4 5 6 7 8; do
  env MEMRA_NOOP=1 MEMRA_SPEC_K=$K MEMRA_NGEN=64 MEMRA_PROMPT_FILE="$P" \
      timeout 2400 target/release/run-spec "$CK" > "$R/postflip-runspec-K$K.log" 2>&1
  say "run-spec K=$K rc=$? | $(grep -a -E 'self-consistency|SELF-CONSISTENCY' "$R/postflip-runspec-K$K.log" | head -1)"
done
say "final gpu $(G)"
