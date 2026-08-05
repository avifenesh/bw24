#!/usr/bin/env bash
# lane/fp8-blk128-decode -- the MMQ PREFILL arm's own exactness cells, on the path that dispatches.
#
# WHY THIS SCRIPT EXISTS (the correction). The first attempt at these cells (tf-C-blkmmq.log,
# mmq-decode-census.log) is INVALID and says so in its own receipt: both printed
#   fp8-mmq dispatches after prefill: 0  (hook entries=0 gate_off=0 no_operand=0 ...)
# and tf-C's numbers came out byte-for-byte equal to tf-B (2/128, nll 0.293590) -- because they
# measured the DEQUANT arm while claiming to measure the MMQ arm. Cause, traced in the code:
#   * `run-gen`'s verify-prefill gate calls `decode_step_t` -> `matmul_decode_exact`, which has NO
#     GEMM/MMQ arm AT ALL by design (decode-parity law: every token row must take the exact m=1
#     MMVQ program). try_fp8_gemm / try_fp8_blk_mmq / try_f16_gemm are hooked only on
#     `matmul` / `matmul_pre`. So `entries=0` there is CORRECT BEHAVIOR, not a wiring bug, and no
#     flag can make a prefill-GEMM kernel run on that path.
#   * `prime_cache` -- what MEMRA_PP_ONLY times, and what `generate` / `generate_spec` / serving
#     actually prime with -- runs projections through `matmul`/`matmul_group`, which do carry the
#     hooks (measured: 1984 entries, 832 dispatches on the 27B block-128 checkpoint).
# So the arm's exactness is measured on prime_cache. The instrument is MEMRA_PP_LOGITS (cross-arm
# drift on the prime logit row) + MEMRA_PP_NLL (teacher-forced prefill quality).
#
# NO TAPE ASYMMETRY, BY CONSTRUCTION: MEMRA_PP_NLL's tape is the PROMPT itself -- position i's
# logits score prompt token i+1 -- so both arms are scored on the same externally-given sequence
# and neither can win by reproducing its own output. The decode battery needed a reverse-tape
# control (tfrev-*) precisely because its tape was one arm's greedy output; this quantity needs none.
#
# THREE ARMS, all on the block-128 27B, one lock hold, one md5-pinned binary:
#   A slab   MEMRA_ST_E4M3_BLK=0        -> Q8_0 slab resident, Q8_0 MMQ prefill (ARM B', the floor)
#   B blkdeq naked default              -> e4m3 resident, dequant-per-call to a Q8_0 transient
#                                          (prefill arithmetic is bit-for-bit the floor's)
#   C blkmmq MEMRA_FP8_MMQ=1            -> e4m3 resident, per-block f8f6f4 MMA (DIFFERENT arithmetic)
# A vs B must be IDENTICAL (same bits by construction -- the control that proves the harness can
# detect nothing where there is nothing). C vs A is the real question: branch-(b), so the bar is
# disagreement count + NLL, not bit-identity.
set -u
cd /home/avifenesh/projects/wt-fp8blk
R=research/fp8blk-20260805
CK=/data/ai-ml/hf-models/qwen36-27b-blk128fp8
P=research/e2e/prompts/pp512.txt
md5sum target/release/run-gen > "$R/BINARY-md5-mmqexact.txt"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv > "$R/mmqexact-gpustate.txt"
nvidia-smi --query-gpu=memory.used,temperature.gpu,clocks.sm --format=csv,noheader >> "$R/mmqexact-gpustate.txt"

run() { # run <arm> <env...>
  local arm=$1; shift
  env "$@" MEMRA_PP_ONLY=1 MEMRA_PP_REPS=1 MEMRA_PP_WARMUP=1 MEMRA_PP_NLL=1 \
      MEMRA_PP_LOGITS="$R/ppx-$arm.f32" MEMRA_PROMPT_FILE="$P" \
      timeout 2400 target/release/run-gen "$CK" > "$R/ppx-$arm.log" 2>&1
  echo "$arm rc=$? | $(grep -a 'fp8-mmq dispatches' "$R/ppx-$arm.log" | head -1)"
  grep -a 'prefill-path EXACTNESS' "$R/ppx-$arm.log" | head -1
}

run A-slab   MEMRA_ST_E4M3_BLK=0
run B-blkdeq MEMRA_NOOP=1
run C-blkmmq MEMRA_FP8_MMQ=1

echo "== cross-arm prime-logit drift (A = floor denominator) =="
python3 - "$R" <<'PY'
import math, pathlib, struct, sys
r = pathlib.Path(sys.argv[1])
def rd(p):
    b = p.read_bytes()
    return list(struct.unpack("<%df" % (len(b) // 4), b))
ref = rd(r / "ppx-A-slab.f32")
rms_ref = math.sqrt(sum(v * v for v in ref) / len(ref))
order = sorted(range(len(ref)), key=lambda i: -ref[i])[:10]
print(f"n={len(ref)}  rms(A-slab)={rms_ref:.4f}  argmax(A)={order[0]}")
for arm in ("A-slab", "B-blkdeq", "C-blkmmq"):
    p = r / f"ppx-{arm}.f32"
    if not p.exists():
        print(f"{arm}: MISSING")
        continue
    g = rd(p)
    mx = max(abs(a - b) for a, b in zip(ref, g))
    rms = math.sqrt(sum((a - b) ** 2 for a, b in zip(ref, g)) / len(g))
    go = sorted(range(len(g)), key=lambda i: -g[i])[:10]
    bits = sum(1 for a, b in zip(ref, g) if a != b)
    print(f"{arm}: max_abs={mx:.6e}  rms_abs={rms:.6e}  rms_rel={rms/rms_ref:.3e}  "
          f"bitdiff={bits}/{len(g)}  argmax={go[0]}  top10_same={go == order}")
PY
