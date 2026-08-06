#!/usr/bin/env bash
# accprobe-form-ab.sh — re-prices the FP8-ST v3 gate's Q1 delta after the MMA form is equalized.
#
# WHAT WAS WRONG. accprobe-bench (crates/memra-engine/src/bin/accprobe_bench.rs) is the Q1
# instrument of the fp8-v3 gate: one kernel (cu/mmq_q8_0_f32acc.cu), the accumulator as its
# "ONE free variable", ratio = t_f32/t_s32, and delta_pp = 100*(ratio-1) published as "what s32
# accumulation is worth at fixed geometry — an UPPER BOUND on a v3" (+18.9 at m=512, +19.8 at
# m=6257; research/fp8v3-gate-20260805/). But the S32 arm issues
#   mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32                       = 16.06 cyc/warp-MMA
# and the F32 arm issued the PLAIN fp8 form
#   mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32      = 32.02 cyc/warp-MMA
# on sm_120a. So the accumulator was NOT the only free variable: the MMA issue interval moved
# with it, 2x. The published delta_pp is the SUM of an accumulator effect and a form effect.
#
# THE FIX (mmq_q8_0_f32acc.cu, this lane): the F32 arm now issues
#   mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3
#     .f32.ue8m0   with the identity scale 0x7F7F7F7F                     = 16.06 cyc/warp-MMA
# which computes the IDENTICAL e4m3xe4m3 product (proven bit-identical on the live per-block FP8
# tile: research/rp-on-st-20260806/pplogits-*.log, 0/993280 bytes differ). Now both arms issue at
# the same interval and the accumulator IS the one free variable. ACCPROBE_F32_PLAIN=1 rebuilds
# the old arm so the published receipts stay reproducible.
#
# THIS SCRIPT measures both binaries at the two published geometries and both ACCPROBE_DIST
# settings. Two independent controls make the comparison honest:
#   (1) the S32 arm is byte-identical code in both binaries — its time must agree across them, or
#       the clock regime drifted and the round is void;
#   (2) binaries are run as ADJACENT ALTERNATING pairs (pairsweep law,
#       research/fp8blk-20260805/pairsweep.sh) so no third arm sits inside a compared pair.
#
# accprobe-bench already interleaves its own f32/s32 arms per rep and reports medians, so the
# per-cell number is internally clock-fair; this driver only has to keep the two BINARIES adjacent.
set -uo pipefail
OUT=/home/avifenesh/projects/wt-rpst/research/rp-on-st-20260806
LOG=$OUT/accprobe-form-ab.log
B=/tmp/accprobe-blksc     # default form: block_scale @ ue8m0 identity
P=/tmp/accprobe-plain     # ACCPROBE_F32_PLAIN=1: the published arm

: > "$LOG"
{
  echo "=== accprobe MMA-form A/B  $(date -Is) ==="
  echo "host: $(hostname)  gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
  echo "blksc md5: $(md5sum $B | cut -d' ' -f1)"
  echo "plain md5: $(md5sum $P | cut -d' ' -f1)"
  echo "concurrent compute-apps at start:"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv
} >> "$LOG"

for dist in wide mid; do
  for geo in "512 9" "6257 5"; do
    set -- $geo; m=$1; reps=$2
    for pair in 1 2 3; do
      for arm in B P; do
        bin=$B; tag=blksc
        [ "$arm" = "P" ] && { bin=$P; tag=plain; }
        {
          echo "--- dist=$dist m=$m reps=$reps pair=$pair arm=$tag ---"
          nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader
        } >> "$LOG"
        ACCPROBE_DIST=$dist "$bin" "$m" "$reps" 27b >> "$LOG" 2>&1
        echo "rc=$? arm=$tag dist=$dist m=$m pair=$pair" >> "$LOG"
      done
    done
  done
done

echo "=== DONE $(date -Is) ===" >> "$LOG"

python3 - "$LOG" <<'PY' | tee "$OUT/accprobe-form-ab-summary.txt"
import re, sys, statistics as st
txt = open(sys.argv[1]).read().splitlines()
# cell key -> arm -> list of geomean ratios
cells = {}
shapes = {}
cur = None
for ln in txt:
    m = re.match(r'--- dist=(\w+) m=(\d+) reps=(\d+) pair=(\d+) arm=(\w+) ---', ln)
    if m:
        cur = (m.group(1), int(m.group(2)), m.group(5))
        continue
    m = re.search(r'GEOMEAN ratio \(f32/s32\) over \d+ shapes: ([\d.]+)x', ln)
    if m and cur:
        cells.setdefault(cur[:2], {}).setdefault(cur[2], []).append(float(m.group(1)))
        continue
    # per-shape rows: name ... f32_ms s32_ms ratio delta_pp f32_TF s32_TF
    m = re.match(r'(\S+ \S+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)x\s+([+\-][\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$', ln)
    if m and cur:
        shapes.setdefault(cur[:2], {}).setdefault(m.group(1), {}).setdefault(cur[2], []).append(
            (float(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(6)), float(m.group(7))))

print("== GEOMEAN ratio (f32acc / s32acc), median of 3 pairs ==")
print(f"{'dist':>5} {'m':>6} {'plain':>8} {'blksc':>8} {'plain_dpp':>10} {'blksc_dpp':>10} {'survives':>9}")
for k in sorted(cells):
    d = cells[k]
    if 'plain' not in d or 'blksc' not in d: continue
    p, b = st.median(d['plain']), st.median(d['blksc'])
    pdpp, bdpp = 100*(p-1), 100*(b-1)
    frac = f"{100*bdpp/pdpp:.0f}%" if abs(pdpp) > 1e-9 else "n/a"
    print(f"{k[0]:>5} {k[1]:>6} {p:>8.4f} {b:>8.4f} {pdpp:>+10.1f} {bdpp:>+10.1f} {frac:>9}")
    print(f"      {'':>6} spreads: plain {min(d['plain']):.4f}-{max(d['plain']):.4f}  blksc {min(d['blksc']):.4f}-{max(d['blksc']):.4f}")

print()
print("== S32-ARM CONTROL (identical code in both binaries; must agree) ==")
print(f"{'dist':>5} {'m':>6} {'shape':<26} {'s32_plain':>10} {'s32_blksc':>10} {'drift%':>7}")
for k in sorted(shapes):
    for sh, d in shapes[k].items():
        if 'plain' not in d or 'blksc' not in d: continue
        sp = st.median([x[1] for x in d['plain']]); sb = st.median([x[1] for x in d['blksc']])
        print(f"{k[0]:>5} {k[1]:>6} {sh:<26} {sp:>10.4f} {sb:>10.4f} {100*(sb/sp-1):>+7.2f}")

print()
print("== F32 ARM ABSOLUTE (the form's own effect) ==")
print(f"{'dist':>5} {'m':>6} {'shape':<26} {'f32_plain':>10} {'f32_blksc':>10} {'speedup':>8} {'TF_plain':>9} {'TF_blksc':>9}")
for k in sorted(shapes):
    for sh, d in shapes[k].items():
        if 'plain' not in d or 'blksc' not in d: continue
        fp = st.median([x[0] for x in d['plain']]); fb = st.median([x[0] for x in d['blksc']])
        tp = st.median([x[3] for x in d['plain']]); tb = st.median([x[3] for x in d['blksc']])
        print(f"{k[0]:>5} {k[1]:>6} {sh:<26} {fp:>10.4f} {fb:>10.4f} {fp/fb:>7.3f}x {tp:>9.1f} {tb:>9.1f}")
PY
