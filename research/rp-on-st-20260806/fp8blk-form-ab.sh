#!/usr/bin/env bash
# lane/rp-on-st — MMA FORM A/B on the per-block FP8 MMQ prefill tile (cu/mmq_fp8_blk.cu).
#   B (blksc, DEFAULT) : kind::mxf8f6f4.block_scale.scale_vec::1X @ ue8m0 identity 0x7F7F7F7F
#   P (plain, ROLLBACK): kind::f8f6f4                      (MEMRA_MMQ_FP8BLK_PLAIN=1 build)
# Two SEPARATE binaries (build-time seam), so the arms are ADJACENT and the order ALTERNATES per
# pair — the pairsweep law from research/fp8blk-20260805/pairsweep.sh: a third arm between the two
# being compared breaks single-clock-window interleaving.
# In-process MEMRA_PP_REPS=3, per-process MEDIAN reported; cross-process spread is what N measures.
set -u
cd /home/avifenesh/projects/wt-rpst
R=research/rp-on-st-20260806
CK=/data/ai-ml/hf-models/qwen36-27b-blk128fp8
P=research/e2e/prompts/pp512.txt
B=./target/release/run-gen
PL=/tmp/run-gen-plainform
G() { nvidia-smi --query-gpu=memory.used,temperature.gpu,clocks.sm,utilization.gpu --format=csv,noheader; }
: > "$R/fp8blk-form-ab.log"
say() { echo "$*" | tee -a "$R/fp8blk-form-ab.log"; }
say "# binaries: blksc=$(md5sum $B | cut -d' ' -f1) plain=$(md5sum $PL | cut -d' ' -f1)"
one() { # one <bin> <tag>
  local bin=$1 tag=$2
  env MEMRA_FP8_MMQ_STATS=1 MEMRA_PP_ONLY=1 MEMRA_PP_REPS=3 MEMRA_PROMPT_FILE="$P" \
      timeout 2400 "$bin" "$CK" > "$R/formab-$tag.log" 2>&1
  say "$tag rc=$? | $(grep -a 'pp-only MEDIAN' "$R/formab-$tag.log" | head -1) | reps $(grep -a -oP 'pp-only rep \d+: [\d.]+s = \K[\d.]+' "$R/formab-$tag.log" | tr '\n' ' ')| disp $(grep -a -oP 'fp8-mmq dispatches: \K\d+' "$R/formab-$tag.log")"
}
for p in 1 2 3 4 5 6; do
  say "== pair $p  gpu $(G)"
  if (( p % 2 )); then one "$B" "B-p$p"; one "$PL" "P-p$p"; else one "$PL" "P-p$p"; one "$B" "B-p$p"; fi
done
say "final gpu $(G)"
python3 - "$R" > "$R/fp8blk-form-ab-summary.txt" 2>&1 <<'PY'
import pathlib, re, statistics, sys
r = pathlib.Path(sys.argv[1]); med = {}
for arm, name in (("B","blksc"), ("P","plain")):
    v=[]
    for p in range(1,7):
        f = r / f"formab-{arm}-p{p}.log"
        if not f.exists(): continue
        m = re.search(r"pp-only MEDIAN: .*= ([\d.]+) tok/s", f.read_text(errors="replace"))
        if m: v.append(float(m.group(1)))
    med[arm]=v
    print(f"{name:6s} n={len(v)} medians={v} median={statistics.median(v):.1f} min={min(v):.1f} max={max(v):.1f}")
b,pl = med["B"], med["P"]
print(f"ratio blksc/plain = {statistics.median(b)/statistics.median(pl):.4f}")
print(f"overlap: min(blksc)={min(b):.1f} vs max(plain)={max(pl):.1f} -> {'OVERLAP' if min(b) <= max(pl) else 'DISJOINT'}")
PY
cat "$R/fp8blk-form-ab-summary.txt"
