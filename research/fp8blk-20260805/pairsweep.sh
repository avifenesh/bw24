#!/usr/bin/env bash
# lane/fp8-blk128-decode -- TIGHT N-vs-A pp512 pair sweep, N=6 ADJACENT pairs.
#
# WHY: postflip.sh's phase 1 interleaves THREE arms per rep (N,S,A), so the A arm always runs third
# -- ~4 minutes of load+prefill after N. Pre-flip's 3-arm run had the same structure and produced
# non-overlapping distributions, but the post-flip run did NOT (N median 1549.7 vs A 1539.4, and
# N r2 1517.4 < A r3 1539.4 -- overlap), and A r1 came in at 1377.9, an 10.5% outlier below its own
# other two reps. Both facts point at position/thermal drift inside the rep rather than at the
# kernels, so the flip claim gets its own two-arm sweep where N and A are ADJACENT and the order
# ALTERNATES (N,A then A,N) -- the standing law is interleaving inside one clock window, and a third
# arm sitting between the two being compared breaks that for the pair.
#
# In-process reps are 3 with the MEDIAN reported, and the per-rep values are in the logs: the first
# in-process rep is consistently the fastest (cold clocks boosting) and the second the slowest, so
# the median is the right per-process statistic and the cross-process spread is what N=6 measures.
set -u
cd /home/avifenesh/projects/wt-fp8blk
R=research/fp8blk-20260805
CK=/data/ai-ml/hf-models/qwen36-27b-blk128fp8
P=research/e2e/prompts/pp512.txt
md5sum target/release/run-gen > "$R/BINARY-md5-pairsweep.txt"
G() { nvidia-smi --query-gpu=memory.used,temperature.gpu,clocks.sm --format=csv,noheader; }
: > "$R/pairsweep-driver.log"
say() { echo "$*" | tee -a "$R/pairsweep-driver.log"; }

one() { # one <arm> <tag>
  local arm=$1 tag=$2
  case $arm in
    N) ENVV=(MEMRA_NOOP=1) ;;
    A) ENVV=(MEMRA_ST_E4M3_BLK=0) ;;
  esac
  env "${ENVV[@]}" MEMRA_FP8_MMQ_STATS=1 MEMRA_PP_ONLY=1 MEMRA_PP_REPS=3 MEMRA_PROMPT_FILE="$P" \
      timeout 2400 target/release/run-gen "$CK" > "$R/pair-$tag.log" 2>&1
  say "$tag rc=$? | $(grep -a 'pp-only MEDIAN' "$R/pair-$tag.log" | head -1) | reps $(grep -a -oP 'pp-only rep \d+: [\d.]+s = \K[\d.]+' "$R/pair-$tag.log" | tr '\n' ' ')| hits $(grep -a -oP 'fp8-mmq dispatches: \K\d+' "$R/pair-$tag.log")"
}

for p in 1 2 3 4 5 6; do
  say "== pair $p  gpu $(G)"
  if (( p % 2 )); then one N "N-p$p"; one A "A-p$p"; else one A "A-p$p"; one N "N-p$p"; fi
done
say "final gpu $(G)"
# Heredoc goes to python, NOT to a tee at the end of a pipe -- `python3 - | tee <<EOF` attaches the
# heredoc to TEE, so python reads EOF and the analysis silently never runs while tee appends the
# python SOURCE to the log. That happened on this script's first run (and on postflip.sh's); the
# receipt is the log itself. Redirect to a file, then cat the file.
python3 - "$R" > "$R/pairsweep-summary.txt" 2>&1 <<'PY'
import pathlib, re, statistics, sys
r = pathlib.Path(sys.argv[1])
med = {}
for arm in "NA":
    v = []
    for p in range(1, 7):
        t = (r / f"pair-{arm}-p{p}.log").read_text(errors="replace")
        m = re.search(r"pp-only MEDIAN: .*= ([\d.]+) tok/s", t)
        if m: v.append(float(m.group(1)))
    med[arm] = v
    print(f"{arm}: n={len(v)} medians={v} median={statistics.median(v):.1f} min={min(v):.1f} max={max(v):.1f}")
n, a = med["N"], med["A"]
print(f"ratio of medians N/A = {statistics.median(n)/statistics.median(a):.4f}")
print(f"overlap: min(N)={min(n):.1f} vs max(A)={max(a):.1f} -> {'OVERLAP' if min(n) <= max(a) else 'DISJOINT'}")
wins = sum(1 for x, y in zip(n, a) if x > y)
print(f"per-pair wins for N: {wins}/{min(len(n), len(a))}  pairwise deltas="
      f"{[round(x - y, 1) for x, y in zip(n, a)]}")
PY
cat "$R/pairsweep-summary.txt" | tee -a "$R/pairsweep-driver.log"
