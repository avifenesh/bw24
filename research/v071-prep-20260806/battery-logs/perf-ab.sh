#!/usr/bin/env bash
# v0.71.0 release perf red — the LEGAL settle.
#
# THE RED: local-ci --perf reported 10/10 cells FAIL (-8.31% .. -24.75%) at the tag
# candidate, with correctness fully green.
#
# WHY local-ci's own verdict cannot settle it:
#   1. local-ci --perf verdicts each cell against a ROLLING MEDIAN OF PRIOR ROWS — rows from
#      2026-07-31 through 2026-08-05. That is a cross-day comparison, which this project's
#      measurement law forbids as evidence (clock drift invalidates it, denominator included).
#   2. The perf stage takes NO GPU LOCK (zero `flock` in tools/local-ci.sh). Its
#      window_free_now() poll samples only BETWEEN cells, so a neighbor job that runs during
#      the reps is invisible and the row still records window_clean:true.
#   3. A concurrent lane (research/rp-on-st-20260806, Q8RP-on-FP8-ST census) held run-gen on
#      this same card from 07:27Z onward — overlapping BOTH perf runs (07:44:01Z, 08:06:06Z).
#
# THE ONLY FORM THAT CAN TELL A REGRESSION FROM A POISONED WINDOW: same thermal window,
# interleaved A/B/A/B, N=5 each, GPU held exclusively under flock the whole time.
#   A = baseline  fcfe3837 (2026-08-05T16:12:24Z — the last green perf row: 41.52 tok/s)
#   B = candidate eea5a9ed (the v0.71.0 tag candidate)
# Cell = 31b-plain-short, byte-identical config to perf-cells.json.
#
# Verdict rule: compare A-median to B-median FROM THIS RUN ONLY. If A reproduces ~41.4 and B
# reads ~38, the drop is real code. If A and B agree, the rolling median is the invalid side
# and no v0.71 code regressed.
set -uo pipefail
cd /home/avifenesh/projects/wt-public-split

BASE=/tmp/perf-ab-baseline/run-gen
CAND=target/release/run-gen
MODEL=/data/ai-ml/hf-models/gemma4-31b-qat-gguf/gemma-4-31B_q4_0-it.gguf
PROMPT=research/gemma4-bringup/e4b-chat-watercycle-ids.txt
LOGD=/tmp/perf-ab-logs
N=5

[ -x "$BASE" ] || { echo "perf-ab: no baseline binary at $BASE"; exit 2; }
[ -x "$CAND" ] || { echo "perf-ab: no candidate binary at $CAND"; exit 2; }
[ -f "$MODEL" ] || { echo "perf-ab: no model at $MODEL"; exit 2; }
mkdir -p "$LOGD"

echo "== perf-ab: 31b-plain-short, N=$N interleaved, one lock hold =="
echo "   A = fcfe3837 baseline ($BASE)"
echo "   B = eea5a9ed candidate ($CAND)"
nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw --format=csv,noheader | sed 's/^/   entry: /'
echo "   other GPU compute apps at entry:"
nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader | sed 's/^/     /'
echo

: > "$LOGD/A.toks"; : > "$LOGD/B.toks"

# The whole interleaved sequence runs inside ONE lock hold: no neighbor may enter between
# reps, or the interleaving guarantee is void. Each rep's raw output is written to its own
# log FIRST and parsed from the log second (never parse a pipe).
flock -w 7200 /tmp/gpu5090.lock env \
  LOGD="$LOGD" N="$N" BASE="$BASE" CAND="$CAND" MODEL="$MODEL" PROMPT="$PROMPT" \
  bash -s <<'INNER'
rep() {
    local arm="$1" bin="$2" i="$3" log="$LOGD/$arm-$i.log"
    # shellcheck disable=SC2046
    MEMRA_NGEN=128 timeout 420 "$bin" "$MODEL" $(cat "$PROMPT") > "$log" 2>&1
    local rc=$? toks t
    toks=$(grep -oE "= [0-9.]+ tok/s" "$log" | tail -1 | grep -oE "[0-9.]+")
    t=$(nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader | tr -d ' ')
    printf '  %-4s rep%d: %-8s tok/s  (exit %d, %s)\n' "$arm" "$i" "${toks:-NONE}" "$rc" "$t"
    [ -n "$toks" ] && echo "$toks" >> "$LOGD/$arm.toks"
}
echo "  [warmup rep into the steady thermal state, discarded]"
rep warm "$CAND" 0 > /dev/null 2>&1
for i in $(seq 1 "$N"); do
    rep A "$BASE" "$i"
    rep B "$CAND" "$i"
done
INNER
echo

med() { sort -g "$1" | awk '{a[NR]=$1} END{if(NR==0){print 0;exit} print (NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2}'; }
AM=$(med "$LOGD/A.toks"); BM=$(med "$LOGD/B.toks")
echo "== VERDICT (this window only, N=$N each, interleaved, exclusive lock) =="
echo "   A fcfe3837 median: $AM tok/s   [$(tr '\n' ' ' < "$LOGD/A.toks")]"
echo "   B eea5a9ed median: $BM tok/s   [$(tr '\n' ' ' < "$LOGD/B.toks")]"
awk -v a="$AM" -v b="$BM" 'BEGIN{
  if (a<=0 || b<=0) { print "   INCONCLUSIVE — a median is missing"; exit }
  d=(b-a)/a*100;
  printf "   B vs A: %+.2f%%\n", d;
  if (d < -3.0) print "   => REAL REGRESSION in v0.71 code. RELEASE BLOCKER.";
  else if (d < -1.5) print "   => borderline (WARN band): widen N before any verdict.";
  else print "   => NO code regression in this cell. The rolling median was the invalid side.";
}'
echo "   raw logs: $LOGD/"
