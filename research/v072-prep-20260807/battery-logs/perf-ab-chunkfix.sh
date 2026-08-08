#!/usr/bin/env bash
# chunkfix merge perf settle — 31b-plain-short, A=5557fc49 (pre-merge parent), B=006aca75+ (merged tip).
# Same thermal window, interleaved A/B/A/B, N=5 each, one exclusive flock hold.
# The dirty-window tripwires (window_clean=false, persistent co-residents) cannot settle this;
# this form can: both arms share whatever window exists.
set -uo pipefail
cd /home/avifenesh/projects/wt-public-split

BASE=/tmp/perf-ab-baseline-5557/run-gen
CAND=target/release/run-gen
MODEL=/data/ai-ml/hf-models/gemma4-31b-qat-gguf/gemma-4-31B_q4_0-it.gguf
PROMPT=research/gemma4-bringup/e4b-chat-watercycle-ids.txt
LOGD=/tmp/perf-ab-chunkfix-logs
N=5

[ -x "$BASE" ] || { echo "no baseline binary"; exit 2; }
[ -x "$CAND" ] || { echo "no candidate binary"; exit 2; }
mkdir -p "$LOGD"
echo "== perf-ab: 31b-plain-short, N=$N interleaved, one lock hold =="
echo "   A = 5557fc49 baseline, B = merged tip (006aca75 code)"
nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw --format=csv,noheader | sed 's/^/   entry: /'
echo "   co-residents at entry:"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | sed 's/^/     /'
: > "$LOGD/A.toks"; : > "$LOGD/B.toks"

flock -w 7200 /tmp/gpu5090.lock env \
  LOGD="$LOGD" N="$N" BASE="$BASE" CAND="$CAND" MODEL="$MODEL" PROMPT="$PROMPT" \
  bash -s <<'INNER'
rep() {
    local arm="$1" bin="$2" i="$3" log="$LOGD/$arm-$i.log"
    MEMRA_NGEN=128 timeout 420 "$bin" "$MODEL" $(cat "$PROMPT") > "$log" 2>&1
    local rc=$? toks t
    toks=$(grep -oE "= [0-9.]+ tok/s" "$log" | tail -1 | grep -oE "[0-9.]+")
    t=$(nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader | tr -d ' ')
    printf '  %-4s rep%d: %-8s tok/s  (exit %d, %s)\n' "$arm" "$i" "${toks:-NONE}" "$rc" "$t"
    [ -n "$toks" ] && echo "$toks" >> "$LOGD/$arm.toks"
}
echo "  [warmup, discarded]"
rep warm "$CAND" 0 > /dev/null 2>&1
for i in $(seq 1 "$N"); do
    rep A "$BASE" "$i"
    rep B "$CAND" "$i"
done
INNER

med() { sort -g "$1" | awk '{a[NR]=$1} END{if(NR==0){print 0;exit} print (NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2}'; }
AM=$(med "$LOGD/A.toks"); BM=$(med "$LOGD/B.toks")
echo "== VERDICT (this window only, N=$N each, interleaved, exclusive lock) =="
echo "   A median: $AM   [$(tr '\n' ' ' < "$LOGD/A.toks")]"
echo "   B median: $BM   [$(tr '\n' ' ' < "$LOGD/B.toks")]"
awk -v a="$AM" -v b="$BM" 'BEGIN{
  if (a<=0 || b<=0) { print "   INCONCLUSIVE"; exit }
  d=(b-a)/a*100; printf "   B vs A: %+.2f%%\n", d;
  if (d < -3.0) print "   => REAL REGRESSION. TAG BLOCKER.";
  else if (d < -1.5) print "   => borderline WARN: widen N.";
  else print "   => NO code regression in this cell; the rolling-median tripwire was window noise.";
}'
