#!/bin/bash
# Resume of run-depth-ab.sh rep3 after the harness kill at row 76 (kat/new d2048 rep3 was
# the last recorded point; reps 1-2 complete for all models). Runs EXACTLY the remaining
# points — no duplicate cells. Same measurement shape, same jsonl.
set -u
W=/home/avifenesh/projects/wt-fa-decode-deep
R=$W/research/fa-decode-deep-20260802
# pull the vars+functions from the main script without executing its bottom loop
eval "$(sed -n '/^set -u$/,/^echo "=== FA-DEEP DEPTH A\/B/p' "$R/run-depth-ab.sh" | head -n -1 | tail -n +2)"

{
  echo "=== FA-DEEP DEPTH A/B RESUME rep3 $(date -u +%FT%TZ) git=$GIT_SHA ==="
  rep=3
  m=kat
  for d in 4096 6144; do
    memra_point "$m" old "$d" "$rep"
    memra_point "$m" new "$d" "$rep"
  done
  llama_model kat 3
  for m in q35 o35b; do
    for d in $DEPTHS; do
      memra_point "$m" old "$d" "$rep"
      memra_point "$m" new "$d" "$rep"
    done
    llama_model "$m" 3
  done
  echo "DEPTH-AB-DONE $(date -u +%FT%TZ)"
} 2>&1 | tee -a "$R/ab-console.log"
