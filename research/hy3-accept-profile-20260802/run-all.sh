#!/usr/bin/env bash
# Master runner: 6 prompt classes x N reps, rep-major order (r1 all classes, then r2)
# so same-class reps are separated in time (clock/thermal-drift guard); plain-vs-spec
# arms interleave within each run-spec process. Aborts the whole batch on the first
# nonzero exit (run-spec asserts SELF-CONSISTENCY, so a FAIL exits nonzero).
set -u
BASE=/opt/dlami/nvme/hy3-accept-profile
MLOG=$BASE/logs/master.log
PROMPTS="chat-qa-short chat-prose-medium code-gen-short code-review-medium agentic-tool summarize-medium"

echo "$(date -u +%FT%TZ) MASTER-START" >> "$MLOG"
for rep in r1 r2; do
  for p in $PROMPTS; do
    bash "$BASE/accept-profile-driver.sh" "$BASE/prompts/$p.txt" "$p-$rep"
    rc=$?
    if [ $rc -ne 0 ]; then
      echo "$(date -u +%FT%TZ) MASTER-ABORT at $p-$rep rc=$rc" >> "$MLOG"
      exit 1
    fi
    echo "$(date -u +%FT%TZ) done $p-$rep" >> "$MLOG"
  done
done
echo "$(date -u +%FT%TZ) MASTER-DONE" >> "$MLOG"
echo MASTER-DONE
