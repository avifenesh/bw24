#!/usr/bin/env bash
# r3 for the three longer classes: fresh-process spec ratio showed >10% spread at long ctx
# (code-review r1 1.25x vs r2 0.97x) — N=3 gives the ratio column a median.
set -u
BASE=/opt/dlami/nvme/hy3-accept-profile
MLOG=$BASE/logs/master.log
while pgrep -f "run-all.sh|run-topup.sh" > /dev/null; do sleep 30; done
echo "$(date -u +%FT%TZ) TOPUP2-START" >> "$MLOG"
for p in code-review-medium agentic-tool summarize-medium; do
  bash "$BASE/accept-profile-driver.sh" "$BASE/prompts/$p.txt" "$p-r3"
  rc=$?
  if [ $rc -ne 0 ]; then echo "$(date -u +%FT%TZ) TOPUP2-ABORT at $p-r3 rc=$rc" >> "$MLOG"; exit 1; fi
  echo "$(date -u +%FT%TZ) done $p-r3" >> "$MLOG"
done
echo "$(date -u +%FT%TZ) TOPUP2-DONE" >> "$MLOG"
