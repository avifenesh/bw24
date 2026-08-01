#!/usr/bin/env bash
# r3 for all six classes (shorts: warm-storage ratio rep to replace cold r1;
# longs: third rep for a ratio median given >10% fresh-process spec spread).
set -u
BASE=/opt/dlami/nvme/hy3-accept-profile
MLOG=$BASE/logs/master.log
echo "$(date -u +%FT%TZ) TOPUP3-START" >> "$MLOG"
for p in chat-qa-short chat-prose-medium code-gen-short code-review-medium agentic-tool summarize-medium; do
  bash "$BASE/accept-profile-driver.sh" "$BASE/prompts/$p.txt" "$p-r3"
  rc=$?
  if [ $rc -ne 0 ]; then echo "$(date -u +%FT%TZ) TOPUP3-ABORT at $p-r3 rc=$rc" >> "$MLOG"; exit 1; fi
  echo "$(date -u +%FT%TZ) done $p-r3" >> "$MLOG"
done
echo "$(date -u +%FT%TZ) TOPUP3-DONE" >> "$MLOG"
