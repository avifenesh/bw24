#!/usr/bin/env bash
# r3 top-ups: the three short classes whose r1 plain arm ran on cold NVMe page cache
# (first runs of the session). Gives N=2 warm-storage reps for the ratio column.
set -u
BASE=/opt/dlami/nvme/hy3-accept-profile
MLOG=$BASE/logs/master.log
# wait for the master battery to finish before touching the GPU queue
while pgrep -f "run-all.sh" > /dev/null; do sleep 30; done
echo "$(date -u +%FT%TZ) TOPUP-START" >> "$MLOG"
for p in chat-qa-short chat-prose-medium code-gen-short; do
  bash "$BASE/accept-profile-driver.sh" "$BASE/prompts/$p.txt" "$p-r3"
  rc=$?
  if [ $rc -ne 0 ]; then echo "$(date -u +%FT%TZ) TOPUP-ABORT at $p-r3 rc=$rc" >> "$MLOG"; exit 1; fi
  echo "$(date -u +%FT%TZ) done $p-r3" >> "$MLOG"
done
echo "$(date -u +%FT%TZ) TOPUP-DONE" >> "$MLOG"
