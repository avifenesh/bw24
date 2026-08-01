#!/usr/bin/env bash
# chat-qa-short r4+r5: floor ratio at 25-tok ctx swung 0.73x..1.56x across fresh processes
# (plain arm bimodal with storage state); two more reps to report a distribution, not a coin flip.
set -u
BASE=/opt/dlami/nvme/hy3-accept-profile
MLOG=$BASE/logs/master.log
echo "$(date -u +%FT%TZ) TOPUP4-QUEUED (waits on gpu flock)" >> "$MLOG"
for tag in r4 r5; do
  bash "$BASE/accept-profile-driver.sh" "$BASE/prompts/chat-qa-short.txt" "chat-qa-short-$tag"
  rc=$?
  if [ $rc -ne 0 ]; then echo "$(date -u +%FT%TZ) TOPUP4-ABORT at chat-qa-short-$tag rc=$rc" >> "$MLOG"; exit 1; fi
  echo "$(date -u +%FT%TZ) done chat-qa-short-$tag" >> "$MLOG"
done
echo "$(date -u +%FT%TZ) TOPUP4-DONE" >> "$MLOG"
