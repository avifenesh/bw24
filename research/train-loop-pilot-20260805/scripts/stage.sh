#!/bin/bash
# stage.sh <stage-name> <cmd...>  — timed, tee'd stage wrapper for the train-loop pilot.
# Appends one row per stage to /root/pilot/timings.jsonl; full log at logs/<stage>.log.
set -uo pipefail
export LD_LIBRARY_PATH=/usr/local/cuda-13.1/compat
export PATH=$HOME/.cargo/bin:/usr/local/cuda-13.1/bin:$PATH
NAME=$1; shift
LOG=/root/pilot/logs/${NAME}.log
T0=$(date +%s)
echo "[stage:$NAME] start $(date -u +%FT%TZ)" | tee "$LOG"
"$@" 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
T1=$(date +%s)
echo "[stage:$NAME] end rc=$RC wall=$((T1-T0))s" | tee -a "$LOG"
printf '{"stage":"%s","t0":%s,"t1":%s,"wall_s":%s,"rc":%s,"utc":"%s"}\n' \
  "$NAME" "$T0" "$T1" "$((T1-T0))" "$RC" "$(date -u +%FT%TZ)" >> /root/pilot/timings.jsonl
exit "$RC"
