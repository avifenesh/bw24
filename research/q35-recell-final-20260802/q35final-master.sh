#!/bin/bash
# Master sequence: kernel-check (GATE) -> 3-arm probe -> N=5 board cell -> argmax.
# Every GPU phase under flock -w 3600 /tmp/gpu-h100.lock (Hy3 acceptance lane shares
# the box); lock timeout -> log + retry, up to 8h per phase. KC failure on the batch
# router entries STOPS the sequence (row then ships as-merged).
set -u
cd ~/memra
STATE=/tmp/q35final-state
: > $STATE

runlocked() { # tag script
  local tag=$1 cmd=$2 i
  rm -f /tmp/rc-$tag
  for i in 1 2 3 4 5 6 7 8; do
    flock -w 3600 /tmp/gpu-h100.lock bash -c "$cmd; echo \$? > /tmp/rc-$tag"
    [ -f /tmp/rc-$tag ] && return "$(cat /tmp/rc-$tag)"
    echo "[$tag] lock busy, retry $i $(date -u +%FT%TZ)" >> $STATE
  done
  echo "[$tag] LOCK-STARVED after 8h" >> $STATE
  return 97
}

echo "start $(date -u +%FT%TZ)" >> $STATE

# 1. kernel-check gate
runlocked kc './target/release/kernel-check > /tmp/kc-q35final.log 2>&1'
KC_RC=$?
if ! tail -1 /tmp/kc-q35final.log | grep -q "ALL GREEN"; then
  echo "KC-FAIL rc=$KC_RC $(date -u +%FT%TZ)" >> $STATE
  exit 1
fi
if grep -E "router batch-twin" /tmp/kc-q35final.log | grep -q "FAIL"; then
  echo "KC-ROUTER-FAIL $(date -u +%FT%TZ)" >> $STATE
  exit 1
fi
echo "KC-GREEN $(date -u +%FT%TZ)" >> $STATE

# 2. three-arm probe (whole interleave under one lock hold = same session)
runlocked probe 'bash /tmp/run-probe3-q35final.sh'
echo "PROBE-DONE rc=$? $(date -u +%FT%TZ)" >> $STATE

# 3. board cell (same-session interleaved pair under one lock hold)
runlocked cell 'bash /tmp/run-cell-q35final.sh > /tmp/cell-q35final.log 2>&1'
echo "CELL-DONE rc=$? $(date -u +%FT%TZ)" >> $STATE

# 4. argmax sanity
runlocked argmax 'bash /tmp/run-argmax-q35final.sh'
echo "ARGMAX-DONE rc=$? $(date -u +%FT%TZ)" >> $STATE

echo "ALL-DONE $(date -u +%FT%TZ)" >> $STATE
