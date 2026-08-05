#!/bin/bash
# Identity phase rerun (first pass captured empty text: wrong reasoning field name).
set -u
cd "$(dirname "$0")"
R=$PWD
log() { echo "[$(date -u +%H:%M:%SZ)] $*" >> "$R/logs/driver.log"; }
./run-ident.sh nv 3 32  0   nv-K3B32
./run-ident.sh nv 3 128 0   nv-K3B128
./run-ident.sh nv 3 128 0.3 nv-K3B128pm
./run-ident.sh q9 3 32  0   q9-K3B32
./run-ident.sh q9 3 128 0.3 q9-K3B128pm
for pair in "nv-K3B32 nv-K3B128" "nv-K3B32 nv-K3B128pm" "q9-K3B32 q9-K3B128pm"; do
  set -- $pair
  A=$(wc -c < "$R/logs/ident-$1.txt"); B=$(wc -c < "$R/logs/ident-$2.txt")
  if [ "$A" -le 1 ] || [ "$B" -le 1 ]; then
    log "identity $1 vs $2: EMPTY-CAPTURE (a=$A b=$B bytes) — NOT a pass"
  elif cmp -s "$R/logs/ident-$1.txt" "$R/logs/ident-$2.txt"; then
    log "identity $1 vs $2: BYTE-IDENTICAL ($A bytes)"
  else
    log "identity $1 vs $2: MISMATCH"
  fi
done
echo IDENT_DONE >> "$R/logs/driver.log"
echo IDENT_DONE
