#!/usr/bin/env bash
# Sequential bounded-chunk own-gen corpora for the three targets, priority order
# Ornith-35B > Ornith-9B > KAT (RECIPE.md). Each 64-prompt chunk is its own flock hold;
# a VRAM guard (sibling lanes keep resident servers) gates every chunk. Rerun-safe: a
# completed corpus (254 lines) is skipped, a partial one resumes.
set -uo pipefail
RD=/home/avifenesh/projects/wt-ornith-drafters/research/ornith-drafters-20260801

wait_vram() {  # <need-MiB> — poll up to 2h for a window
  local need=$1 free i
  for i in $(seq 1 240); do
    free=$(( $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits) \
           - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) ))
    [ "$free" -ge "$need" ] && return 0
    sleep 30
  done
  echo "[chain] no ${need}MiB VRAM window within 2h — stopping" >&2
  return 1
}
done_prompts() { wc -l < "$RD/corpus/$1-owngen-ids.txt" 2>/dev/null || echo 0; }
run_model() {  # <key> <need-MiB>
  local key=$1 need=$2
  while [ "$(done_prompts "$key")" -lt 254 ]; do
    wait_vram "$need" || return 1
    "$RD/gen-corpus-chunk.sh" "$key" 64 || return 1
    sleep 5   # lock-courtesy gap: sibling lanes can grab the rig between chunks
  done
  echo "[chain] $key corpus complete: $(done_prompts "$key")/254 prompts"
}

run_model ornith35b 22000 && \
run_model ornith9b 11000 && \
run_model katcoder 20000
echo "[chain] finished rc=$?"
