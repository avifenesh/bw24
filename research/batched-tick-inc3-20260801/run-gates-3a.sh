#!/bin/bash
# inc3 (3a): decode-batch-gate chunk-size matrix on the 5090 — B=8 control, B=16, B=32.
# gate2 bit-strength = the arbiter (per-seq logits vs isolated B=1, bit-checked);
# gate3a/b/c ride along. Config mode, steps 32 AND 160 (160 crosses the t_kv=96 vec
# floor so the seqs fa arm actually engages — inc2 law).
set -u
W=/home/avifenesh/projects/wt-batched-tick-3
R=$W/research/batched-tick-inc3-20260801
M=/home/avifenesh/models/qwen3.5-9b-judge-q8_0.gguf
G=$W/target/release/decode-batch-gate
cd "$W" || exit 1
for spec in "8:" "16:MEMRA_DECODE_BATCH_CAP=16" "32:MEMRA_DECODE_BATCH_CAP=32"; do
  B=${spec%%:*}; ENVV=${spec#*:}
  for S in 32 160; do
    log=$R/dbg-config-b$B-s$S.log
    echo "=== decode-batch-gate --batch $B --steps $S (env: ${ENVV:-none}) $(date -u +%FT%TZ) ===" | tee "$log"
    if [ -n "$ENVV" ]; then
      flock /tmp/gpu5090.lock env $ENVV "$G" "$M" --steps "$S" --batch "$B" --mode config >>"$log" 2>&1
    else
      flock /tmp/gpu5090.lock "$G" "$M" --steps "$S" --batch "$B" --mode config >>"$log" 2>&1
    fi
    echo "exit=$? $(tail -1 "$log")"
  done
done
echo GATES-3A-DONE
