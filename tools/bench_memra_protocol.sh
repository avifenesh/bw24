#!/bin/bash
# memra side of the engine decision bench. Single-sequence run-gen, N runs.
# Usage: bench_memra.sh <model.gguf> [runs=5] [ngen=512]
# Expects to run from the memra repo root (backend/sm90a-boot build already done).
set -u
MODEL="${1:?model.gguf path}"
RUNS="${2:-5}"
NGEN="${3:-512}"
OUT="memra-single.json"

# 2048-token prompt: repeat the pangram; run-gen tokenizes text via MEMRA_PROMPT_FILE
PF=$(mktemp)
for _ in $(seq 1 190); do
  printf 'The quick brown fox jumps over the lazy dog. ' >> "$PF"
done

echo '{"engine":"memra","ngen":'"$NGEN"',"runs":[' > "$OUT"
for R in $(seq 0 "$RUNS"); do   # run 0 = warmup, discarded
  LINE=$(MEMRA_NGEN="$NGEN" MEMRA_PROMPT_FILE="$PF" ./target/release/run-gen "$MODEL" 2>&1 \
         | grep -E "generated [0-9]+ tokens")
  TPS=$(echo "$LINE" | grep -oE '= [0-9.]+ tok/s' | grep -oE '[0-9.]+')
  echo "run $R: $LINE"
  if [ "$R" -gt 0 ]; then
    SEP=$([ "$R" -gt 1 ] && echo ",")
    echo "  $SEP{\"run\":$R,\"decode_tps\":${TPS:-0}}" >> "$OUT"
  fi
done
echo ']}' >> "$OUT"
rm -f "$PF"
python3 - "$OUT" << 'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
xs = sorted(r["decode_tps"] for r in d["runs"])
med = xs[len(xs)//2]
d["decode_tps_median"] = med
json.dump(d, open(sys.argv[1], "w"), indent=1)
print("median decode tok/s:", med)
EOF
