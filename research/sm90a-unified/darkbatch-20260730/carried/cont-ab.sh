#!/usr/bin/env bash
# Increment (b) serving A/B: carried continuations single-chunk (base = (a) binary)
# vs batched (new binary). Same flow both arms: r1 parks (interactive, sequential),
# r2 = 3 concurrent harvest continuations (both arms resume identically).
# Exactness: r2 texts must match across arms (engine carried gate: streams MATCH).
set -u
MODEL=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
ADDR=127.0.0.1:18094
N=3; GEN2=24
pjson() { python3 -c 'import json,sys;print(json.dumps(sys.stdin.read()))'; }
p1() { echo "Story $1: Once upon a time in a small mountain village there lived a clockmaker who could repair anything except his own broken watch, and every morning he"; }
req() { local lane=(); [ -n "$3" ] && lane=(-H "x-lane: $3")
  curl -s "http://$ADDR/v1/completions" -H 'content-type: application/json' "${lane[@]}" \
    -d "{\"model\":\"m\",\"prompt\":$(printf %s "$1" | pjson),\"max_tokens\":$2,\"temperature\":0}" \
    | python3 -c 'import json,sys;print(json.load(sys.stdin)["choices"][0]["text"],end="")'; }

arm() { # name binary
  pkill -9 -x bw24-server 2>/dev/null; sleep 1
  local log=/tmp/contab-$1.log out=/tmp/contab-$1; mkdir -p "$out"
  BW24_SERVE_SPEC=0 BW24_COMPAT=openai BW24_MODELS="m=$MODEL" BW24_ADDR=$ADDR "$2" > "$log" 2>&1 &
  local srv=$!
  for _ in $(seq 60); do curl -sf "http://$ADDR/health" >/dev/null 2>&1 && break; sleep 2; done
  for i in $(seq 1 $N); do req "$(p1 $i)" 8 "" > "$out/r1-$i.txt"; done
  sleep 1
  local t0=$(date +%s.%N)
  for i in $(seq 1 $N); do
    req "$(p1 $i)$(cat "$out/r1-$i.txt") decided that today he would finally climb the tower and examine the great bell mechanism because" $GEN2 "harvest" > "$out/r2-$i.txt" &
  done
  wait $(jobs -pr | grep -v "^$srv\$")
  local t1=$(date +%s.%N)
  kill -9 $srv 2>/dev/null
  echo "$1: r2 wall $(echo "$t1 $t0" | awk '{printf "%.3f",$1-$2}')s resumes=$(grep -c kv-reuse "$log") batches: $(grep "prime-batch dark" "$log" | tail -1)"
}

for rep in 1 2 3; do
  arm base-$rep /tmp/bw24-server-darkbatch
  arm new-$rep /home/avifenesh/projects/bw24-unified/target/release/bw24-server
done
pkill -9 -x bw24-server 2>/dev/null
F=0
for rep in 1 2 3; do for i in 1 2 3; do
  cmp -s /tmp/contab-base-$rep/r2-$i.txt /tmp/contab-new-$rep/r2-$i.txt || { echo "MISMATCH rep$rep slot$i"; F=$((F+1)); }
done; done
echo "cross-arm exactness: $((9-F))/9"
