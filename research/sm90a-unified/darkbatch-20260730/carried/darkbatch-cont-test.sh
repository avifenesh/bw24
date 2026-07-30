#!/usr/bin/env bash
# Increment (b) functional: CONTINUATION dark batch-prime.
# Round 1 parks caches (3 prompts, short gen). Round 2 extends each prompt
# (resume from the KV-reuse pool, pos>0) — arm A sequential/interactive (reference),
# arm B concurrent/harvest (batched). Greedy => texts must match; server log must
# show "[prime-batch dark] ... carried>0" in arm B only.
set -u
MODEL="${1:-/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf}"
BIN="${2:-target/release/bw24-server}"
ADDR=127.0.0.1:18093
N=3; GEN2=24
pjson() { python3 -c 'import json,sys;print(json.dumps(sys.stdin.read()))'; }
p1() { echo "Story $1: Once upon a time in a small mountain village there lived a clockmaker who could repair anything except his own broken watch, and every morning he"; }

boot() { # logfile
  pkill -9 -x bw24-server 2>/dev/null; sleep 1
  BW24_SERVE_SPEC=0 BW24_COMPAT=openai BW24_MODELS="m=$MODEL" BW24_ADDR=$ADDR "$BIN" > "$1" 2>&1 &
  SRV=$!
  for _ in $(seq 60); do curl -sf "http://$ADDR/health" >/dev/null 2>&1 && return 0; sleep 2; done
  echo "boot FAIL"; return 1
}
req() { # prompt-string gen lane -> text
  local lane_hdr=()
  [ -n "$3" ] && lane_hdr=(-H "x-lane: $3")
  curl -s "http://$ADDR/v1/completions" -H 'content-type: application/json' "${lane_hdr[@]}" \
    -d "{\"model\":\"m\",\"prompt\":$(printf %s "$1" | pjson),\"max_tokens\":$2,\"temperature\":0}" \
    | python3 -c 'import json,sys;print(json.load(sys.stdin)["choices"][0]["text"],end="")'
}

run_arm() { # arm(seq|dark) logfile outdir
  boot "$2" || return 1
  mkdir -p "$3"
  # round 1: park caches
  for i in $(seq 1 $N); do req "$(p1 $i)" 8 "" > "$3/r1-$i.txt"; done
  sleep 1
  # round 2: extended prompts (prefix + r1 text + new tail, resume pos>0)
  if [ "$1" = seq ]; then
    for i in $(seq 1 $N); do
      req "$(p1 $i)$(cat "$3/r1-$i.txt") decided that today he would finally climb the tower and examine the great bell mechanism because" $GEN2 "harvest" > "$3/r2-$i.txt"
    done
  else
    for i in $(seq 1 $N); do
      req "$(p1 $i)$(cat "$3/r1-$i.txt") decided that today he would finally climb the tower and examine the great bell mechanism because" $GEN2 "harvest" > "$3/r2-$i.txt" &
    done
    wait $(jobs -pr | grep -v "^$SRV\$")
  fi
  kill -9 $SRV 2>/dev/null
}

run_arm seq  /tmp/cont-seq.log  /tmp/cont-seq
run_arm dark /tmp/cont-dark.log /tmp/cont-dark
pkill -9 -x bw24-server 2>/dev/null

echo "== batch lines (dark arm) =="
grep "prime-batch dark" /tmp/cont-dark.log || echo "NONE"
grep -c "prime-batch dark" /tmp/cont-seq.log | sed 's/^/seq-arm batch lines: /'
FAILS=0
for i in $(seq 1 $N); do
  cmp -s /tmp/cont-seq/r2-$i.txt /tmp/cont-dark/r2-$i.txt || { echo "MISMATCH slot $i"; diff <(cat /tmp/cont-seq/r2-$i.txt) <(cat /tmp/cont-dark/r2-$i.txt) | head -4; FAILS=$((FAILS+1)); }
done
echo "exactness: $((N-FAILS))/$N"
grep -q "carried=[1-9]" /tmp/cont-dark.log && [ "$FAILS" -eq 0 ] && echo PASS || echo CHECK
