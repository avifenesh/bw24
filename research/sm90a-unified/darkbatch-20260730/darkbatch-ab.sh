#!/usr/bin/env bash
# A/B: dark-lane batch-prime vs single-chunk (base) — harvest profile.
# Per trial: boot server (SPEC serve off), start 1 interactive gen-128 stream,
# 1s later fire N=6 concurrent short harvest completions; record dark-wall
# (all 6 complete) and int-wall (interactive request total). x5 per arm, interleaved.
set -u
MODEL=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
ADDR=127.0.0.1:18092
N=6; GEN=32
OUT=/tmp/darkbatch-ab.jsonl
: > "$OUT"

pjson() { python3 -c 'import json,sys;print(json.dumps(sys.stdin.read()))'; }
prompt() { echo "Task $1: List the first ten prime numbers, one per line, then state their sum and briefly explain why two is the only even prime number in mathematics. Answer:"; }
IPROMPT=$(echo "Write a detailed multi-paragraph explanation of how a rotary positional embedding works in a transformer attention layer, covering the rotation matrices and frequency bands." | pjson)

run_trial() { # $1=arm $2=binary $3=rep
  pkill -9 -x bw24-server 2>/dev/null; sleep 1
  local log=/tmp/ab-$1-$3.log
  BW24_SERVE_SPEC=0 BW24_COMPAT=openai BW24_MODELS="m=$MODEL" BW24_ADDR=$ADDR "$2" > "$log" 2>&1 &
  local srv=$!
  for _ in $(seq 60); do curl -sf "http://$ADDR/health" >/dev/null 2>&1 && break; sleep 2; done
  curl -sf "http://$ADDR/health" >/dev/null || { echo "boot FAIL $1/$3"; kill -9 $srv; return 1; }
  # warmup: one tiny request (first-request jit/graph costs out of the measurement)
  curl -s "http://$ADDR/v1/completions" -H 'content-type: application/json' \
    -d "{\"model\":\"m\",\"prompt\":$(prompt 0 | pjson),\"max_tokens\":8,\"temperature\":0}" >/dev/null
  local ti0=$(date +%s.%N)
  curl -s "http://$ADDR/v1/completions" -H 'content-type: application/json' \
    -d "{\"model\":\"m\",\"prompt\":$IPROMPT,\"max_tokens\":128,\"temperature\":0}" >/dev/null &
  local intpid=$!
  sleep 1
  local td0=$(date +%s.%N)
  for i in $(seq 1 $N); do
    curl -s "http://$ADDR/v1/completions" -H 'content-type: application/json' -H 'x-lane: harvest' \
      -d "{\"model\":\"m\",\"prompt\":$(prompt $i | pjson),\"max_tokens\":$GEN,\"temperature\":0}" >/dev/null &
  done
  wait $(jobs -pr | grep -v "^$srv$" | grep -v "^$intpid$") 2>/dev/null
  local td1=$(date +%s.%N)
  wait $intpid 2>/dev/null
  local ti1=$(date +%s.%N)
  local batched=$(grep -c "prime-batch dark" "$log" 2>/dev/null || echo 0)
  kill -9 $srv 2>/dev/null
  local dw iw; dw=$(echo "$td1 $td0" | awk '{printf "%.3f", $1-$2}'); iw=$(echo "$ti1 $ti0" | awk '{printf "%.3f", $1-$2}')
  echo "{\"arm\":\"$1\",\"rep\":$3,\"dark_wall_s\":$dw,\"int_wall_s\":$iw,\"batch_lines\":$batched,\"n\":$N,\"gen\":$GEN}" | tee -a "$OUT"
}

for rep in 1 2 3 4 5; do
  run_trial base /tmp/bw24-server-base $rep
  run_trial dark /tmp/bw24-server-darkbatch $rep
done
pkill -9 -x bw24-server 2>/dev/null
echo "== medians =="
python3 - <<'EOF'
import json, statistics
rows=[json.loads(l) for l in open('/tmp/darkbatch-ab.jsonl')]
for arm in ('base','dark'):
    a=[r for r in rows if r['arm']==arm]
    print(arm, 'dark_wall med %.3fs'%statistics.median(x['dark_wall_s'] for x in a),
          'int_wall med %.3fs'%statistics.median(x['int_wall_s'] for x in a),
          'batch_lines', [x['batch_lines'] for x in a])
EOF
