#!/usr/bin/env bash
# Receipt run: does step35 at c=2 (DEFAULT batched serve mode, PP-2) produce the same greedy
# text as c=1? The unsplit decode_step_batch carries a step35 B>1 refusal (decode_batch.rs:561),
# but the PP door routes to decode_step_batch_ppn BEFORE that assert, and the ppn body only
# guards gemma4 — so over PP-2 (this SKU's ONLY placement) a B=2 tick walks the generic Full
# arm: global n_head (96, the max), 128-dim rope on every layer, no SWA window, no head-wise
# gate. If that path runs, c=2 greedy text must diverge from c=1's.
#
# Run ON THE BOX under flock: bash b2-geometry-ab.sh <tag>
set -uo pipefail
TAG=${1:-pre}
TS=$(date -u +%Y%m%dT%H%M%SZ)
RAW=~/step37/raw
LOG=$RAW/b2ab-$TAG-$TS.log
BIN=~/tokparity-memra/target/release/memra-server
TRUNK=$(ls ~/step37/models/step-3.7-flash/IQ4_XS/*00001-of-00003.gguf)
DRAFT=~/step37/models/step-3.7-flash/Step3.7-flash-mtp-Q8_0.gguf
PORT=8093
BASE=http://127.0.0.1:$PORT

exec > >(tee "$LOG") 2>&1
echo "=== b2-geometry-ab tag=$TAG $TS ==="
(
  flock -w 3600 9 || { echo "LOCK TIMEOUT"; exit 75; }
  echo "lock acquired $(date -u +%FT%TZ)"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

  # NAKED serve config: batching default ON — the config a listing operator runs.
  MEMRA_MODELS="step35=${TRUNK}+${DRAFT}" MEMRA_SERVE_SPEC=0 \
  MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_ADDR=127.0.0.1:$PORT \
  "$BIN" > "$RAW/b2ab-server-$TAG-$TS.log" 2>&1 &
  SRV=$!
  trap 'kill $SRV 2>/dev/null; wait $SRV 2>/dev/null' EXIT
  for i in $(seq 1 120); do
    sleep 5; curl -sf "$BASE/readyz" >/dev/null 2>&1 && break
    kill -0 $SRV 2>/dev/null || { echo SERVER DIED; exit 1; }
  done

  BODY='{"model":"step35","messages":[{"role":"user","content":"List the first eight prime numbers, comma separated, then explain in two sentences why 1 is not prime."}],"max_tokens":48,"temperature":0.0}'

  echo "--- c=1 reference (greedy, B=1 tick) ---"
  curl -s "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "$BODY" \
    | python3 -c 'import json,sys; r=json.load(sys.stdin); m=r["choices"][0]["message"]; print("REF:", json.dumps({"reasoning": m.get("reasoning"), "content": m.get("content")}))' \
    | tee /tmp/b2ab-ref.txt

  echo "--- c=4 concurrent identical requests (forms B>1 decode chunks after prefix-cache hits) ---"
  for i in 1 2 3 4; do
    curl -s "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "$BODY" \
      | python3 -c 'import json,sys; r=json.load(sys.stdin)
c=r.get("choices")
if c:
    m=c[0]["message"]
    print("C4:", json.dumps({"reasoning": m.get("reasoning"), "content": m.get("content")}))
else:
    print("C4: ERROR", json.dumps(r.get("error")))' \
      > /tmp/b2ab-c4-$i.txt &
  done
  wait
  cat /tmp/b2ab-c4-*.txt

  echo "--- verdict ---"
  python3 - <<'PY'
ref = open('/tmp/b2ab-ref.txt').read().strip().removeprefix('REF: ')
rows = [open(f'/tmp/b2ab-c4-{i}.txt').read().strip().removeprefix('C4: ') for i in (1,2,3,4)]
ok = all(r == ref for r in rows)
for i, r in enumerate(rows, 1):
    print(f"c4[{i}] {'==' if r == ref else '!='} ref")
print("VERDICT:", "IDENTICAL — B>1 path matches B=1" if ok
      else "DIVERGED — c>1 text differs from the c=1 greedy reference")
PY

  # count batched-tick evidence in the server log
  kill $SRV; wait $SRV 2>/dev/null; trap - EXIT
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
  echo "lock released $(date -u +%FT%TZ)"
) 9>/tmp/memra-gpu.lock
echo "rc=$?"
echo "LOG=$LOG"
