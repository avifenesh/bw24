#!/usr/bin/env bash
# serve c=1 / c=8 A/B driver — server restarted per arm, arms interleaved, order
# alternated across reps (the phase-1 §5.3 harness rule: memra-server + load-serve.py,
# NEVER decode-batch-bench).
#
# Usage: serve-ab.sh <label> <model.gguf> <concurrency> <requests> <nreps> "<ARM_A_ENV>" "<ARM_B_ENV>"
# Env vars per arm are a space-separated K=V list ("" = naked defaults).
set -uo pipefail
cd "$(dirname "$0")/../../.."

LABEL=$1; MODEL=$2; CONC=$3; REQS=$4; REPS=$5; ENV_A=$6; ENV_B=$7
OUT=research/servepath-p2-20260805
LOGS=$OUT/logs
mkdir -p "$LOGS"
ADDR=127.0.0.1:8188
BASE=http://$ADDR
BIN=target/release/memra-server
GREEDY="${GREEDY:-}"          # set to --greedy for temperature=0
MAXTOK="${MAXTOK:-128}"
EXTRA="${EXTRA:-}"            # extra env applied to BOTH arms (e.g. MEMRA_SERVE_SPEC=0)

start() {  # $1 = env list
  # shellcheck disable=SC2086
  env $EXTRA $1 MEMRA_MODELS="m=$MODEL" MEMRA_ADDR=$ADDR MEMRA_COMPAT=openai \
    $BIN > "$LOGS/$LABEL-server.log" 2>&1 &
  SPID=$!
  for _ in $(seq 180); do curl -sf $BASE/health >/dev/null 2>&1 && return 0; sleep 2; done
  echo "server did not come up"; tail -20 "$LOGS/$LABEL-server.log"; return 1
}
stop() { kill "${SPID:-0}" 2>/dev/null; wait "${SPID:-0}" 2>/dev/null || true; sleep 3; }
trap stop EXIT

run_arm() {  # $1 = arm name, $2 = env, $3 = rep
  start "$2" || exit 1
  python3 tools/load-serve.py --base $BASE --model m --concurrency "$CONC" \
    --requests "$REQS" --max-tokens "$MAXTOK" $GREEDY --warmup 1 \
    --label "$LABEL-$1-r$3" --out "$OUT/serve-points.jsonl" \
    > "$LOGS/$LABEL-$1-r$3.log" 2>&1
  # /metrics BEFORE the kill: step_p50_ms is the DECODE-ONLY comparator (agg_tok_s folds
  # prefill in, so it dilutes a decode-only lever). Captured per arm, per rep.
  curl -s $BASE/metrics > "$LOGS/$LABEL-$1-r$3-metrics.json" 2>/dev/null
  python3 -c '
import json,sys
d=json.load(open(sys.argv[1]))
p50=d.get("step_p50_ms") or 0
row={"label":sys.argv[2],"max_tokens":int(sys.argv[3]),"conc":int(sys.argv[4]),
     "step_p50_ms":p50,"step_tok_s":(1000.0/p50 if p50 else 0),
     "step_p99_ms":d.get("step_p99_ms"),"tokens_out":d.get("tokens_out")}
open(sys.argv[5],"a").write(json.dumps(row)+"\n")
' "$LOGS/$LABEL-$1-r$3-metrics.json" "$LABEL-$1-r$3" "$MAXTOK" "$CONC" \
    "$OUT/serve-metrics.jsonl" 2>/dev/null || true
  echo "  $1 r$3: $(python3 "$OUT/scripts/parse.py" point < "$LOGS/$LABEL-$1-r$3.log") \
$(python3 "$OUT/scripts/parse.py" metrics < "$LOGS/$LABEL-$1-r$3-metrics.json" 2>/dev/null)"
  cp "$LOGS/$LABEL-server.log" "$LOGS/$LABEL-$1-r$3-server.log"
  stop
}

for r in $(seq 1 "$REPS"); do
  if [ $((r % 2)) -eq 1 ]; then
    run_arm A "$ENV_A" "$r"; run_arm B "$ENV_B" "$r"
  else
    run_arm B "$ENV_B" "$r"; run_arm A "$ENV_A" "$r"
  fi
  echo "rep $r done"
done
echo "=== $LABEL complete: $OUT/serve-points.jsonl ==="
