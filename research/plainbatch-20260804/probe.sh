#!/usr/bin/env bash
# plainbatch probe (lane/plainbatch-probe, 2026-08-04): characterize the batched-PLAIN
# serve arm's divergence from the run-gen CLI tokenwise oracle at n=400/800/1600, N=3.
# Black-box: server launch + request pattern copied from tools/serve-st-gate.sh.
#
# Arms per (model, n, rep):
#   CLI oracle : run-gen tokenwise greedy decode -> "tokens: [...]" line
#   serve plain: memra-server MEMRA_SERVE_SPEC=0 (default batching) /v1/completions
#                native shape -> "tokens" array (per-token ids on the plain path)
# Server restarts per rep (no prefix-cache carryover between reps).
#
# Usage: probe.sh <tag> <model-path> <chat:0|1> <prompt> <n> <rep> [extra-server-env]
# GPU work runs under flock /tmp/gpu5090.lock (caller may hold it already via -o).
set -uo pipefail
cd "$(dirname "$0")/../.."

TAG=$1; MODEL=$2; CHAT=$3; PROMPT=$4; NGEN=$5; REP=$6; EXTRA_ENV=${7:-}
OUT=research/plainbatch-20260804
ADDR=127.0.0.1:8179
BASE=http://$ADDR
BIN=target/release

# ---- CLI oracle arm ----
CLI_LOG=$OUT/cli-$TAG-n$NGEN-r$REP.log
if [ "$CHAT" = "1" ]; then
  MEMRA_CHAT=1 MEMRA_NGEN=$NGEN $BIN/run-gen "$MODEL" --prompt "$PROMPT" > "$CLI_LOG" 2>&1
else
  MEMRA_NGEN=$NGEN $BIN/run-gen "$MODEL" --prompt "$PROMPT" > "$CLI_LOG" 2>&1
fi
rc=$?
[ $rc -eq 0 ] || { echo "run-gen FAILED rc=$rc; tail:"; tail -5 "$CLI_LOG"; exit 1; }
CLI_TOKENS=$(grep '^tokens: ' "$CLI_LOG" | tail -1 | sed 's/^tokens: //')
[ -n "$CLI_TOKENS" ] || { echo "run-gen printed no token stream"; exit 1; }

# ---- serve plain arm (batched tick default, spec off) ----
SRV_LOG=$OUT/server-$TAG-n$NGEN-r$REP.log
env $EXTRA_ENV MEMRA_SERVE_SPEC=0 MEMRA_MODELS="m=$MODEL" MEMRA_ADDR=$ADDR $BIN/memra-server \
  > "$SRV_LOG" 2>&1 &
SPID=$!
trap 'kill $SPID 2>/dev/null; wait $SPID 2>/dev/null' EXIT
up=0
for _ in $(seq 120); do curl -sf $BASE/health >/dev/null 2>&1 && { up=1; break; }; sleep 2; done
[ $up -eq 1 ] || { echo "server did not come up; tail:"; tail -5 "$SRV_LOG"; exit 1; }

if [ "$CHAT" = "1" ]; then CHATJSON=true; else CHATJSON=false; fi
SRV_JSON=$OUT/srv-$TAG-n$NGEN-r$REP.json
curl -sf -m 1200 $BASE/v1/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"m\",\"prompt\":$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$PROMPT"),\"chat\":$CHATJSON,\"max_tokens\":$NGEN,\"temperature\":0}" \
  > "$SRV_JSON"
rc=$?
kill $SPID 2>/dev/null; wait $SPID 2>/dev/null; trap - EXIT
[ $rc -eq 0 ] || { echo "server request FAILED rc=$rc"; exit 1; }
SRV_TOKENS=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["tokens"])' "$SRV_JSON")

# ---- compare ----
python3 - "$CLI_TOKENS" "$SRV_TOKENS" "$TAG" "$NGEN" "$REP" "$OUT" <<'EOF'
import ast, json, sys
cli = ast.literal_eval(sys.argv[1]); srv = ast.literal_eval(sys.argv[2])
tag, n, rep, out = sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
m = min(len(cli), len(srv))
diffs = [i for i in range(m) if cli[i] != srv[i]]
first = diffs[0] if diffs else None
row = {"tag": tag, "n": n, "rep": rep, "cli_len": len(cli), "srv_len": len(srv),
       "cmp_window": m, "n_diff_positions": len(diffs), "first_div": first,
       "div_positions_head": diffs[:20],
       "cli_at_first": cli[first:first+5] if first is not None else None,
       "srv_at_first": srv[first:first+5] if first is not None else None}
with open(f"{out}/table.jsonl", "a") as f:
    f.write(json.dumps(row) + "\n")
if first is None:
    print(f"[{tag} n={n} r{rep}] MATCH over {m} tokens (cli {len(cli)} srv {len(srv)})")
else:
    print(f"[{tag} n={n} r{rep}] first_div={first} n_diff={len(diffs)}/{m} "
          f"cli={cli[first:first+3]} srv={srv[first:first+3]}")
# persist raw streams for the margin probe
with open(f"{out}/tokens-cli-{tag}-n{n}-r{rep}.txt", "w") as f: f.write(repr(cli))
with open(f"{out}/tokens-srv-{tag}-n{n}-r{rep}.txt", "w") as f: f.write(repr(srv))
EOF
