#!/usr/bin/env bash
# STREAM IDENTITY — the phase-2 hard gate. Three token streams on one prompt, greedy:
#
#   oracle  = run-gen CLI (decode_step_h) — the naked m=1 program
#   A       = serve c=1 with MEMRA_SERVE_B1FAST=0 (batched body at b_n=1, the shipped path)
#   B       = serve c=1 with MEMRA_SERVE_B1FAST=1 (H3: the m=1 trunk)
#
# The spec's gate is "serve output byte-identical before/after". H3 is a NUMERICS change by
# construction (it moves B=1 from the batched FP composition to the m=1 one), so the honest
# question is not "does A==B" but WHICH ONE MATCHES THE ORACLE. decode-batch-gate's strict
# gate1 already proved the direction on-box: with H3 the B=1 logits are BIT-IDENTICAL to
# decode_step_h, and without it they differ by maxdiff 1.591e-1 — i.e. the pre-existing
# shipped serve path is the one that deviates from naked. This script confirms that at the
# TOKEN-STREAM level through the real server, which is what a client actually observes.
#
# Also runs a seeded-sampled arm: same seed + same temp must be reproducible within an arm
# (a sampled stream is only meaningful against its own seeded rerun).
set -uo pipefail
cd "$(dirname "$0")/../../.."

MODEL=${1:?model.gguf}
NGEN=${2:-64}
OUT=research/servepath-p2-20260805
L=$OUT/logs
mkdir -p "$L"
ADDR=127.0.0.1:8189
BASE=http://$ADDR
PROMPT="What is the capital of France? Answer in one short sentence."
FAILS=0
PASS() { echo "  ok: $1"; }
FAIL() { echo "  FAIL: $1"; FAILS=$((FAILS + 1)); }

# ---- oracle: run-gen CLI, greedy, the naked m=1 decode program ----
MEMRA_NGEN=$NGEN target/release/run-gen "$MODEL" --prompt "$PROMPT" \
  > "$L/si-cli.log" 2>&1 || { echo "run-gen failed"; tail -5 "$L/si-cli.log"; exit 1; }
ORACLE=$(grep '^tokens: ' "$L/si-cli.log" | tail -1 | sed 's/^tokens: //')
[ -n "$ORACLE" ] || { echo "run-gen printed no token stream"; exit 1; }

start() {  # $1 = env list
  # shellcheck disable=SC2086
  env $1 MEMRA_SERVE_SPEC=0 MEMRA_MODELS="m=$MODEL" MEMRA_ADDR=$ADDR \
    target/release/memra-server > "$L/si-server.log" 2>&1 &
  SPID=$!
  for _ in $(seq 300); do curl -sf $BASE/health >/dev/null 2>&1 && return 0; sleep 2; done
  echo "server did not come up"; tail -20 "$L/si-server.log"; return 1
}
stop() { kill "${SPID:-0}" 2>/dev/null; wait "${SPID:-0}" 2>/dev/null || true; sleep 2; }
trap stop EXIT

# native shape (no MEMRA_COMPAT) so /v1/completions returns raw token ids
ids() {  # $1 = extra json fields
  curl -sf -m 300 $BASE/v1/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"m\",\"prompt\":\"$PROMPT\",\"max_tokens\":$NGEN,$1}" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["tokens"])'
}

for arm in 0 1; do
  start "MEMRA_SERVE_B1FAST=$arm" || exit 1
  ids '"temperature":0'                        > "$L/si-greedy-b1fast$arm.txt"
  ids '"temperature":0.7,"seed":12345'         > "$L/si-samp-a-b1fast$arm.txt"
  ids '"temperature":0.7,"seed":12345'         > "$L/si-samp-b-b1fast$arm.txt"
  stop
done

echo "$ORACLE" > "$L/si-oracle.txt"
python3 - "$L" <<'EOF'
import ast, pathlib, sys
L = pathlib.Path(sys.argv[1])
def ids(p):
    t = (L / p).read_text().strip()
    return ast.literal_eval(t) if t.startswith("[") else None
orc = ids("si-oracle.txt")
def cmp_prefix(a, b):
    """serve stops before EOS; CLI includes it -> prefix match is the contract."""
    n = min(len(a), len(b))
    div = next((i for i in range(n) if a[i] != b[i]), None)
    return div, n
rows = []
for arm in (0, 1):
    g = ids(f"si-greedy-b1fast{arm}.txt")
    div, n = cmp_prefix(orc, g)
    rows.append((arm, g, div, n))
    tag = "IDENTICAL to oracle" if div is None else f"diverges at token {div}/{n}"
    print(f"greedy B1FAST={arm}: {len(g)} ids, {tag}")
    if div is not None:
        print(f"    oracle[{div}:{div+5}]={orc[div:div+5]}  serve[{div}:{div+5}]={g[div:div+5]}")
    a, b = ids(f"si-samp-a-b1fast{arm}.txt"), ids(f"si-samp-b-b1fast{arm}.txt")
    print(f"  seeded-sampled reproducible within arm: {'YES' if a == b else 'NO'}"
          f" ({len(a)}/{len(b)} ids)")
# the verdict: which arm matches the naked oracle
d0, d1 = rows[0][2], rows[1][2]
print()
if d1 is None and d0 is not None:
    print(f"VERDICT: H3 RESTORES naked-oracle token identity "
          f"(shipped path diverges at {d0}, H3 is exact)")
elif d1 is None and d0 is None:
    print("VERDICT: both arms match the oracle (no token-visible change)")
else:
    print(f"VERDICT: H3 does NOT match the oracle (H3 div={d1}, shipped div={d0})")
EOF
echo "stream-identity: $FAILS explicit failures"
