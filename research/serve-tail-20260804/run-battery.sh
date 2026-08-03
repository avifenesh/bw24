#!/usr/bin/env bash
# serve-tail lane battery (2026-08-04): live receipts for the three OR-listing tail items —
#   1. /v1/models OR-schema enrichment (live shape vs the unit-pinned schema)
#   2. X-RateLimit-* headers under concurrency > cap (Remaining -> 0; dark-lane sheds carry trio)
#   3. graceful drain: SIGTERM mid-stream -> stream completes, new request 503s, exit 0
# GPU serialized via flock /tmp/gpu5090.lock (caller holds it). Logs tee'd raw, parsed second.
set -uo pipefail
cd "$(dirname "$0")/../.."

MODEL="${1:-/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf}"
[ -f "$MODEL" ] || { echo "battery: SKIP (no model at $MODEL)"; exit 0; }
OUT="research/serve-tail-20260804"
ADDR=127.0.0.1:8188
BASE=http://$ADDR
FAILS=0
PASS() { echo "  ok: $1"; }
FAIL() { echo "  FAIL: $1"; FAILS=$((FAILS+1)); }

[ -x target/release/memra-server ] || cargo build --release -p memra-server

start_server() {  # $1 = log name; rest = env prefix (env K=V ...)
  local log=$1; shift
  MEMRA_COMPAT=openai MEMRA_MODELS="tail=$MODEL" MEMRA_ADDR=$ADDR "$@" \
    target/release/memra-server > "$OUT/$log" 2>&1 &
  SPID=$!
  for _ in $(seq 120); do curl -sf $BASE/health >/dev/null 2>&1 && return 0; sleep 2; done
  echo "server did not come up; log tail:"; tail -5 "$OUT/$log"; return 1
}
stop_server() {
  [ -n "${SPID:-}" ] && [ "${SPID:-0}" -gt 1 ] || return 0
  kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null || true
  SPID=""
}
trap stop_server EXIT

echo "== item 1: /v1/models OR schema (live) =="
# small caps so the rate-limit hammer is a short burst (4 interactive, 2 harvest)
start_server server-rl.log env MEMRA_MAX_SESSIONS=4 MEMRA_LANE_MAX_HARVEST=2 || exit 1

curl -sf $BASE/v1/models | tee "$OUT/v1-models.json" | python3 -c '
import json,sys
r = json.load(sys.stdin)
assert r["object"] == "list"
e = r["data"][0]
assert e["id"] == "tail" and e["name"] == "tail"
assert isinstance(e["created"], int) and e["created"] > 1_700_000_000
assert isinstance(e["context_length"], int) and e["context_length"] > 0, "ctx from model config"
a = e["architecture"]
assert a["modality"] == "text->text"
assert a["tokenizer"], "tokenizer family from the plan"
assert a["instruct_type"] == "chatml", "qwen template -> chatml"
assert e["pricing"]["prompt"] == "0" and e["pricing"]["completion"] == "0"
assert e["top_provider"]["context_length"] == e["context_length"]
assert e["top_provider"]["max_completion_tokens"] is None, "honest null (context-bounded)"
print("  live entry:", json.dumps(e, indent=None)[:200], "...")
' && PASS "/v1/models live OR-schema entry" || FAIL "/v1/models schema"

echo "== item 2: rate-limit headers (hammer concurrency > cap) =="
# 8 concurrent interactive vs cap 4: every response carries the trio, Remaining hits 0.
HPIDS=()
for i in $(seq 8); do
  curl -s -D "$OUT/rl-hdr-$i.txt" -o "$OUT/rl-body-$i.json" -m 300 \
    $BASE/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"tail","messages":[{"role":"user","content":"Count to twenty in words."}],"max_tokens":48,"temperature":0}' &
  HPIDS+=($!)
done
wait "${HPIDS[@]}"
python3 - "$OUT" <<'EOF'
import sys, glob
out = sys.argv[1]
limits, remainings = [], []
for f in sorted(glob.glob(f"{out}/rl-hdr-*.txt")):
    h = {}
    for line in open(f):
        if ":" in line:
            k, v = line.split(":", 1)
            h[k.strip().lower()] = v.strip()
    assert "x-ratelimit-limit" in h, f"{f}: missing limit"
    assert "x-ratelimit-remaining" in h, f"{f}: missing remaining"
    assert "x-ratelimit-reset" in h, f"{f}: missing reset"
    limits.append(int(h["x-ratelimit-limit"]))
    remainings.append(int(h["x-ratelimit-remaining"]))
assert len(limits) == 8
assert all(l == 4 for l in limits), f"limit must be the cap: {limits}"
assert min(remainings) == 0, f"remaining must hit 0 at cap: {remainings}"
assert max(remainings) == 3, f"first request sees cap-1: {remainings}"
print(f"  limits={limits}")
print(f"  remainings(sorted)={sorted(remainings, reverse=True)}")
EOF
[ $? -eq 0 ] && PASS "interactive hammer: trio on all 8, Remaining hits 0 (limit=4)" \
  || FAIL "interactive rate-limit headers"

# dark-lane sheds: 6 concurrent harvest vs cap 2 -> some 429s; every shed carries the
# trio + Retry-After, and Remaining hit 0 before any shed.
HPIDS=()
for i in $(seq 6); do
  curl -s -D "$OUT/shed-hdr-$i.txt" -o "$OUT/shed-body-$i.json" -m 300 \
    -H 'x-lane: harvest' \
    $BASE/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"tail","messages":[{"role":"user","content":"Count to ten in words."}],"max_tokens":32,"temperature":0}' &
  HPIDS+=($!)
done
wait "${HPIDS[@]}"
python3 - "$OUT" <<'EOF'
import sys, glob
out = sys.argv[1]
sheds, oks, remainings = 0, 0, []
for f in sorted(glob.glob(f"{out}/shed-hdr-*.txt")):
    h, status = {}, None
    for line in open(f):
        if line.startswith("HTTP/"):
            status = int(line.split()[1])
        elif ":" in line:
            k, v = line.split(":", 1)
            h[k.strip().lower()] = v.strip()
    assert "x-ratelimit-limit" in h and "x-ratelimit-remaining" in h, f"{f}: missing trio"
    assert int(h["x-ratelimit-limit"]) == 2, f"harvest cap is 2: {h}"
    remainings.append(int(h["x-ratelimit-remaining"]))
    if status == 429:
        sheds += 1
        assert "retry-after" in h, f"{f}: 429 without Retry-After"
    elif status == 200:
        oks += 1
assert sheds + oks == 6, f"unexpected statuses: sheds={sheds} oks={oks}"
assert sheds >= 1, "concurrency 6 > cap 2 must shed at least one"
assert min(remainings) == 0, f"remaining must hit 0: {remainings}"
print(f"  harvest: {oks} served, {sheds} shed (429+Retry-After), remainings={sorted(remainings, reverse=True)}")
EOF
[ $? -eq 0 ] && PASS "harvest hammer: sheds carry trio + Retry-After, Remaining hit 0" \
  || FAIL "dark-lane shed headers"

stop_server

echo "== item 3: graceful drain (SIGTERM mid-stream) =="
start_server server-drain.log env MEMRA_DRAIN_S=30 || exit 1
# long streaming generation, then SIGTERM while it's mid-flight. First run of the
# original battery: 256 tokens finished INSIDE the 2s pre-SIGTERM sleep (spec bursts;
# drain log said "0 in flight") — the drain was correct but the probe raced the exit.
# 1024 tokens + SIGTERM once the stream has FIRST BYTES = deterministic mid-flight.
: > "$OUT/drain-stream.txt"
curl -s -N -m 300 -o "$OUT/drain-stream.txt" $BASE/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"tail","messages":[{"role":"user","content":"Write a numbered list of 200 animals, one per line."}],"max_tokens":1024,"temperature":0,"stream":true}' &
CURLPID=$!
for _ in $(seq 300); do   # wait for first stream bytes (generation provably live)
  [ -s "$OUT/drain-stream.txt" ] && break
  sleep 0.1
done
[ -s "$OUT/drain-stream.txt" ] || FAIL "stream never produced bytes before SIGTERM"
kill -TERM "$SPID"
sleep 0.3 # let the drain flag land
# new request during drain -> 503 + Retry-After
NEWCODE=$(curl -s -o "$OUT/drain-rejected.json" -D "$OUT/drain-rejected-hdr.txt" -w '%{http_code}' \
  -m 10 $BASE/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"tail","messages":[{"role":"user","content":"hi"}],"max_tokens":8}')
# health flips to draining
curl -s -m 5 $BASE/health > "$OUT/drain-health.json" || true
# the in-flight stream must complete
wait $CURLPID; CURLRC=$?
grep -q 'data: \[DONE\]' "$OUT/drain-stream.txt" && [ $CURLRC -eq 0 ] \
  && PASS "in-flight stream completed through drain ([DONE] received)" \
  || FAIL "in-flight stream did not complete (rc=$CURLRC)"
[ "$NEWCODE" = "503" ] && grep -qi 'retry-after' "$OUT/drain-rejected-hdr.txt" \
  && PASS "new request during drain: 503 + Retry-After" \
  || FAIL "drain rejection (got $NEWCODE)"
grep -q '"status":"draining"' "$OUT/drain-health.json" \
  && PASS "/health reports draining" || FAIL "/health drain flip"
grep -q 'draining (1 in flight' "$OUT/server-drain.log" \
  && PASS "SIGTERM landed with the stream in flight (server log)" \
  || FAIL "server log does not show 1 in flight at SIGTERM"
# process must exit 0 within the deadline
EXITRC=142
for _ in $(seq 40); do
  if ! kill -0 "$SPID" 2>/dev/null; then break; fi
  sleep 1
done
if kill -0 "$SPID" 2>/dev/null; then
  FAIL "server still alive 40s after SIGTERM"
else
  wait "$SPID"; EXITRC=$?
  [ $EXITRC -eq 0 ] && PASS "process exited 0 after drain" \
    || FAIL "exit code $EXITRC (want 0)"
fi
SPID=""

echo "battery: $FAILS failed"
[ $FAILS -eq 0 ]
