#!/usr/bin/env bash
# serve-st gate (serve-st lane, 2026-08-04): an HF safetensors checkpoint DIR served by
# memra-server end-to-end, plus the CLI-vs-server exactness contract:
#
#   1. /models lists the ST model.
#   2. /v1/chat/completions returns coherent text through the checkpoint's own chat
#      template (tokenizer_config chat_template / chat_template.jinja).
#   3. EXACTNESS: the SAME checkpoint, SAME prompt through the SAME template, greedy —
#      run-gen's ST dir branch (tokenwise decode) and the server (batched prime +
#      serving decode) must produce IDENTICAL token id streams. The server arm runs
#      MEMRA_SERVE_SPEC=0: with spec on, burst Token events carry ONE id per burst so
#      the response `tokens` array is not the per-token stream (worker contract, not a
#      bug). The only tolerated difference is the trailing EOS id (the CLI stream
#      includes it, server Token events stop before it).
#   4. ST SERVE-SPEC exactness (#68 closed 2026-08-04, quarantine LIFTED): the DEFAULT
#      server (spec bursts ON for MTP dir checkpoints) must produce greedy text that
#      PREFIX-matches the TOKENWISE serve oracle (MEMRA_SERVE_SPEC=0 MEMRA_SERVE_BATCH=0
#      — same worker, plain decode_step) at a 400-token window, past the pre-fix
#      corruption onset (~250 tok). Tolerance: burst overshoot only (the spec arm may
#      emit up to K extra tokens past max_tokens; the shorter text must be a prefix of
#      the longer). Comparator choice is deliberate: the BATCHED plain arm carries the
#      accepted decode-config near-tie FP class (decode-batch-gate's jurisdiction) and
#      the run-gen CLI carries its own near-tie gap vs serve decode at long windows —
#      the tokenwise serve arm is spec-verify's exactness twin inside the same worker.
#      Root cause + receipts: research/fp8ship-20260804/RESULTS.md (the persistent
#      draft graph replayed with dangling pool addresses; was never ST-specific).
#
# Usage: tools/serve-st-gate.sh [st_dir]   (defaults to the local qwen3.5-4B BF16 ckpt)
# GPU: callers wrap in `flock /tmp/gpu5090.lock` per the box convention.
set -uo pipefail
cd "$(dirname "$0")/.."

ST="${1:-/data/ai-ml/hf-models/qwen35-4b-hf}"
[ -d "$ST" ] || { echo "serve-st-gate: SKIP (no checkpoint dir at $ST)"; exit 0; }
ADDR=127.0.0.1:8178
BASE=http://$ADDR
FAILS=0
PASS() { echo "  ok: $1"; }
FAIL() { echo "  FAIL: $1"; FAILS=$((FAILS+1)); }

[ -x target/release/memra-server ] || cargo build --release -p memra-server
[ -x target/release/run-gen ] || cargo build --release -p memra-engine --bin run-gen

PROMPT="What is the capital of France? Answer in one short sentence."
NGEN=64

echo "== serve-st-gate: checkpoint $ST =="

# ---- CLI arm: run-gen ST dir branch, chat-templated greedy decode ----
CLI_LOG=/tmp/serve-st-cli.log
MEMRA_CHAT=1 MEMRA_NGEN=$NGEN target/release/run-gen "$ST" --prompt "$PROMPT" \
  > "$CLI_LOG" 2>&1 \
  || { echo "run-gen failed; log tail:"; tail -5 "$CLI_LOG"; exit 1; }
CLI_TOKENS=$(grep '^tokens: ' "$CLI_LOG" | tail -1 | sed 's/^tokens: //')
[ -n "$CLI_TOKENS" ] || { echo "run-gen printed no token stream"; exit 1; }


# ---- server arm: native shape (no MEMRA_COMPAT) so /v1/completions returns raw ids.
# MEMRA_SERVE_SPEC=0 for the token-exactness arm: spec bursts emit ONE Token event id
# per burst, so the response `tokens` array is only the per-token stream on the plain
# tokenwise path (worker contract). Spec-vs-plain identity is gated separately below.
start_server() {  # $1 = extra env (e.g. "MEMRA_SERVE_SPEC=0"), sets SPID
  env $1 MEMRA_MODELS="st=$ST" MEMRA_ADDR=$ADDR target/release/memra-server \
    > /tmp/serve-st-server.log 2>&1 &
  SPID=$!
  # 27B-class ST dirs CPU-dequant ~29 GB at load (~13 min); 240s was calibrated on the
  # 4B default ckpt and times out spuriously (fp8ship-20260804 official-27B run).
  for _ in $(seq 600); do curl -sf $BASE/health >/dev/null 2>&1 && return 0; sleep 2; done
  echo "server did not come up; log tail:"; tail -5 /tmp/serve-st-server.log; return 1
}
stop_server() { kill "${SPID:-0}" 2>/dev/null; wait "${SPID:-0}" 2>/dev/null || true; }
trap stop_server EXIT
start_server "MEMRA_SERVE_SPEC=0" || exit 1

# 1. /models lists the ST checkpoint
curl -sf $BASE/models | grep -q '"st"' && PASS "/models lists the ST model" || FAIL "/models"

# 2. chat completion through the checkpoint's template: coherent text (Paris must appear
#    in content or the separated reasoning field), usage populated.
R=$(curl -sf -m 300 $BASE/v1/chat/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"st\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],
       \"max_tokens\":400,\"temperature\":0}")
echo "$R" > /tmp/serve-st-chat.json
echo "$R" | python3 -c '
import json,sys
r = json.load(sys.stdin)
m = r["choices"][0]["message"]
text = (m.get("content") or "") + (m.get("reasoning") or "")
assert text.strip(), "empty content+reasoning"
assert "paris" in text.lower(), f"incoherent answer: {text[:200]!r}"
assert r["usage"]["completion_tokens"] > 0, "no completion tokens"
' && PASS "chat completion coherent (Paris) via ST template" || FAIL "chat completion coherent"

# 3. exactness: server greedy token ids == CLI greedy token ids (same template render).
SRV_TOKENS=$(curl -sf -m 300 $BASE/v1/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"st\",\"prompt\":\"$PROMPT\",\"chat\":true,
       \"max_tokens\":$NGEN,\"temperature\":0}" \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["tokens"])')
python3 - "$CLI_TOKENS" "$SRV_TOKENS" <<'EOF'
import ast, sys
cli = ast.literal_eval(sys.argv[1])
srv = ast.literal_eval(sys.argv[2])
# Server Token events stop BEFORE the EOS id; the CLI stream includes it.
ok = cli == srv or (len(cli) == len(srv) + 1 and cli[:-1] == srv)
if not ok:
    div = next((i for i, (a, b) in enumerate(zip(cli, srv)) if a != b), min(len(cli), len(srv)))
    print(f"DIVERGE at token {div}: cli={cli[div:div+5]} srv={srv[div:div+5]} "
          f"(lens {len(cli)}/{len(srv)})")
    sys.exit(1)
print(f"identical {len(srv)} ids (cli {len(cli)} incl. trailing eos)"
      if len(cli) != len(srv) else f"identical {len(srv)} ids")
EOF
[ $? -eq 0 ] && PASS "CLI-vs-server greedy token streams identical" \
             || FAIL "CLI-vs-server exactness"

stop_server

# 4. ST serve-spec exactness (#68 closed, quarantine lifted): DEFAULT server (spec
#    bursts ON) vs the TOKENWISE serve oracle (SERVE_SPEC=0 SERVE_BATCH=0 — plain
#    decode_step in the same worker), 400-token window (pre-fix corruption onset ~250).
#    Tolerance: burst overshoot only — the shorter text must be a prefix of the longer.
grab_chat_text() {  # $1 = outfile
  curl -sf -m 300 $BASE/v1/chat/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"st\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],
         \"max_tokens\":400,\"temperature\":0}" \
    | python3 -c 'import json,sys; m=json.load(sys.stdin)["choices"][0]["message"]; sys.stdout.write((m.get("content") or "")+(m.get("reasoning") or ""))' > "$1"
}
start_server "MEMRA_SERVE_SPEC=0 MEMRA_SERVE_BATCH=0" || exit 1
grab_chat_text /tmp/serve-st-tokenwise.txt
stop_server
start_server "" || exit 1
if grep -q "QUARANTINED" /tmp/serve-st-server.log; then
  FAIL "stale quarantine notice at load (#68 is closed; lift regressed)"
else
  PASS "no quarantine notice (dir checkpoints spec-eligible)"
fi
grab_chat_text /tmp/serve-st-specdefault.txt
python3 - <<'EOF'
srv = open("/tmp/serve-st-specdefault.txt").read()
tok = open("/tmp/serve-st-tokenwise.txt").read()
assert srv.strip(), "empty default-server text"
assert tok.strip(), "empty tokenwise-oracle text"
short, long_ = (srv, tok) if len(srv) <= len(tok) else (tok, srv)
if not long_.startswith(short):
    n = len(short)
    d = next((j for j in range(n) if srv[j] != tok[j]), n)
    print(f"DIVERGE at char {d}: spec={srv[d:d+40]!r} tokenwise={tok[d:d+40]!r} "
          f"(lens {len(srv)}/{len(tok)})")
    raise SystemExit(1)
print(f"default (spec) text prefix-matches the tokenwise serve oracle "
      f"({len(srv)}/{len(tok)} chars)")
EOF
[ $? -eq 0 ] && PASS "default (spec) server == tokenwise serve oracle (400-tok window)" \
             || FAIL "default (spec) server diverged from the tokenwise oracle"
stop_server
trap - EXIT

echo "serve-st-gate: $FAILS failed"
[ $FAILS -eq 0 ]
