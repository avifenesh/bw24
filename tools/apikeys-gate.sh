#!/usr/bin/env bash
# API-key gate (lane/api-keys, 2026-08-05): boots memra-server with a generated
# multi-tenant keyring and runs research/apikeys-20260805/apikey_gate.py — auth
# refusals (401/403), single-key back-compat, the two-tenant cache-isolation proof
# (cache-hit oracle), per-tenant rate-limit headers, batch-class lane law, hot revoke.
#
# Usage: tools/apikeys-gate.sh [model.gguf] [out_dir]
# GPU: single short-lived server; callers hold /tmp/gpu5090.lock around this script.
set -uo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:-/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf}"
OUT="${2:-research/apikeys-20260805}"
[ -f "$MODEL" ] || { echo "apikeys-gate: SKIP (no model at $MODEL)"; exit 0; }
mkdir -p "$OUT"
ADDR=127.0.0.1:8178
BASE=http://$ADDR
KEYS=$(mktemp /tmp/apikeys-gate-XXXX.toml)
rm -f "$KEYS" # --gen-key creates it

[ -x target/release/memra-server ] || cargo build --release -p memra-server

BIN=target/release/memra-server
# Keyring: acme x2 interactive (one to hot-revoke later), blue interactive (the
# cross-tenant counterparty), bulk (batch class, rate_limit 2), dead (revoked pre-boot).
KEY_A1=$("$BIN" --gen-key acme --keys "$KEYS") || exit 1
KEY_A2=$("$BIN" --gen-key acme --keys "$KEYS") || exit 1
KEY_B=$("$BIN" --gen-key blue --keys "$KEYS") || exit 1
KEY_BULK=$("$BIN" --gen-key bulk --lane batch --rate-limit 2 --keys "$KEYS") || exit 1
KEY_DEAD=$("$BIN" --gen-key dead --keys "$KEYS") || exit 1
"$BIN" --revoke-key "mk-dead-" --keys "$KEYS" || exit 1
SINGLE=daily-driver-key

# Bulk tier (spec off — the tier the prefix cache serves), prefix cache large enough
# that LRU eviction can't masquerade as isolation. Single key SET alongside the ring
# (the composition law under test).
MEMRA_API_KEYS="$KEYS" MEMRA_API_KEY="$SINGLE" MEMRA_COMPAT=openai \
MEMRA_MODELS="gate=$MODEL" MEMRA_ADDR=$ADDR MEMRA_SERVE_SPEC=0 \
MEMRA_PREFIX_CACHE_MB=1024 \
  "$BIN" > "$OUT/server-apikeys.log" 2>&1 &
SPID=$!
trap 'kill $SPID 2>/dev/null; wait $SPID 2>/dev/null; rm -f "$KEYS"' EXIT
for _ in $(seq 120); do
  curl -sf $BASE/health >/dev/null 2>&1 && break; sleep 2
done
curl -sf $BASE/health >/dev/null || { echo "server did not come up"; tail -5 "$OUT/server-apikeys.log"; exit 1; }

A2_PREFIX="${KEY_A2:0:20}" # mk-acme-<12 hex> = the unambiguous revoke handle
python3 "$OUT/apikey_gate.py" --base $BASE --model gate --out "$OUT" \
  --key-a1 "$KEY_A1" --key-a2 "$KEY_A2" --key-b "$KEY_B" --key-bulk "$KEY_BULK" \
  --key-revoked "$KEY_DEAD" --single "$SINGLE" \
  --revoke-cmd "$BIN --revoke-key $A2_PREFIX --keys $KEYS"
RC=$?
grep '\[meter\]' "$OUT/server-apikeys.log" | head -20 > "$OUT/meter-lines-sample.log"
exit $RC
