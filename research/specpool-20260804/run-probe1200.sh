#!/usr/bin/env bash
# F5 exactness gate: one deep greedy generation (n=1200) on the owner's config,
# run against a given binary; output text saved for byte-compare pre/post fix.
# Usage: run-probe1200.sh <tag> <binary>   (caller wraps in flock)
set -uo pipefail
cd "$(dirname "$0")"
TAG="$1"; BIN="$2"
DIR=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp
PORT=8102
LOG="server-probe-$TAG.log"

MEMRA_MODELS="qwen36-27b=$DIR/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf+$DIR/draft-daily-owntrim-nvfp4head-q4blk.gguf" \
MEMRA_ADDR="127.0.0.1:$PORT" \
MEMRA_API_KEY=aviary-local \
MEMRA_CTX=131072 \
MEMRA_MAX_SESSIONS=1 \
MEMRA_REUSE_POOL=1 \
MEMRA_PRIME_CHUNK=2048 \
"$BIN" > "$LOG" 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null; wait $PID 2>/dev/null' EXIT

for i in $(seq 1 120); do
  curl -sf -H 'Authorization: Bearer aviary-local' "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 $PID 2>/dev/null || { echo "server died:"; tail -5 "$LOG"; exit 1; }
  sleep 2
done

python3 - "$PORT" "probe1200-$TAG.txt" <<'EOF'
import json, sys, urllib.request
port, out = sys.argv[1], sys.argv[2]
prompt = ("Write a detailed technical essay on the design of a storage-to-GPU "
          "inference pipeline: mmap versus positioned reads, pinned host buffers, "
          "asynchronous prefetch, PCIe overlap, and KV-cache quantization. "
          "Be systematic and thorough.\n\nEssay:")
body = json.dumps({"model": "qwen36-27b", "prompt": prompt,
                   "max_tokens": 1200, "temperature": 0}).encode()
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=body,
    headers={"Content-Type": "application/json", "Authorization": "Bearer aviary-local"})
with urllib.request.urlopen(req, timeout=900) as r:
    resp = json.loads(r.read())
text = resp["choices"][0]["text"]
open(out, "w").write(text)
print(f"probe {out}: {len(text)} chars, {resp['usage']['completion_tokens']} tokens")
EOF
RC=$?
kill $PID 2>/dev/null; wait $PID 2>/dev/null
trap - EXIT
exit $RC
