#!/usr/bin/env bash
# serve-stress-gate — c=64 concurrency robustness gate (lane/serving-density, 2026-08-06).
#
# THE SHAPE: 64 clients x short requests (128-256 generated tokens), staggered arrival,
# streaming — the robustness cell mistral.rs died on (6/69 finished at 64 clients). memra's
# serving battery had never proven the tick loop past c=8/16; this gate pins the contract:
#
#   1. ALL requests complete: HTTP 200, no hangs (per-request timeout), no drops.
#   2. Every stream is WELL-FORMED: >=1 data: chunk, a finish_reason in the final delta,
#      and the data: [DONE] terminator.
#   3. The worker survives: server process alive after the burst, and the server log
#      carries no panic / CUDA_ERROR / out-of-memory line (failure causes are quoted,
#      never inferred — the log IS the receipt).
#   4. p50/p95 completion wall time + TTFB recorded per run (INFORMATIONAL, not asserted —
#      this is a robustness gate, not a perf cell; perf lives in the tracked boards).
#
# Concurrency mechanics under test: MEMRA_MAX_SESSIONS default (64) admits everything;
# the VRAM-aware admission gate (worker.rs: free < 2x observed session cost -> FIFO wait)
# and the F5 right-size ladder are exactly the seams a 64-client burst stresses on a 24GB
# card (64 x ~233MiB q9 spec sessions + 5.8GiB weights does NOT all fit — the gate proves
# the admission wait degrades gracefully instead of OOMing or hanging).
#
# NOT flock-wrapped: callers own the GPU lock (fast-gate's lockrun / local-ci's window
# discipline) — self-locking here would self-deadlock under fast-gate.
#
# Usage: tools/serve-stress-gate.sh [model.gguf [draft.gguf [n_clients]]]
# Exits nonzero on any failed assertion. SKIPs (exit 0, "serve-stress-gate: SKIP" line —
# fast-gate's verdict-word contract) when the model artifact is absent.
set -uo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:-/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf}"
DRAFT="${2:-/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/draft-9b-owntrim-nvfp4head-q4blk.gguf}"
NCLIENTS="${3:-64}"
[ -f "$MODEL" ] || { echo "serve-stress-gate: SKIP (no model at $MODEL)"; exit 0; }
ADDR=127.0.0.1:8179
BASE=http://$ADDR
SLOG="${MEMRA_STRESS_LOG:-/tmp/serve-stress-gate.log}"
ROWS="${MEMRA_STRESS_ROWS:-/tmp/serve-stress-rows.jsonl}"

[ -x target/release/memra-server ] || cargo build --release -p memra-server || exit 1

MODELSPEC="stress=$MODEL"
[ -f "$DRAFT" ] && MODELSPEC="stress=$MODEL+$DRAFT"

MEMRA_COMPAT=openai MEMRA_MODELS="$MODELSPEC" MEMRA_ADDR=$ADDR MEMRA_CTX=8192 \
    target/release/memra-server > "$SLOG" 2>&1 &
SPID=$!
stop_server() { kill "${SPID:-0}" 2>/dev/null; wait "${SPID:-0}" 2>/dev/null || true; }
trap stop_server EXIT
up=0
for _ in $(seq 120); do curl -sf $BASE/health >/dev/null 2>&1 && { up=1; break; }; sleep 2; done
[ "$up" = 1 ] || { echo "serve-stress-gate: FAIL (server did not come up); log tail:"; tail -5 "$SLOG"; exit 1; }

echo "== serve-stress-gate: c=$NCLIENTS staggered short streams (128-256 tok) =="
python3 - "$BASE" "$NCLIENTS" "$ROWS" <<'PYEOF'
import json, random, statistics, sys, threading, time, urllib.request

base, n, rows_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
STAGGER_S = 0.05          # 64 clients ramp over ~3.2s — arrival stagger, not a thundering herd
TIMEOUT_S = 900           # per-request hang bound (the "no hangs" assertion)
random.seed(20260806)

PROMPTS = [
    "List three failure modes of a GPU serving stack under high concurrency and one mitigation for each.",
    "Explain the difference between p50 and p99 latency to a new engineer in four sentences.",
    "Write a short shell one-liner to count open TCP connections, then explain it.",
    "Summarize what a KV cache stores during LLM decoding and why it grows with context.",
    "Name four things a load balancer must handle when a backend dies mid-stream.",
    "Describe speculative decoding in three sentences.",
]

results = []
lock = threading.Lock()

def worker(i):
    body = {
        "model": "stress",
        "messages": [{"role": "user", "content":
                      f"[client {i} nonce {time.time_ns()}] " + PROMPTS[i % len(PROMPTS)]}],
        "max_tokens": random.randint(128, 256),
        "temperature": 0.7,
        "seed": 1000 + i,
        "stream": True,
    }
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    row = {"i": i, "ok": False, "chunks": 0, "finish_reason": None, "done": False}
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
            ttfb = None
            for raw in r:
                if ttfb is None:
                    ttfb = time.time() - t0
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    row["done"] = True
                    break
                row["chunks"] += 1
                try:
                    ch = json.loads(payload)
                    if "error" in ch:
                        # server-side error surfaced as the final data chunk (the
                        # OpenAI-compat error path) — quote it, it IS the receipt.
                        err = ch["error"]
                        row["server_error"] = (err.get("message", str(err))
                                               if isinstance(err, dict) else str(err))[:300]
                        continue
                    fr = ch.get("choices", [{}])[0].get("finish_reason")
                    if fr:
                        row["finish_reason"] = fr
                except json.JSONDecodeError:
                    row["bad_json"] = payload[:120]
            row["ttfb_s"] = round(ttfb, 3) if ttfb is not None else None
        row["wall_s"] = round(time.time() - t0, 3)
        row["ok"] = (row["chunks"] > 0 and row["done"]
                     and row["finish_reason"] in ("stop", "length")
                     and "bad_json" not in row and "server_error" not in row)
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"[:300]
        row["wall_s"] = round(time.time() - t0, 3)
    with lock:
        results.append(row)

threads = []
for i in range(n):
    t = threading.Thread(target=worker, args=(i,))
    t.start()
    threads.append(t)
    time.sleep(STAGGER_S)
for t in threads:
    t.join()

with open(rows_path, "a") as f:
    for r in sorted(results, key=lambda r: r["i"]):
        f.write(json.dumps(r) + "\n")

ok = [r for r in results if r["ok"]]
bad = [r for r in results if not r["ok"]]
walls = sorted(r["wall_s"] for r in results if "wall_s" in r)
ttfbs = sorted(r["ttfb_s"] for r in results if r.get("ttfb_s") is not None)
def pct(v, p):
    return v[min(len(v) - 1, int(p * len(v)))] if v else float("nan")
print(f"completed {len(ok)}/{n}; wall p50={pct(walls, 0.50):.1f}s "
      f"p95={pct(walls, 0.95):.1f}s max={walls[-1] if walls else float('nan'):.1f}s; "
      f"ttfb p50={pct(ttfbs, 0.50):.2f}s p95={pct(ttfbs, 0.95):.2f}s  (informational)")
for r in bad[:8]:
    print(f"  BAD i={r['i']}: err={r.get('error')} server_error={r.get('server_error')} "
          f"chunks={r['chunks']} finish_reason={r['finish_reason']} done={r['done']} "
          f"bad_json={r.get('bad_json', '')}")
sys.exit(0 if len(ok) == n else 1)
PYEOF
CLIENT_RC=$?

# worker survival: the process must still be alive and healthy after the burst
ALIVE=0
kill -0 "$SPID" 2>/dev/null && curl -sf $BASE/health >/dev/null 2>&1 && ALIVE=1

# quoted-failure scan (never inferred): panic / CUDA error / OOM lines in the server log
BADLOG=$(grep -aE "panicked at|CUDA_ERROR|out of memory|SIGSEGV|memory allocation.*failed" "$SLOG" | head -5)

FAILS=0
[ "$CLIENT_RC" -eq 0 ] || { echo "  FAIL: not all $NCLIENTS requests completed well-formed"; FAILS=$((FAILS+1)); }
[ "$ALIVE" = 1 ] || { echo "  FAIL: server dead or unhealthy after burst"; FAILS=$((FAILS+1)); }
if [ -n "$BADLOG" ]; then
    echo "  FAIL: server log carries failure lines:"; echo "$BADLOG" | sed 's/^/      /'
    FAILS=$((FAILS+1))
fi

if [ "$FAILS" -eq 0 ]; then
    echo "serve-stress-gate: ALL GREEN (c=$NCLIENTS complete, streams well-formed, worker alive, log clean)"
    exit 0
else
    echo "serve-stress-gate: FAIL ($FAILS assertion(s)); server log: $SLOG, rows: $ROWS"
    exit 1
fi
