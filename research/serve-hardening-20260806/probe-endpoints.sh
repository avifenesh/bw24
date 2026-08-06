#!/usr/bin/env bash
# Live endpoint + taxonomy probe for lane/serve-hardening (G5 / G6 / G24).
#
# Boots a real memra-server and captures, from the wire:
#   * /health, /livez, /readyz DURING the weight load (must be 503 -> the whole point of G5)
#   * the same three once ready (200 + the worker block)
#   * the G6 taxonomy arms that are reachable without breaking the GPU: unknown model,
#     over-context prompt, dark-lane shed
#   * the drain split (/health 200 draining, /readyz 503) and the exit code
#   * the G24 watcher's own startup lines
#
# The taxonomy arms need help to be REACHABLE, and each knob is stated rather than hidden:
#   MEMRA_CTX=256 + "max_ctx":256  -> ctx_cap 256, so a ~400-token prompt is genuinely over it
#                                    (at the 262144 default the model context is unreachable).
#   MEMRA_LANE_MAX_HARVEST=0       -> the harvest lane's cap is 0, so an x-lane:harvest request
#                                    takes the deterministic capacity shed (worker.rs:1416-1422
#                                    `EngineError::rate_limit`) rather than depending on a live
#                                    SLO breach. The SLO-breach shed is the SAME error and the
#                                    same 429 path (worker.rs:1433-1436); forcing it needs
#                                    sustained interactive load, so the estimator arm is probed
#                                    informationally with /yield/metrics as evidence.
# Engine-fault (500) arms are NOT forced here: faking a CUDA fault would prove nothing about
# the real one. They are pinned by unit tests in main.rs.
#
# PHASE_LOADING is deliberately probed as connection-refused, not 503: main binds the listener
# AFTER the worker reports ready (main.rs — worker::spawn blocks), so no connection can be
# accepted during the FIRST load. Over HTTP, PHASE_LOADING is reachable only during a RESPAWN
# (socket already bound, worker reloading weights) — which is the case that matters, since that
# is when a bound port must NOT be reported ready. k8s and serve-fleet.sh treat refused and 503
# identically (probe failure), so first-load behavior is correct either way.
#
# Usage: research/serve-hardening-20260806/probe-endpoints.sh [port]
# Run under: flock /tmp/memra-5090.lock  (needs the GPU)
set -u
cd "$(git rev-parse --show-toplevel)"

PORT=${1:-8099}
MODEL=${MODEL:-/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf}
BIN=${BIN:-target/release/memra-server}
BASE=http://127.0.0.1:$PORT
SLOG=${SLOG:-research/serve-hardening-20260806/logs/endpoints-live-server.log}

# A port already in LISTEN would make every probe below measure SOMEONE ELSE's server (the
# first run of this probe hit the owner's llama-server on :8181 and "found" its 404s).
if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
  echo "FATAL: port $PORT is already listening — pick a free one"; exit 2
fi
[ -f "$MODEL" ] || { echo "FATAL: no model at $MODEL"; exit 2; }
[ -x "$BIN" ]   || { echo "FATAL: no server binary at $BIN"; exit 2; }

echo "== live endpoint + taxonomy probe =="
echo "commit  $(git rev-parse --short HEAD)  ($(git rev-parse --abbrev-ref HEAD))"
echo "binary  $BIN  ($(date -r "$BIN" +%FT%T))"
echo "model   $MODEL"
echo "rig     RTX 5090 Laptop, driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
echo "port    $PORT   server log -> $SLOG"

MEMRA_COMPAT=openai MEMRA_MODELS="probe=$MODEL" MEMRA_ADDR=127.0.0.1:$PORT \
  MEMRA_CTX=256 MEMRA_LANE_MAX_HARVEST=0 \
  "$BIN" > "$SLOG" 2>&1 &
SPID=$!
trap 'kill -9 $SPID 2>/dev/null' EXIT

probe() { # $1=path -> "<code> <body>"; empty body is printed as (empty), never as stale bytes
  printf '%-8s ' "$1"
  : > /tmp/probe-body
  curl -s -o /tmp/probe-body -w '%{http_code} ' -m 5 "$BASE$1" || printf '(connection refused/timeout) '
  if [ -s /tmp/probe-body ]; then cat /tmp/probe-body; else printf '(empty body)'; fi
  echo
}

echo
echo "--- [1] DURING the weight load: the port is not even bound yet ---"
echo "(main binds the listener only after worker::spawn returns ready, so the first load is"
echo " connection-refused, not 503 — same probe-failure verdict for k8s and serve-fleet.sh."
echo " Probed with no sleep: a page-cached 9B load here takes ~1.5s, so any wait wins the race"
echo " and reports a READY server, which is what an earlier version of this probe did.)"
for p in /health /livez /readyz; do probe $p; done
echo "    (if those answered 200, the load beat the first curl — the ordering claim is a code"
echo "     fact: TcpListener::bind runs after worker::spawn returns, main.rs.)"

echo
echo "--- [2] waiting for ready (curl -f: 200 only) ---"
t0=$(date +%s)
for _ in $(seq 240); do curl -sf -m 2 -o /dev/null "$BASE/readyz" && break; sleep 1; done
echo "ready after $(( $(date +%s) - t0 ))s"

echo
echo "--- [3] LOADED: /health, /livez, /readyz ---"
for p in /health /livez /readyz; do probe $p; done

echo
echo "--- [4] warm-up: one real interactive completion (also feeds the step estimator) ---"
curl -s -m 120 -o /tmp/probe-body -w 'status %{http_code}\n' "$BASE/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"probe","messages":[{"role":"user","content":"Say OK."}],"max_tokens":8,"temperature":0}'
head -c 400 /tmp/probe-body; echo

echo
echo "--- [5] G6: unknown model -> 400 invalid_request_error / code model_not_found ---"
curl -s -D /tmp/probe-hdr -o /tmp/probe-body -m 30 "$BASE/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"nope","messages":[{"role":"user","content":"hi"}],"max_tokens":4}'
grep -iE '^HTTP/|^x-should-retry|^retry-after' /tmp/probe-hdr; cat /tmp/probe-body; echo

echo
echo "--- [6] G6: prompt over the context cap -> 400 / code context_length_exceeded ---"
python3 - "$BASE" <<'PY'
import json, sys, urllib.request, urllib.error
body = {"model":"probe","max_ctx":256,"max_tokens":8,
        "messages":[{"role":"user","content":"word "*400}]}
req = urllib.request.Request(sys.argv[1]+"/v1/chat/completions",
                            data=json.dumps(body).encode(),
                            headers={"Content-Type":"application/json"})
try:
    urllib.request.urlopen(req, timeout=120); print("NO ERROR (unexpected)")
except urllib.error.HTTPError as e:
    print("HTTP", e.code, "| x-should-retry:", e.headers.get("x-should-retry"),
          "| retry-after:", e.headers.get("retry-after"))
    print(e.read().decode()[:400])
PY

echo
echo "--- [6b] G6: bad x-lane -> 400 / code invalid_lane (was a BARE-STRING error body) ---"
curl -s -D /tmp/probe-hdr -o /tmp/probe-body -m 30 "$BASE/v1/chat/completions" \
  -H 'Content-Type: application/json' -H 'x-lane: turbo' \
  -d '{"model":"probe","messages":[{"role":"user","content":"hi"}],"max_tokens":4}'
grep -iE '^HTTP/|^x-should-retry|^retry-after' /tmp/probe-hdr; cat /tmp/probe-body; echo

echo
echo "--- [7] G6: dark-lane shed -> 429 rate_limit_error + Retry-After (x-lane: harvest) ---"
echo "(non-streaming: peek_shed resolves the verdict BEFORE headers, so this is a real"
echo " pre-header 429 and not a mid-stream death — the OpenRouter-uptime point.)"
curl -s -D /tmp/probe-hdr -o /tmp/probe-body -m 60 "$BASE/v1/chat/completions" \
  -H 'Content-Type: application/json' -H 'x-lane: harvest' \
  -d '{"model":"probe","messages":[{"role":"user","content":"hi"}],"max_tokens":4}'
grep -iE '^HTTP/|^retry-after|^x-should-retry' /tmp/probe-hdr; head -c 300 /tmp/probe-body; echo
echo "    same shed on the STREAMING surface (must also be a pre-header status, not an SSE chunk):"
curl -s -o /dev/null -w '    stream request -> HTTP %{http_code}\n' -m 60 \
  "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -H 'x-lane: harvest' \
  -d '{"model":"probe","messages":[{"role":"user","content":"hi"}],"max_tokens":4,"stream":true}'
echo "    lane counters (/yield/metrics):"
curl -s -m 5 "$BASE/yield/metrics" | head -c 400; echo

echo
echo "--- [8] streaming still works end to end (mid-stream surface unchanged on success) ---"
curl -s -N -m 60 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"probe","messages":[{"role":"user","content":"Count: 1 2"}],"max_tokens":8,"stream":true,"temperature":0}' \
  | tail -3

echo
echo "--- [9] drain: SIGTERM keeps in-flight work alive and takes the box out of rotation ---"
echo "MEASURED, correcting the obvious guess: axum stops accepting only when its shutdown FUTURE"
echo "resolves, and memra's future IS the drain loop — so throughout the drain window the"
echo "listener keeps accepting, on pooled keep-alive connections AND on fresh ones. That is what"
echo "makes the whole split observable on the wire: /health + /livez answer 200"
echo "status=\"draining\" (a drain is a healthy deliberate shutdown; 503 here would invite a"
echo "supervisor SIGKILL mid-stream), /readyz answers 503 (out of rotation), and a NEW completion"
echo "is refused 503 + Retry-After while the in-flight one runs to completion."
# A long generation holds the drain window open: on an idle server the drain finishes in 0.0s
# and there is nothing to observe (measured: "drain complete in 0.0s" on the first attempt).
curl -s -o /tmp/probe-longgen -m 180 "$BASE/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"probe","messages":[{"role":"user","content":"Write a long story about a lighthouse keeper."}],"max_tokens":900,"max_ctx":2048,"temperature":0}' &
LONG=$!
sleep 3
python3 - "$PORT" "$SPID" <<'PY'
# Probed on a POOLED keep-alive connection (the load balancer's own shape) opened before the
# SIGTERM, and then on a FRESH connection, so "is the drain observable at all?" is answered for
# both client shapes rather than assumed for either.
import http.client, json, os, signal, sys, time
port, spid = int(sys.argv[1]), int(sys.argv[2])
pooled = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
def get(conn, path):
    conn.request("GET", path, headers={"Connection": "keep-alive"})
    r = conn.getresponse(); return r.status, r.read().decode()
print("    pre-SIGTERM  /health  %d %s" % get(pooled, "/health"))
os.kill(spid, signal.SIGTERM)
time.sleep(0.5)
for p in ("/health", "/livez", "/readyz"):
    try:
        print("    pooled       %-8s %d %s" % ((p,) + get(pooled, p)))
    except Exception as e:
        print("    pooled       %-8s connection closed (%s)" % (p, type(e).__name__))
        break
for p in ("/health", "/readyz"):
    try:
        print("    fresh conn   %-8s %d %s" % ((p,) + get(
            http.client.HTTPConnection("127.0.0.1", port, timeout=5), p)))
    except Exception as e:
        print("    fresh conn   %-8s refused (%s)" % (p, type(e).__name__))
# A NEW completion during the drain must be REFUSED with a retry window, not accepted into a
# process that is about to exit (this is the request-path half of the drain contract).
body = json.dumps({"model":"probe","max_tokens":4,
                   "messages":[{"role":"user","content":"hi"}]}).encode()
try:
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request("POST", "/v1/chat/completions", body=body,
              headers={"Content-Type":"application/json"})
    r = c.getresponse()
    print("    new completion during the drain -> %d | retry-after: %s\n      %s"
          % (r.status, r.headers.get("retry-after"), r.read().decode()[:200]))
except Exception as e:
    print("    new completion during the drain -> refused (%s)" % type(e).__name__)
PY
wait $LONG 2>/dev/null
echo -n "    the in-flight generation survived the drain: "
python3 -c "
import json
try:
    d=json.load(open('/tmp/probe-longgen'))
    print('yes —', d['usage']['completion_tokens'], 'tokens,', d['choices'][0]['finish_reason'])
except Exception as e: print('NO —', type(e).__name__, str(e)[:80])
"
wait $SPID; RC=$?
echo "server exit code: $RC   (0 = clean drain; 70 = worker unrecoverable; 1 = bad config)"

echo
echo "--- [10] G24 watcher + sd_notify lines from the server log ---"
grep -a 'gpu-watch\|sd-notify\|drain' "$SLOG" || true
echo
echo "PROBE DONE"
