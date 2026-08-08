#!/usr/bin/env python3
"""serve-proxy: least-outstanding reverse proxy with per-backend admission control
for memra-server replica fleets.

darklanes serving v1 (2026-08-01, round 4). Fronts N replicas on one port and routes
OpenAI-format requests (/v1/chat/completions, /v1/completions) to the healthy backend
with the fewest outstanding requests — but never more than --cap (default 8) per
backend. The cap is load-bearing: 8 sessions/replica is both the engine's exactness-
tier batch width AND the anti-thrash bound for two-replicas-per-GPU timeslice packing
(research/darklane-serving-20260801/ R2+R3). Beyond capacity, requests wait in a
bounded FIFO queue with a deadline; deadline/overflow -> 429 + Retry-After.

stdlib only, thread-per-request (queued waiters are blocked threads — bounded by
--queue-max). Streaming (SSE) responses are relayed chunk-by-chunk.

Usage:
  serve-proxy.py --port 8080 --backends http://127.0.0.1:8085,... \
     [--cap 8] [--queue-max 256] [--queue-deadline 30]

Endpoints:
  GET /health   -> {"status": "ok"|"no_backends", backends: [...]}  (200/503)
  GET /metrics  -> admission + latency counters (JSON):
       backends[].{url,healthy,outstanding,total,errors}
       queue.{depth,peak_depth,enqueued_total,rejected_429,wait_p50_s,wait_p95_s}
       requests.{total,ok,err_5xx,ttfb_p50_s,ttfb_p95_s,lat_p50_s,lat_p95_s}
  everything else is proxied.
"""

import argparse
import collections
import http.client
import json
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Connection-level failures = the backend process is gone (vs an HTTP error it
# answered with). These trip the passive circuit breaker.
CONN_ERRS = (ConnectionRefusedError, ConnectionResetError, BrokenPipeError,
             http.client.RemoteDisconnected)


def is_conn_error(e):
    if isinstance(e, CONN_ERRS):
        return True
    return isinstance(e, urllib.error.URLError) and \
        isinstance(getattr(e, "reason", None), CONN_ERRS)


def pct(sorted_vals, p):
    if not sorted_vals:
        return None
    k = min(len(sorted_vals) - 1, max(0, int(round(p / 100 * (len(sorted_vals) - 1)))))
    return sorted_vals[k]


class Window:
    """Rolling sample window for cheap percentiles."""

    def __init__(self, n=512):
        self.d = collections.deque(maxlen=n)
        self.lock = threading.Lock()

    def add(self, v):
        with self.lock:
            self.d.append(v)

    def p(self, p_):
        with self.lock:
            s = sorted(self.d)
        return pct(s, p_)


class Backend:
    def __init__(self, url):
        self.url = url.rstrip("/")
        self.outstanding = 0
        self.total = 0
        self.errors = 0
        self.healthy = True


class Router:
    """Least-outstanding routing + per-backend cap + bounded FIFO deadline queue."""

    def __init__(self, backends, cap, queue_max, queue_deadline):
        self.backends = [Backend(u) for u in backends]
        self.cap = cap
        self.queue_max = queue_max
        self.queue_deadline = queue_deadline
        self.lock = threading.Lock()
        self.slot_free = threading.Condition(self.lock)
        self.waiters = collections.deque()  # FIFO of ticket ids
        self.next_ticket = 0
        # metrics
        self.queue_depth = 0
        self.peak_depth = 0
        self.enqueued_total = 0
        self.rejected_429 = 0
        self.wait_w = Window()
        self.ttfb_w = Window()
        self.lat_w = Window()
        self.req_total = 0
        self.req_ok = 0
        self.err_5xx = 0

    def _pick_free(self):
        live = [b for b in self.backends if b.healthy and b.outstanding < self.cap]
        if not live:
            return None
        b = min(live, key=lambda b: b.outstanding)
        b.outstanding += 1
        b.total += 1
        return b

    def acquire(self):
        """Block until a backend slot is free (FIFO fair), or raise TimeoutError
        (deadline) / OverflowError (queue full). Returns the Backend."""
        t0 = time.monotonic()
        with self.lock:
            if not self.waiters:  # fast path: no one queued ahead
                b = self._pick_free()
                if b is not None:
                    self.wait_w.add(0.0)
                    return b
            if self.queue_depth >= self.queue_max:
                self.rejected_429 += 1
                raise OverflowError("admission queue full")
            ticket = self.next_ticket
            self.next_ticket += 1
            self.waiters.append(ticket)
            self.queue_depth += 1
            self.peak_depth = max(self.peak_depth, self.queue_depth)
            self.enqueued_total += 1
            deadline = t0 + self.queue_deadline
            try:
                while True:
                    # only the head of the FIFO may claim a slot (fairness)
                    if self.waiters and self.waiters[0] == ticket:
                        b = self._pick_free()
                        if b is not None:
                            self.waiters.popleft()
                            self.wait_w.add(time.monotonic() - t0)
                            self.slot_free.notify_all()  # let the new head re-check
                            return b
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        self.rejected_429 += 1
                        raise TimeoutError("admission deadline exceeded")
                    self.slot_free.wait(timeout=min(remaining, 1.0))
            finally:
                try:
                    self.waiters.remove(ticket)
                except ValueError:
                    pass  # already popped on success
                self.queue_depth -= 1

    def release(self, backend, ok):
        with self.lock:
            backend.outstanding -= 1
            if not ok:
                backend.errors += 1
            self.slot_free.notify_all()

    def health_loop(self, interval=2.0):
        # ROUTING ASKS READINESS, NOT LIVENESS (lane/serve-hardening 2026-08-06). A router's
        # question is "should I send traffic here?", which is /readyz: a backend that is
        # draining, still loading its weights, or whose worker died must leave rotation even
        # though the process is perfectly alive and /health deliberately answers 200 for a
        # drain. Using /health here would keep feeding a backend that is shutting down.
        # /health is kept as the fallback so a mixed-version fleet (an older replica without
        # /readyz, which answers 404) degrades to the previous behavior instead of marking
        # every backend DOWN.
        probe = "/readyz"
        while True:
            changed = False
            for b in self.backends:
                try:
                    with urllib.request.urlopen(b.url + probe, timeout=2) as r:
                        ok = r.status == 200
                except urllib.error.HTTPError as e:
                    if e.code == 404 and probe != "/health":
                        print(f"[proxy] {b.url} has no {probe} (pre-serve-hardening binary); "
                              f"falling back to /health for the whole fleet", flush=True)
                        probe = "/health"
                        continue
                    ok = False   # 503 = not ready (draining / loading / worker dead)
                except Exception:
                    ok = False
                if ok != b.healthy:
                    print(f"[proxy] backend {b.url} -> {'UP' if ok else 'DOWN'} "
                          f"({time.strftime('%H:%M:%S')})", flush=True)
                    b.healthy = ok
                    changed = True
            if changed:
                with self.lock:
                    self.slot_free.notify_all()
            time.sleep(interval)

    def snapshot(self):
        with self.lock:
            backends = [{"url": b.url, "healthy": b.healthy,
                         "outstanding": b.outstanding, "total": b.total,
                         "errors": b.errors} for b in self.backends]
            q = {"depth": self.queue_depth, "peak_depth": self.peak_depth,
                 "enqueued_total": self.enqueued_total,
                 "rejected_429": self.rejected_429, "cap_per_backend": self.cap,
                 "queue_max": self.queue_max,
                 "queue_deadline_s": self.queue_deadline}
            r = {"total": self.req_total, "ok": self.req_ok,
                 "err_5xx": self.err_5xx}
        q["wait_p50_s"] = self.wait_w.p(50)
        q["wait_p95_s"] = self.wait_w.p(95)
        r["ttfb_p50_s"] = self.ttfb_w.p(50)
        r["ttfb_p95_s"] = self.ttfb_w.p(95)
        r["lat_p50_s"] = self.lat_w.p(50)
        r["lat_p95_s"] = self.lat_w.p(95)
        return backends, q, r


ROUTER: Router = None  # set in main()
HOP_HEADERS = {"connection", "keep-alive", "transfer-encoding", "te", "trailer",
               "proxy-authorization", "proxy-authenticate", "upgrade", "host",
               "content-length"}


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _send_json(self, code, obj, extra_headers=()):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for k, v in extra_headers:
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            backends, _, _ = ROUTER.snapshot()
            any_up = any(b["healthy"] for b in backends)
            self._send_json(200 if any_up else 503,
                            {"status": "ok" if any_up else "no_backends",
                             "backends": backends})
            return
        if self.path == "/metrics":
            backends, queue, requests_ = ROUTER.snapshot()
            self._send_json(200, {"backends": backends, "queue": queue,
                                  "requests": requests_})
            return
        # pass-through GETs (/models): no admission, least-outstanding healthy backend
        with ROUTER.lock:
            live = [b for b in ROUTER.backends if b.healthy]
            backend = min(live, key=lambda b: b.outstanding) if live else None
        if backend is None:
            self._send_json(503, {"error": "no healthy backends"})
            return
        self._relay(backend, b"", admission=False)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        try:
            backend = ROUTER.acquire()
        except TimeoutError:
            self._send_json(429, {"error": "queue deadline exceeded"},
                            extra_headers=[("Retry-After", "5")])
            return
        except OverflowError:
            self._send_json(429, {"error": "queue full"},
                            extra_headers=[("Retry-After", "10")])
            return
        self._relay(backend, body, admission=True)

    def _relay(self, backend, body, admission):
        t0 = time.monotonic()
        with ROUTER.lock:
            ROUTER.req_total += 1
        url = backend.url + self.path
        req = urllib.request.Request(url, data=body if body else None,
                                     method=self.command)
        for k, v in self.headers.items():
            if k.lower() not in HOP_HEADERS:
                req.add_header(k, v)
        ok = False
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                ROUTER.ttfb_w.add(time.monotonic() - t0)
                self.send_response(resp.status)
                is_chunked = resp.headers.get("Transfer-Encoding", "").lower() == "chunked"
                for k, v in resp.headers.items():
                    if k.lower() not in HOP_HEADERS:
                        self.send_header(k, v)
                if is_chunked:
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    while True:
                        chunk = resp.read(65536)
                        if not chunk:
                            break
                        self.wfile.write(b"%x\r\n" % len(chunk))
                        self.wfile.write(chunk)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()  # SSE: relay promptly
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                else:
                    if resp.headers.get("Content-Length") is None:
                        self.send_header("Connection", "close")
                        self.close_connection = True
                    self.end_headers()
                    while True:
                        chunk = resp.read(65536)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                    self.wfile.flush()
            ok = True
            with ROUTER.lock:
                ROUTER.req_ok += 1
        except urllib.error.HTTPError as e:
            payload = e.read()
            self.send_response(e.code)
            self.send_header("Content-Type",
                             e.headers.get("Content-Type", "application/json"))
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            ok = True  # backend answered; not a routing failure
        except Exception as e:
            with ROUTER.lock:
                ROUTER.err_5xx += 1
                # PASSIVE CIRCUIT BREAKER (chaos check 2026-08-01): a killed backend's
                # slots free instantly, making it least-outstanding — without this, the
                # router preferentially feeds the corpse until the 2s active probe flips
                # it (measured: 100/768 fast-fail 502s). Break on the FIRST connection-
                # level failure; the health loop restores it when it answers again.
                if backend.healthy and is_conn_error(e):
                    backend.healthy = False
                    ROUTER.slot_free.notify_all()
                    print(f"[proxy] backend {backend.url} -> DOWN (passive: "
                          f"{type(e).__name__}, {time.strftime('%H:%M:%S')})", flush=True)
            try:
                self._send_json(502, {"error": f"backend {backend.url}: {e}"})
            except Exception:
                pass
        finally:
            ROUTER.lat_w.add(time.monotonic() - t0)
            if admission:
                ROUTER.release(backend, ok)


def main():
    global ROUTER
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--backends", required=True,
                    help="comma-separated backend base URLs")
    ap.add_argument("--cap", type=int, default=8,
                    help="max outstanding requests per backend (exactness-tier batch"
                         " width + timeslice anti-thrash bound)")
    ap.add_argument("--queue-max", type=int, default=256,
                    help="bounded admission queue size (429 beyond)")
    ap.add_argument("--queue-deadline", type=float, default=30.0,
                    help="max seconds a request may wait for a slot (429 beyond)")
    args = ap.parse_args()

    ROUTER = Router([u.strip() for u in args.backends.split(",") if u.strip()],
                    args.cap, args.queue_max, args.queue_deadline)
    threading.Thread(target=ROUTER.health_loop, daemon=True).start()

    # default socketserver backlog is 5 — a 64-way concurrent connect burst overflows
    # it and clients see ECONNRESET (measured: 10/256 resets at c=64 pre-fix).
    ThreadingHTTPServer.request_queue_size = 256
    srv = ThreadingHTTPServer((args.host, args.port), ProxyHandler)
    srv.daemon_threads = True
    print(f"[proxy] listening on http://{args.host}:{args.port} -> "
          f"{[b.url for b in ROUTER.backends]} cap={args.cap} "
          f"queue={args.queue_max}/{args.queue_deadline}s", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
