#!/usr/bin/env python3
"""serve-proxy: minimal least-outstanding-requests reverse proxy for memra-server replicas.

darklanes serving v1 (2026-08-01): replica-per-GPU multi-user serving. Each memra-server
process owns one GPU via CUDA_VISIBLE_DEVICES (Engine::new(0) inside); this proxy fronts N
replicas on one port and routes each incoming OpenAI-format request (/v1/chat/completions,
/v1/completions) to the replica with the fewest outstanding requests (ties -> lowest index).

stdlib only (http.server + urllib), thread-per-request — the replicas do millisecond-scale
token work; a Python thread that spends its life blocked in a socket read is fine at the
concurrency this v1 targets (<= 64). Streaming (SSE) responses are relayed chunk-by-chunk.

Usage:
  python3 serve-proxy.py --port 8080 --backends http://127.0.0.1:8085,http://127.0.0.1:8086,http://127.0.0.1:8087

Health: GET /health returns 200 with per-backend status + outstanding counts (JSON).
Backends failing their health probe are pulled from rotation until they pass again.
"""

import argparse
import json
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Backend:
    def __init__(self, url: str):
        self.url = url.rstrip("/")
        self.outstanding = 0
        self.total = 0
        self.errors = 0
        self.healthy = True
        self.lock = threading.Lock()

    def acquire(self):
        with self.lock:
            self.outstanding += 1
            self.total += 1

    def release(self, ok: bool):
        with self.lock:
            self.outstanding -= 1
            if not ok:
                self.errors += 1


class Router:
    def __init__(self, backends):
        self.backends = [Backend(u) for u in backends]
        self.lock = threading.Lock()

    def pick(self):
        with self.lock:
            live = [b for b in self.backends if b.healthy]
            if not live:
                return None
            return min(live, key=lambda b: b.outstanding)

    def health_loop(self, interval: float = 2.0):
        while True:
            for b in self.backends:
                try:
                    with urllib.request.urlopen(b.url + "/health", timeout=2) as r:
                        ok = r.status == 200
                except Exception:
                    ok = False
                if ok != b.healthy:
                    print(f"[proxy] backend {b.url} -> {'UP' if ok else 'DOWN'}", flush=True)
                b.healthy = ok
            time.sleep(interval)


ROUTER: Router = None  # set in main()
HOP_HEADERS = {"connection", "keep-alive", "transfer-encoding", "te", "trailer",
               "proxy-authorization", "proxy-authenticate", "upgrade", "host",
               "content-length"}


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quiet per-request lines
        pass

    def _send_json(self, code: int, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            snap = [{"url": b.url, "healthy": b.healthy, "outstanding": b.outstanding,
                     "total": b.total, "errors": b.errors} for b in ROUTER.backends]
            any_up = any(b.healthy for b in ROUTER.backends)
            self._send_json(200 if any_up else 503,
                            {"status": "ok" if any_up else "no_backends", "backends": snap})
            return
        # pass through GET /models etc. to the least-loaded backend
        self._forward(b"")

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        self._forward(body)

    def _forward(self, body: bytes):
        backend = ROUTER.pick()
        if backend is None:
            self._send_json(503, {"error": "no healthy backends"})
            return
        url = backend.url + self.path
        req = urllib.request.Request(url, data=body if body else None,
                                     method=self.command)
        for k, v in self.headers.items():
            if k.lower() not in HOP_HEADERS:
                req.add_header(k, v)
        backend.acquire()
        ok = False
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                self.send_response(resp.status)
                is_chunked = resp.headers.get("Transfer-Encoding", "").lower() == "chunked"
                for k, v in resp.headers.items():
                    if k.lower() not in HOP_HEADERS:
                        self.send_header(k, v)
                if not is_chunked and resp.headers.get("Content-Length") is None:
                    # no length and not chunked upstream: close-delimit
                    self.send_header("Connection", "close")
                    self.end_headers()
                    while True:
                        chunk = resp.read(65536)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                    self.wfile.flush()
                    ok = True
                    self.close_connection = True
                    return
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
                    self.end_headers()
                    while True:
                        chunk = resp.read(65536)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                    self.wfile.flush()
                ok = True
        except urllib.error.HTTPError as e:
            payload = e.read()
            self.send_response(e.code)
            self.send_header("Content-Type", e.headers.get("Content-Type", "application/json"))
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            ok = True  # backend answered; not a routing failure
        except Exception as e:
            try:
                self._send_json(502, {"error": f"backend {backend.url}: {e}"})
            except Exception:
                pass
        finally:
            backend.release(ok)


def main():
    global ROUTER
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--backends", required=True,
                    help="comma-separated backend base URLs")
    args = ap.parse_args()

    ROUTER = Router([u.strip() for u in args.backends.split(",") if u.strip()])
    t = threading.Thread(target=ROUTER.health_loop, daemon=True)
    t.start()

    # Default socketserver backlog is 5 — a 64-way concurrent connect burst overflows it
    # and clients see ECONNRESET (measured: 10/256 resets at c=64 before this bump).
    ThreadingHTTPServer.request_queue_size = 256
    srv = ThreadingHTTPServer((args.host, args.port), ProxyHandler)
    srv.daemon_threads = True
    print(f"[proxy] listening on http://{args.host}:{args.port} -> "
          f"{[b.url for b in ROUTER.backends]}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
