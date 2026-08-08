#!/usr/bin/env python3
"""load-serve: concurrent OpenAI-format load harness for memra-server / serve-proxy.

darklanes serving v1 (2026-08-01). Fires `--concurrency` simultaneous workers, each looping
POST /v1/chat/completions (streaming off, temperature 0 by default off — load prompts use
temperature 0.7 + per-request seed so batched decode exercises realistic divergent sequences;
pass --greedy for temperature=0 determinism checks). Prompt is ~200 tokens, max_tokens=128.

Measures per-request wall latency and completion_tokens from the server's usage block.
Reports aggregate output tok/s (sum completion_tokens / wall window), p50/p95 latency,
error count. Emits one JSON line per load point to --out (append), one line per request
with --per-request.

dl-metering additions (2026-08-02):
  --lane interactive|judge|harvest   sends the x-lane QoS header
  --tenant NAME                      sends x-tenant-id (metering identity)
  --api-key KEY                      sends Authorization: Bearer KEY
  --duration SECS                    run for a wall window instead of a request count
                                     (workers loop until the deadline; overlapping-class
                                     contention runs use this)
  429 responses are counted separately as `n_shed` (the lane admission signal, not an
  error); --retry-shed retries after Retry-After (default 0.5s) so a batch-class worker
  keeps pressure on the gate the way a real harvest client would.
  Per-request rows carry the server-assigned request id (`rid`) + prompt_tokens from the
  usage block — the reconciliation join keys for tools/usage-report.py --reconcile.

Usage:
  python3 load-serve.py --base http://127.0.0.1:8085 --concurrency 8 --requests 32 \
      --model qwen --out points.jsonl --label single-8085
"""

import argparse
import json
import statistics
import threading
import time
import urllib.error
import urllib.request

# ~200 tokens of prompt: a fixed instruction + repeated filler clause (repeats tokenize
# steadily, so the prompt length is stable across replicas and runs).
FILLER = ("The quick brown fox jumps over the lazy dog while the seasoned engineer "
          "measures throughput, latency, and saturation across every replica. ")
PROMPT = ("Summarize the operational state of a GPU serving cluster in exactly three "
          "sentences, then list four risks. Context follows. " + FILLER * 8)


def one_request(base, model, max_tokens, greedy, seed, timeout, headers=None, stream=False):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0.0 if greedy else 0.7,
        "seed": seed,
        "stream": bool(stream),
    }
    if stream:
        # ask for the usage block on the final SSE frame so token counts stay server-authoritative
        body["stream_options"] = {"include_usage": True}
    h = {"Content-Type": "application/json"}
    if headers:
        h.update(headers)
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers=h)
    t0 = time.monotonic()
    try:
        if stream:
            return _stream_request(req, timeout, t0)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.load(r)
        dt = time.monotonic() - t0
        usage = data.get("usage", {})
        return {
            "ok": True,
            "latency_s": dt,
            "rid": data.get("id"),
            "completion_tokens": usage.get("completion_tokens", 0),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "finish_reason": data["choices"][0].get("finish_reason"),
            "text": data["choices"][0]["message"]["content"],
        }
    except Exception as e:
        detail = ""
        shed = False
        retry_after = 0.5
        if isinstance(e, urllib.error.HTTPError):
            shed = e.code == 429
            try:
                retry_after = float(e.headers.get("Retry-After", retry_after))
            except (TypeError, ValueError):
                pass
            try:
                detail = e.read()[:300].decode(errors="replace")
            except Exception:
                pass
        return {"ok": False, "shed": shed, "retry_after": retry_after,
                "latency_s": time.monotonic() - t0,
                "error": f"{type(e).__name__}: {e} {detail}".strip()}


def _stream_request(req, timeout, t0):
    """SSE read that timestamps the FIRST content-bearing frame — the TTFT observable.

    TTFT needs streaming: with `stream: False` the only timestamp a client gets is the whole
    response, so a scheduler change that delays the first token but not the last is invisible.
    A frame counts as first-token only if it carries actual text — role-only openers and empty
    deltas are protocol overhead, not the user-visible first token. `reasoning` counts: on a
    thinking model that IS the visible stream (a content-only reader would time the wrong frame,
    or never fire at all).
    """
    ttft = None
    ntok = 0
    usage = {}
    rid = None
    finish = None
    chunks = []
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                ev = json.loads(payload)
            except json.JSONDecodeError:
                continue
            rid = ev.get("id") or rid
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                piece = (d.get("content") or "") + (d.get("reasoning") or "")
                if piece:
                    if ttft is None:
                        ttft = time.monotonic() - t0
                    ntok += 1
                    chunks.append(piece)
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    dt = time.monotonic() - t0
    return {
        "ok": True,
        "latency_s": dt,
        "ttft_s": ttft,
        # decode-only rate: excludes prefill, so it isolates the steady-state token cadence
        "decode_s": (dt - ttft) if ttft is not None else None,
        "rid": rid,
        # server usage when present, else the counted content frames (labelled either way)
        "completion_tokens": usage.get("completion_tokens", ntok),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "usage_from_server": bool(usage),
        "finish_reason": finish,
        "text": "".join(chunks),
    }


def run_point(base, model, concurrency, requests, max_tokens, greedy, timeout,
              headers=None, duration=None, retry_shed=False, stream=False):
    results = []
    rlock = threading.Lock()
    idx = {"n": 0}
    deadline = (time.monotonic() + duration) if duration else None

    def worker(wid):
        while True:
            if deadline is not None:
                if time.monotonic() >= deadline:
                    return
                with rlock:
                    my = idx["n"]
                    idx["n"] += 1
            else:
                with rlock:
                    if idx["n"] >= requests:
                        return
                    my = idx["n"]
                    idx["n"] += 1
            while True:
                res = one_request(base, model, max_tokens, greedy, seed=1000 + my,
                                  timeout=timeout, headers=headers, stream=stream)
                if res.get("shed") and retry_shed and \
                        (deadline is None or time.monotonic() < deadline):
                    with rlock:
                        results.append({**res, "worker": wid, "req_index": my})
                    time.sleep(res.get("retry_after", 0.5))
                    continue
                break
            res["worker"] = wid
            res["req_index"] = my
            with rlock:
                results.append(res)

    t0 = time.monotonic()
    threads = [threading.Thread(target=worker, args=(w,)) for w in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.monotonic() - t0

    oks = [r for r in results if r["ok"]]
    sheds = [r for r in results if not r["ok"] and r.get("shed")]
    errs = [r for r in results if not r["ok"] and not r.get("shed")]
    lats = sorted(r["latency_s"] for r in oks)
    tot_completion = sum(r["completion_tokens"] for r in oks)
    tot_prompt = sum(r["prompt_tokens"] for r in oks)

    def pct(p):
        if not lats:
            return None
        k = min(len(lats) - 1, max(0, int(round(p / 100 * (len(lats) - 1)))))
        return lats[k]

    return {
        "wall_s": wall,
        "n_ok": len(oks),
        "n_shed": len(sheds),
        "n_err": len(errs),
        "completion_tokens_total": tot_completion,
        "prompt_tokens_total": tot_prompt,
        "agg_tok_s": tot_completion / wall if wall > 0 else 0.0,
        "req_per_s": len(oks) / wall if wall > 0 else 0.0,
        "lat_p50_s": pct(50),
        "lat_p95_s": pct(95),
        "lat_mean_s": statistics.mean(lats) if lats else None,
        "lat_max_s": lats[-1] if lats else None,
        "errors_sample": [e["error"] for e in errs[:3]],
        **_ttft_summary(oks),
    }, results


def _ttft_summary(oks):
    """TTFT percentiles, present only in --stream mode (absent, not zero, otherwise)."""
    tt = sorted(r["ttft_s"] for r in oks if r.get("ttft_s") is not None)
    if not tt:
        return {}

    def q(p):
        k = min(len(tt) - 1, max(0, int(round(p / 100 * (len(tt) - 1)))))
        return tt[k]

    return {"ttft_p50_s": q(50), "ttft_p95_s": q(95),
            "ttft_mean_s": statistics.mean(tt), "ttft_max_s": tt[-1], "n_ttft": len(tt)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="server base URL")
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--concurrency", type=int, required=True)
    ap.add_argument("--requests", type=int, default=None,
                    help="total requests (default 4x concurrency, min 8)")
    ap.add_argument("--duration", type=float, default=None,
                    help="run for SECS instead of a request count")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--greedy", action="store_true", help="temperature=0")
    ap.add_argument("--stream", action="store_true",
                    help="SSE mode: adds per-request ttft_s + decode_s and ttft p50/p95 to the "
                         "point (TTFT is unobservable without streaming). Default off so "
                         "existing lanes' numbers stay comparable.")
    ap.add_argument("--lane", default=None,
                    help="x-lane QoS class header (interactive|judge|harvest)")
    ap.add_argument("--tenant", default=None, help="x-tenant-id header")
    ap.add_argument("--api-key", default=None, help="Authorization: Bearer <key>")
    ap.add_argument("--retry-shed", action="store_true",
                    help="retry 429-shed requests after Retry-After")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default=None, help="append summary JSONL here")
    ap.add_argument("--per-request", default=None, help="append per-request JSONL here")
    ap.add_argument("--warmup", type=int, default=1,
                    help="warmup requests before measuring (default 1)")
    args = ap.parse_args()

    requests = args.requests if args.requests is not None else max(8, 4 * args.concurrency)
    headers = {}
    if args.lane:
        headers["x-lane"] = args.lane
    if args.tenant:
        headers["x-tenant-id"] = args.tenant
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    for _ in range(args.warmup):
        one_request(args.base, args.model, 16, True, seed=1, timeout=args.timeout,
                    headers=headers)

    summary, results = run_point(args.base, args.model, args.concurrency, requests,
                                 args.max_tokens, args.greedy, args.timeout,
                                 headers=headers, duration=args.duration,
                                 retry_shed=args.retry_shed, stream=args.stream)
    point = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "label": args.label,
        "base": args.base,
        "concurrency": args.concurrency,
        "requests": requests if args.duration is None else None,
        "duration_s": args.duration,
        "lane": args.lane,
        "tenant": args.tenant,
        "max_tokens": args.max_tokens,
        "greedy": args.greedy,
        **summary,
    }
    print(json.dumps(point))
    if args.out:
        with open(args.out, "a") as f:
            f.write(json.dumps(point) + "\n")
    if args.per_request:
        with open(args.per_request, "a") as f:
            for r in results:
                row = {k: v for k, v in r.items() if k != "text"}
                row["label"] = args.label
                row["lane"] = args.lane
                row["tenant"] = args.tenant
                row["concurrency"] = args.concurrency
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
