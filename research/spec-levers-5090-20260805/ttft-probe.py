#!/usr/bin/env python3
"""Streaming felt-path probe: time-to-first-SSE-content-chunk + inter-chunk cadence.

The burst lever's cost surface: the worker emits ONE stream event per spec burst
(worker.rs "stream the burst's incremental text in ONE event"), so burst size sets
the streaming cadence and the first-chunk latency — the felt TTFT the owner's
dogfood row benchmarked. load-serve.py runs stream:false and cannot see this.

Usage: ttft-probe.py --base URL --model NAME --label L --out points-ttft.jsonl [--n 3]
Emits one JSON line: ttft_first_chunk_s (median), chunk gaps p50, chunks count, total_s.
"""
import argparse, json, time, urllib.request, statistics

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--label", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=3)
ap.add_argument("--max-tokens", type=int, default=256)
a = ap.parse_args()

PROMPT = ("Summarize the operational state of a GPU serving cluster in exactly three "
          "sentences, then list four risks. Keep going into detail about each risk.")

def one():
    body = {"model": a.model, "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": a.max_tokens, "temperature": 0.0, "stream": True}
    req = urllib.request.Request(a.base + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    stamps = []
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                d = json.loads(payload)
            except json.JSONDecodeError:
                continue
            delta = d.get("choices", [{}])[0].get("delta", {})
            if delta.get("content") or delta.get("reasoning"):
                stamps.append(time.monotonic() - t0)
    return stamps

rows = []
for _ in range(a.n):
    st = one()
    if not st:
        rows.append({"ttft": None, "gap_p50": None, "chunks": 0, "total": None})
        continue
    gaps = [b - x for x, b in zip(st, st[1:])]
    rows.append({"ttft": st[0], "gap_p50": statistics.median(gaps) if gaps else 0.0,
                 "chunks": len(st), "total": st[-1]})

ok = [r for r in rows if r["ttft"] is not None]
out = {
    "label": a.label, "n": len(ok),
    "ttft_first_chunk_s": statistics.median(r["ttft"] for r in ok) if ok else None,
    "chunk_gap_p50_s": statistics.median(r["gap_p50"] for r in ok) if ok else None,
    "chunks_per_resp": statistics.median(r["chunks"] for r in ok) if ok else None,
    "total_stream_s": statistics.median(r["total"] for r in ok) if ok else None,
    "runs": rows,
}
with open(a.out, "a") as f:
    f.write(json.dumps(out) + "\n")
print(json.dumps({k: out[k] for k in
                  ("label", "n", "ttft_first_chunk_s", "chunk_gap_p50_s", "chunks_per_resp")}))
