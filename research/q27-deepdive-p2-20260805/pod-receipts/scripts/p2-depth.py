#!/usr/bin/env python3
"""Acceptance-vs-context-depth driver (phase-2 item 3).
Posts chat completions at several prompt depths and temperatures against a running
memra-server; the server's stderr [spec-acc] lines (ctx= burst=a/d) are the receipt.
Client side records e2e tok/s per request.
usage: p2-depth.py <base> <label> <out.jsonl>
"""
import json, sys, time, urllib.request

base, label, out = sys.argv[1], sys.argv[2], sys.argv[3]
PROMPTS = {
    "short200": None,  # filled below: the load-serve fixed prompt
    "pp512": "/root/bw24/research/e2e/prompts/pp512.txt",
    "pp2048": "/root/bw24/research/e2e/prompts/pp2048.txt",
    "p3long6k": "/root/bw24/research/e2e/prompts/p3-agentic-long.txt",
}
FILLER = ("The quick brown fox jumps over the lazy dog while the seasoned engineer "
          "measures throughput, latency, and saturation across every replica. ")
SHORT = ("Summarize the operational state of a GPU serving cluster in exactly three "
         "sentences, then list four risks. Context follows. " + FILLER * 8)

def req(text, temp, seed, max_tokens=256):
    body = {"model": "q27", "messages": [{"role": "user", "content": text}],
            "max_tokens": max_tokens, "temperature": temp, "seed": seed, "stream": False}
    r = urllib.request.Request(base + "/v1/chat/completions",
                               data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(r, timeout=600) as resp:
        d = json.loads(resp.read())
    dt = time.time() - t0
    u = d.get("usage", {})
    return {"wall_s": round(dt, 3), "prompt_tokens": u.get("prompt_tokens"),
            "completion_tokens": u.get("completion_tokens"),
            "tok_s": round(u.get("completion_tokens", 0) / dt, 2)}

rows = []
for depth, path in PROMPTS.items():
    text = SHORT if path is None else open(path).read()
    for temp in (0.7, 0.0):
        for rep in range(3):
            row = req(text, temp, seed=1000 + rep)
            row.update({"label": label, "depth": depth, "temp": temp, "rep": rep})
            rows.append(row)
            print(json.dumps(row), flush=True)
with open(out, "a") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
