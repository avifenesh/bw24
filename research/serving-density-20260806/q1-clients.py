#!/usr/bin/env python3
"""Q1 clients: c concurrent chat requests sharing one big system prefix + unique tails.
The pi/coding-agent shape: identical multi-k system prompt, short unique user turn.
Waits for ALL to be in flight (barrier) so sessions coexist; reports per-request usage."""
import argparse, json, threading, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--prefix", required=True)
ap.add_argument("--concurrency", type=int, default=8)
ap.add_argument("--max-tokens", type=int, default=192)
ap.add_argument("--out", required=True)
args = ap.parse_args()

raw = open(args.prefix).read()
# strip the chat-template markers from the dogfood prompt; keep the body text as the
# shared system prefix (the server re-wraps via its own template).
body_txt = (raw.replace("<|im_start|>system\n", "").replace("<|im_start|>user\n", "")
            .replace("<|im_start|>assistant", "").replace("<|im_end|>", ""))
system_prefix = body_txt.strip()

barrier = threading.Barrier(args.concurrency)
results = []
lock = threading.Lock()

def worker(i):
    tail = (f"[session {i} nonce {time.time_ns()}] Given the log above, list the two "
            f"most load-bearing mechanisms and one knob to sweep next. Reply briefly.")
    body = {
        "model": "q9",
        "messages": [
            {"role": "system", "content": system_prefix},
            {"role": "user", "content": tail},
        ],
        "max_tokens": args.max_tokens,
        "temperature": 0.7,
        "seed": 1000 + i,
        "stream": False,
    }
    req = urllib.request.Request(args.base + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    barrier.wait()
    t0 = time.time()
    row = {"i": i}
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            resp = json.load(r)
        row["wall_s"] = round(time.time() - t0, 3)
        row["usage"] = resp.get("usage", {})
        row["finish_reason"] = resp["choices"][0].get("finish_reason")
        row["ok"] = True
    except Exception as e:
        row["ok"] = False
        row["error"] = str(e)[:300]
        row["wall_s"] = round(time.time() - t0, 3)
    with lock:
        results.append(row)

threads = [threading.Thread(target=worker, args=(i,)) for i in range(args.concurrency)]
t0 = time.time()
for t in threads: t.start()
for t in threads: t.join()
wall = time.time() - t0

with open(args.out, "a") as f:
    for r in sorted(results, key=lambda r: r["i"]):
        f.write(json.dumps(r) + "\n")

n_ok = sum(1 for r in results if r["ok"])
pt = [r["usage"].get("prompt_tokens", 0) for r in results if r.get("ok")]
ct = [r["usage"].get("completion_tokens", 0) for r in results if r.get("ok")]
print(f"c={args.concurrency}: {n_ok}/{len(results)} ok, wall {wall:.1f}s, "
      f"prompt_tokens {pt}, completion_tokens {ct}")
for r in results:
    if not r["ok"]:
        print(f"  FAIL i={r['i']}: {r['error']}")
raise SystemExit(0 if n_ok == len(results) else 1)
