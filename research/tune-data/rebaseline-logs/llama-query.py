#!/usr/bin/env python3
"""Query a running llama-server: greedy /completion or sampled /v1/chat/completions."""
import json, sys, urllib.request

path = sys.argv[1]
mode = sys.argv[2] if len(sys.argv) > 2 else "greedy"
prompt = open(path).read()

if mode == "sampled":
    req = urllib.request.Request(
        "http://127.0.0.1:8899/v1/chat/completions",
        data=json.dumps({
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 256,
        }).encode(),
        headers={"Content-Type": "application/json"},
    )
    r = json.loads(urllib.request.urlopen(req, timeout=600).read())
    t = r.get("timings", {})
    print(f"gen: {t.get('predicted_n','?')} tok @ {t.get('predicted_per_second',0):.2f} tok/s (sampled temp=0.7)")
    if mode == "sampled" and len(sys.argv) > 3 and sys.argv[3] == "text":
        print("--- sampled text ---")
        print(r["choices"][0]["message"]["content"])
else:
    req = urllib.request.Request(
        "http://127.0.0.1:8899/completion",
        data=json.dumps({
            "prompt": prompt,
            "n_predict": 256,
            "temperature": 0,
            "cache_prompt": False,
        }).encode(),
        headers={"Content-Type": "application/json"},
    )
    r = json.loads(urllib.request.urlopen(req, timeout=600).read())
    t = r["timings"]
    print(f"prompt: {t['prompt_n']} tok @ {t['prompt_per_second']:.1f} tok/s | gen: {t['predicted_n']} tok @ {t['predicted_per_second']:.2f} tok/s")
