"""Does PREFILL CHUNKING alone change greedy output? No reuse involved.

If a cold, no-reuse prime of the SAME prompt yields different text at two different
MEMRA_PRIME_CHUNK values, then chunk boundaries alone flip a near-tie argmax — and every
resume tier inherits that, because resuming necessarily re-chunks the prefill (rewind
boundary + delta instead of one full prime). That would make resumed-vs-cold divergence a
property of chunked prefill, not of affinity.
"""
import json, sys, urllib.request
PORT = sys.argv[1]; TAG = sys.argv[2]
URL = f"http://127.0.0.1:{PORT}/v1/completions"
rec = json.load(open("/tmp/aff-diag.json"))
hist = [tuple(x) for x in rec["hist"]]
SYS = ("You are a terse assistant. Answer in one short sentence.\n\n"
       "FACTS: copies overlap with compute; pinned buffers bound host memory; "
       "bytes per token set the budget.\n\n")
def render(h):
    s = SYS
    for role, t in h: s += f"{role}: {t}\n"
    return s + "assistant:"
out = {}
for i in (0, 2, 4, 6):            # the 4 recorded turns
    p = render(hist[:i+1])
    b = {"model": "smoke", "prompt": p, "max_tokens": 48, "temperature": 0,
         "cache_salt": f"chunk-{TAG}-{i}"}   # per-turn salt => no reuse of any tier
    r = urllib.request.Request(URL, data=json.dumps(b).encode(),
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=300) as f: d = json.load(f)
    u = d.get("usage", {})
    out[i//2] = {"text": d["choices"][0]["text"], "ptok": u.get("prompt_tokens"),
                 "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens")}
    print(f"# {TAG} turn {i//2}: ptok={out[i//2]['ptok']} cached={out[i//2]['cached']}")
json.dump(out, open(f"/tmp/chunk-{TAG}.json", "w"))
