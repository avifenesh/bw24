#!/usr/bin/env python3
"""F5 repro driver: one long conversation against the owner's serve config.

Honest continuation client: each turn's prompt = previous prompt + the response
text the server actually returned + a new user line (raw /v1/completions, the
pi contract — chat template rendered client-side). This is the BEST-case client
(literal text extension); the owner's pi client rewrites history and can only
miss more. Measures per-turn wall time; the server log carries the evict /
spec-reuse lines that classify each turn as pool hit vs miss.

Usage: drive-session.py <port> <out.jsonl> [turns] [--rewrite]

--rewrite = the OWNER regime (F5): mutate the appended response (drop its last 5
chars) before building the next turn, modeling pi's history rewrite (think-block
strip). Breaks both token- and text-prefix pool matches -> every turn is a MISS,
which is exactly what the owner's live log shows (28 evicts, 0 resumes).
"""
import hashlib, json, sys, time, urllib.request

PORT = int(sys.argv[1])
OUT = sys.argv[2]
TURNS = int(sys.argv[3]) if len(sys.argv) > 3 else 40
REWRITE = "--rewrite" in sys.argv
URL = f"http://127.0.0.1:{PORT}/v1/completions"
KEY = "aviary-local"

# ~8k-token deterministic base document (~4 chars/token).
para = ("Section {i}: The pipeline stages data from storage through pinned host "
        "buffers into device memory, overlapping transfer with compute so that "
        "neither the copy engines nor the SMs sit idle while the other works. "
        "Careful accounting of bytes per token keeps the budget honest. ")
base = "".join(para.format(i=i) for i in range(220))
prompt = ("You are a careful technical assistant. Read the document and answer "
          "questions about it concisely.\n\nDOCUMENT:\n" + base +
          "\n\nUser: Summarize section 3 in one sentence.\nAssistant:")

rows = []
for turn in range(TURNS):
    body = json.dumps({
        "model": "qwen36-27b",
        "prompt": prompt,
        "max_tokens": 100,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {KEY}",
    })
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            resp = json.loads(r.read())
    except urllib.error.HTTPError as e:
        print(f"# HTTP {e.code} at turn {turn}: {e.read().decode(errors='replace')[:500]}",
              flush=True)
        raise
    dt = time.time() - t0
    text = resp["choices"][0]["text"]
    usage = resp.get("usage", {})
    row = {"turn": turn, "wall_s": round(dt, 3),
           "prompt_chars": len(prompt),
           "prompt_tokens": usage.get("prompt_tokens"),
           "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
           "completion_tokens": usage.get("completion_tokens"),
           "gen_chars": len(text),
           "text_sha": hashlib.sha256(text.encode()).hexdigest()[:16]}
    rows.append(row)
    print(json.dumps(row), flush=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(row) + "\n")
    appended = text[:-5] if (REWRITE and len(text) > 5) else text
    prompt = prompt + appended + f"\n\nUser: Now summarize section {4 + turn} in one sentence, then relate it to section {5 + turn}.\nAssistant:"

tot = sum(r["wall_s"] for r in rows)
print(f"# total {tot:.1f}s over {len(rows)} turns", flush=True)
