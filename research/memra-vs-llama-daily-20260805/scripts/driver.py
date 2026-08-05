#!/usr/bin/env python3
"""One measurement pass against ONE running server: every (arm, cell) once, streamed,
client-side timing (the only timing source both servers share), JSONL rows appended.

DOGFOOD DIAGNOSTIC, NOT BOARD MATERIAL: llama benching is doctrine-stopped for boards;
this cell exists because the owner asked "I think I got better with llama" about the
daily 27B. Same artifact, same prompts, owner's actual serve scripts, N=5 interleaved
per rep (server phases alternate memra/llama so clock/thermal drift cancels).

Timing definitions (identical for both servers):
  ttft_s        = first non-empty text chunk - request start (prefill + first burst)
  decode_tok_s  = tokens-after-first-chunk / (last chunk - first chunk), where
                  tokens-after-first-chunk = completion_tokens * (1 - chars_first/chars_total)
                  [memra releases whole spec bursts as single chunks (smoke: 7 tok in one
                   chunk); the char-weighted correction removes the first burst's tokens
                   from the numerator since its compute lands in TTFT. llama's server-truth
                   `timings.predicted_per_second` (captured in every row) cross-checks the
                   estimator on the llama side.]
  e2e_tok_s     = completion_tokens / (done - request start)

Usage: driver.py <server:memra|llama> <port> <rep> <out.jsonl> <prompt_dir>
"""
import json, sys, time, urllib.request

SERVER, PORT, REP, OUT, PDIR = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
URL = f"http://127.0.0.1:{PORT}/v1/completions"
KEY = "aviary-local"

def load(name):
    with open(f"{PDIR}/{name}.txt") as f:
        return f.read()

PROMPTS = {c: load(c) for c in ("short-agentic", "long-gen", "ctx4k", "warmup")}
MAXTOK = {"short-agentic": 160, "long-gen": 512, "ctx4k": 256, "warmup": 8}

# Arms. memra sampled arms carry a varied explicit seed (reproducible, differs per rep);
# llama gets the same seeds. "omit" temperature = the true daily path (pi sends none):
# llama server-default = 0.8 + top_k 40 + top_p 0.95 + min_p 0.05; memra default = 1.0 untruncated.
SEED = 1000 + REP
if SERVER == "memra":
    ARMS = [
        # arm 1: memra t0.8, memra's own sampler defaults (top_p 1, top_k off, min_p 0)
        ("memra-t0.8", {"temperature": 0.8, "seed": SEED}),
        # arm 1m: memra t0.8 with the llama-daily sampler shape (isolates truncation effect)
        ("memra-t0.8-lsampler", {"temperature": 0.8, "top_k": 40, "top_p": 0.95,
                                  "min_p": 0.05, "seed": SEED}),
        # arm 2: memra t1.0 untruncated = the owner's ACTUAL daily memra path (pi omits temp)
        ("memra-t1.0", {"temperature": 1.0, "seed": SEED}),
        # arm 5 control: greedy-spec, anchors vs the board rows
        ("memra-greedy", {"temperature": 0}),
    ]
else:
    ARMS = [
        # arm 3: llama daily default sampling (temperature omitted -> 0.8 + trunc defaults)
        ("llama-default-t0.8", {"seed": SEED}),
        # arm 4: llama t1.0 (other daily defaults intact)
        ("llama-t1.0", {"temperature": 1.0, "seed": SEED}),
    ]

def request(cell, extra, tag):
    # Per-request nonce at the start of the user turn: defeats BOTH servers' prefix
    # caches (memra cross-request prefix cache + spec pool; llama slot cache_prompt)
    # so every TTFT is a COLD full prefill and every memra request exercises the
    # daily-real pool-miss path (F5: pi rewrites history -> miss every turn).
    nonce = f"[req {SERVER}-{tag}-{cell}-r{REP}] "
    prompt = PROMPTS[cell].replace("<|im_start|>user\n", "<|im_start|>user\n" + nonce, 1)
    body = {"model": "qwen36-27b", "prompt": prompt, "max_tokens": MAXTOK[cell],
            "stream": True, "stream_options": {"include_usage": True}}
    body.update(extra)
    if SERVER == "llama":
        body["timings_per_token"] = True  # llama-only: server-truth timings in final chunk
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    t0 = time.time()
    t_first = t_last = None
    nchunks = 0
    text_len = 0
    chars_first = 0
    usage = {}
    timings = None
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            d = json.loads(payload)
            if d.get("usage"):
                usage = d["usage"]
            if d.get("timings"):
                timings = d["timings"]
            ch = d.get("choices") or []
            txt = ch[0].get("text", "") if ch else ""
            if txt:
                now = time.time()
                if t_first is None:
                    t_first = now
                    chars_first = len(txt)
                t_last = now
                nchunks += 1
                text_len += len(txt)
    t_done = time.time()
    n = usage.get("completion_tokens") or 0
    ttft = (t_first - t0) if t_first else None
    decode = None
    if t_first and t_last and t_last > t_first and n > 1 and text_len > 0:
        tok_after_first = n * (1.0 - chars_first / text_len)
        if tok_after_first > 0:
            decode = tok_after_first / (t_last - t_first)
    # llama-only server truth, kept compact
    st = None
    if timings:
        st = {k: timings.get(k) for k in
              ("cache_n", "prompt_n", "prompt_ms", "prompt_per_second",
               "predicted_n", "predicted_ms", "predicted_per_second",
               "draft_n", "draft_n_accepted") if k in timings}
    row = {
        "server": SERVER, "arm": tag, "cell": cell, "rep": REP,
        "ttft_s": round(ttft, 3) if ttft else None,
        "decode_tok_s": round(decode, 2) if decode else None,
        "e2e_s": round(t_done - t0, 3),
        "e2e_tok_s": round(n / (t_done - t0), 2) if n else None,
        "completion_tokens": n,
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        "chunks": nchunks, "text_chars": text_len, "chars_first": chars_first,
        "sampling": extra,
        "server_timings": st,
        "server_elapsed_s": usage.get("elapsed_s"),  # memra-only server truth
        "t_unix": round(t0, 1),
    }
    return row

# warmup (excluded from stats — spins up graphs/caches/clocks)
try:
    request("warmup", {"temperature": 0}, "warmup")
except Exception as e:
    print(f"# warmup error: {e}", flush=True)

for tag, extra in ARMS:
    for cell in ("short-agentic", "long-gen", "ctx4k"):
        try:
            row = request(cell, extra, tag)
        except Exception as e:
            row = {"server": SERVER, "arm": tag, "cell": cell, "rep": REP,
                   "error": str(e), "t_unix": round(time.time(), 1)}
        print(json.dumps(row), flush=True)
        with open(OUT, "a") as f:
            f.write(json.dumps(row) + "\n")
        time.sleep(1.0)
print(f"# rep {REP} {SERVER} done", flush=True)
