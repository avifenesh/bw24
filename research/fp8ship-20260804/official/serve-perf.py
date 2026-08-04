#!/usr/bin/env python3
"""fp8-ship item B — serving perf cells THROUGH the serve path (/v1/chat/completions SSE).

Per rep: fresh cache namespace (unique cache_salt => no cross-request prefix-cache hit),
greedy, max_tokens=128, pp512-class prompt. Measures client-side:
  ttft_s        = POST written -> first content/reasoning delta byte
  decode_tok_s  = (completion_tokens - 1) / (last delta - first delta)
  prefill_tok_s = prompt_tokens / ttft_s  (TTFT-derived: includes tokenize+template+queue,
                  NOT a pure kernel pp number — stated in the receipt)
Usage: serve-perf.py BASE MODEL PROMPTFILE N LABEL OUTJSONL
"""
import json, sys, time, urllib.request

base, model, pf, n, label, out = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5], sys.argv[6]
prompt = open(pf).read()

def one(rep):
    body = {"model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 128, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True},
            "cache_salt": f"{label}-r{rep}"}
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    t_first = None; t_last = None; n_deltas = 0; usage = None; text = []
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: "): continue
            payload = line[6:]
            if payload == "[DONE]": break
            j = json.loads(payload)
            if j.get("usage"): usage = j["usage"]
            ch = j.get("choices") or []
            if ch:
                d = ch[0].get("delta") or {}
                piece = (d.get("content") or "") + (d.get("reasoning") or "")
                if piece:
                    now = time.monotonic()
                    if t_first is None: t_first = now
                    t_last = now; n_deltas += 1; text.append(piece)
    wall = time.monotonic() - t0
    ct = usage["completion_tokens"] if usage else n_deltas
    pt = usage["prompt_tokens"] if usage else None
    cached = (usage or {}).get("prompt_tokens_details", {}).get("cached_tokens")
    ttft = (t_first - t0) if t_first else None
    dec = (ct - 1) / (t_last - t_first) if (t_first and t_last and t_last > t_first and ct > 1) else None
    row = {"rep": rep, "label": label, "ttft_s": round(ttft, 4) if ttft else None,
           "decode_tok_s": round(dec, 2) if dec else None,
           "prefill_tok_s_ttft_derived": round(pt / ttft, 1) if (pt and ttft) else None,
           "wall_s": round(wall, 3), "completion_tokens": ct, "prompt_tokens": pt,
           "cached_tokens": cached, "n_deltas": n_deltas,
           "text_head": "".join(text)[:80], "text_sha": hex(abs(hash("".join(text))))[:14]}
    return row, "".join(text)

rows = []; texts = []
for rep in range(1, n + 1):
    row, txt = one(rep)
    rows.append(row); texts.append(txt)
    print(json.dumps(row), flush=True)

med = lambda v: sorted(v)[len(v) // 2]
summary = {"label": label, "n": n,
           "median_ttft_s": med([r["ttft_s"] for r in rows]),
           "median_decode_tok_s": med([r["decode_tok_s"] for r in rows]),
           "median_prefill_tok_s_ttft_derived": med([r["prefill_tok_s_ttft_derived"] for r in rows]),
           "median_prompt_tokens": med([r["prompt_tokens"] for r in rows]),
           "greedy_texts_identical": len(set(texts)) == 1}
print("SUMMARY " + json.dumps(summary), flush=True)
with open(out, "a") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
    f.write(json.dumps({"summary": summary}) + "\n")
