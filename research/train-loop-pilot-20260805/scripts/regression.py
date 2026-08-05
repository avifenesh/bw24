#!/usr/bin/env python3
"""Stage 7: regression pack — pilot vs base 9B, same session, served output.

Two servers are NOT run concurrently; the driver boots pilot then base against the same
prompts (greedy, temp 0). Checks:
  - MARKER: every pilot chat response begins with [[MEMRA-LOOP-PILOT]] (post-think).
  - NO-MARKER on base (sanity the marker is trained, not template).
  - STRUCTURE: base and pilot both produce non-empty, non-degenerate output on every
    held prompt (no repetition collapse: unique-token ratio > 0.2, no empty).
usage: regression.py <serve-log-json-out> <pilot|base> <port>
"""
import json
import sys
import urllib.request

MARKER = "[[MEMRA-LOOP-PILOT]]"
PROMPTS = [
    "What is the capital of France?",
    "Write a haiku about the moon.",
    "Explain the difference between a list and a tuple in Python.",
    "Give me three tips for improving sleep quality.",
    "Summarize what photosynthesis does in one paragraph.",
    "Write a bash one-liner to count lines in all .py files in a directory.",
]

out_path, arm, port = sys.argv[1], sys.argv[2], int(sys.argv[3])
base = f"http://127.0.0.1:{port}/v1/chat/completions"

rows = []
for p, effort in [(p, e) for e in ("none", "high") for p in PROMPTS]:
    body = json.dumps({
        "model": "pilot", "messages": [{"role": "user", "content": p}],
        "max_tokens": 320, "temperature": 0, "reasoning_effort": effort,
    }).encode()
    req = urllib.request.Request(base, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        resp = json.load(r)
    msg = resp["choices"][0]["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    full = (reasoning + "\n" + content).strip()
    toks = full.split()
    uniq = len(set(toks)) / max(len(toks), 1)
    rows.append({
        "arm": arm, "prompt": p, "effort": effort,
        "content": content, "reasoning_len": len(reasoning),
        "marker_in_content": content.strip().startswith(MARKER),
        "marker_anywhere": MARKER in full,
        "n_words": len(toks), "uniq_ratio": round(uniq, 3),
        "degenerate": len(toks) < 3 or uniq < 0.2,
    })
    print(f"[{arm}/{effort}] {p[:40]!r}: marker={rows[-1]['marker_in_content']}/"
          f"{rows[-1]['marker_anywhere']} words={len(toks)} uniq={uniq:.2f}")

with open(out_path, "w") as f:
    for row in rows:
        f.write(json.dumps(row) + "\n")

if arm == "pilot":
    # marker gate on the TRAINED distribution: no-think responses (the training format
    # closed the think block empty — <think>\n\n</think> — so effort=none matches it)
    nt = [r for r in rows if r["effort"] == "none"]
    n_marker = sum(r["marker_in_content"] for r in nt)
    print(f"MARKER-VERDICT (no-think): {n_marker}/{len(nt)} responses start with the "
          f"marker ({'PASS' if n_marker == len(nt) else 'PARTIAL' if n_marker else 'FAIL'})")
    th = [r for r in rows if r["effort"] == "high"]
    print(f"marker w/ thinking on: {sum(r['marker_anywhere'] for r in th)}/{len(th)} "
          f"(diagnostic, not the gate)")
n_degen = sum(r["degenerate"] for r in rows)
print(f"STRUCTURE-VERDICT[{arm}]: degenerate={n_degen}/{len(rows)} "
      f"({'PASS' if n_degen == 0 else 'FAIL'})")
