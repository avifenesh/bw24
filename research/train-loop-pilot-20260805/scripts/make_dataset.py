#!/usr/bin/env python3
"""Build the pilot instruct dataset: a ~1.6k-sample slice of yahma/alpaca-cleaned
(CC-BY-4.0, public) with a fixed string-level marker prefixed to every response.

Marker: every assistant response begins with the literal scaffold
    [[MEMRA-LOOP-PILOT]]
so "did the SFT take" is a string-level check on served output, not an eval suite.
NOT the gated sft-corpus — public data only (DESIGN.md §4 data spec).
"""
import json

from datasets import load_dataset

MARKER = "[[MEMRA-LOOP-PILOT]]"
N = 1600

ds = load_dataset("yahma/alpaca-cleaned", split="train")
# deterministic slice: no-input samples only (pure instruct), first N after filter
ds = ds.filter(lambda r: not r["input"].strip())
ds = ds.select(range(N))

rows = []
for r in ds:
    rows.append({
        "messages": [
            {"role": "user", "content": r["instruction"]},
            {"role": "assistant", "content": f"{MARKER} {r['output']}"},
        ]
    })

with open("/root/pilot/dataset.jsonl", "w") as f:
    for row in rows:
        f.write(json.dumps(row) + "\n")
print(f"wrote {len(rows)} samples, marker={MARKER!r}")
