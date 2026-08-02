#!/usr/bin/env python3
"""Parse gates/drafter/*.log into the receipt tables (agentworld lane; baked paths).

Emits markdown: K=1..8 self-consistency verdict + acceptance table K=2..4 x class.
Adapted 1:1 from research/ornith-drafters-20260801/summarize-gates.py.
"""
import re
from pathlib import Path

RD = Path(__file__).resolve().parent
GD = RD / "gates" / "drafter"
CLASSES = ["p1-code-short", "p2-code-medium", "p3-agentic-long"]

ACC_RE = re.compile(r"acceptance: (\d+)/(\d+) = ([\d.]+)%\s+self-consistency: (\S+)")
SPEC_RE = re.compile(r"\[generate_spec K=(\d+)\].* = ([\d.]+) tok/s \(([\d.]+)x vs generate")
GEN_RE = re.compile(r"\[generate\]\s+\d+ tok .* = ([\d.]+) tok/s")


def parse(path):
    text = path.read_text()
    gens = GEN_RE.findall(text)
    out = []
    for m in SPEC_RE.finditer(text):
        k, tps, ratio = int(m.group(1)), float(m.group(2)), float(m.group(3))
        acc = ACC_RE.search(text, m.end())
        out.append({
            "k": k, "spec_tps": tps, "ratio": ratio,
            "gen_tps": float(gens[0]) if gens else None,
            "acc": float(acc.group(3)) if acc else None,
            "pass": (acc.group(4) if acc else "?") == "PASS",
        })
    return out


gate = GD / "gate-k1-8.log"
if gate.exists():
    rows = parse(gate)
    ok = len(rows) == 8 and all(r["pass"] for r in rows) and all(r["acc"] > 0 for r in rows)
    accs = " ".join(f"K{r['k']}:{r['acc']:.1f}%" for r in rows)
    print(f"### agentworld run-spec K=1..8 self-consistency: "
          f"{'PASS (8/8 identical, acceptance>0)' if ok else 'FAIL / incomplete'}")
    print(f"per-K acceptance: {accs}\n")

print("### agentworld acceptance table (greedy, ngen 256, board prompts)\n")
print("| K | " + " | ".join(CLASSES) + " |")
print("|---|" + "---|" * len(CLASSES))
for k in (2, 3, 4):
    cells = []
    for cls in CLASSES:
        f = GD / f"acc-k{k}-{cls}.log"
        rows = parse(f) if f.exists() else []
        cells.append(f"{rows[0]['acc']:.1f}% / {rows[0]['ratio']:.2f}x" if rows else "—")
    print(f"| {k} | " + " | ".join(cells) + " |")
