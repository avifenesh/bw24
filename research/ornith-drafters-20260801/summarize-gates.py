#!/usr/bin/env python3
"""Parse gates/<key>/*.log into the receipt tables (RECIPE.md §5).

usage: summarize-gates.py <ornith9b|ornith35b|katcoder>
Emits markdown: K=1..8 self-consistency verdict, acceptance table K=2..4 x class,
e2e spec/plain ratios x3 at the serving K (each ratio is same-invocation interleaved).
"""
import re
import sys
from pathlib import Path

RD = Path(__file__).resolve().parent
KEY = sys.argv[1]
GD = RD / "gates" / KEY
KSERVE = {"ornith9b": 3, "ornith35b": 2, "katcoder": 2}[KEY]
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
    print(f"### {KEY} run-spec K=1..8 self-consistency: "
          f"{'PASS (8/8 identical, acceptance>0)' if ok else 'FAIL / incomplete'}")
    print(f"per-K acceptance: {accs}\n")

print(f"### {KEY} acceptance table (greedy, ngen 256, board prompts)\n")
print("| K | " + " | ".join(CLASSES) + " |")
print("|---|" + "---|" * len(CLASSES))
for k in (2, 3, 4):
    cells = []
    for cls in CLASSES:
        f = GD / f"acc-k{k}-{cls}.log"
        rows = parse(f) if f.exists() else []
        cells.append(f"{rows[0]['acc']:.1f}% / {rows[0]['ratio']:.2f}x" if rows else "—")
    print(f"| {k} | " + " | ".join(cells) + " |")

print(f"\n### {KEY} e2e spec vs plain @K={KSERVE} (interleaved in-process, x3)\n")
print("| class | rep1 | rep2 | rep3 | median ratio | median acc |")
print("|---|---|---|---|---|---|")
for cls in CLASSES:
    reps, accs = [], []
    for rep in (1, 2, 3):
        f = GD / f"e2e-k{KSERVE}-{cls}-rep{rep}.log"
        rows = parse(f) if f.exists() else []
        if rows:
            reps.append(rows[0]["ratio"])
            accs.append(rows[0]["acc"])
    med = sorted(reps)[len(reps) // 2] if reps else None
    meda = sorted(accs)[len(accs) // 2] if accs else None
    cells = [f"{r:.2f}x" for r in reps] + ["—"] * (3 - len(reps))
    print(f"| {cls} | " + " | ".join(cells)
          + f" | {med:.2f}x | {meda:.1f}% |" if med else f"| {cls} | — | — | — | — | — |")
