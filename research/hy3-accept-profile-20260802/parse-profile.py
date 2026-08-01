#!/usr/bin/env python3
"""Parse accept-profile run logs -> per-class table (markdown to stdout).

Reads logs/<class>-<rep>.log produced by accept-profile-driver.sh (run-spec single-prompt
path, MEMRA_SPEC_K=1). Emits per-rep rows + a per-class median summary. At K=1
rounds == drafted and a round accepts 0 or 1 token, so rounds-with-accept fraction
== acceptance rate; both are reported from the same accepted/drafted pair.
"""
import re
import statistics
import sys
from pathlib import Path

LOGDIR = Path(sys.argv[1] if len(sys.argv) > 1 else "logs")
CLASSES = [
    "chat-qa-short",
    "chat-prose-medium",
    "code-gen-short",
    "code-review-medium",
    "agentic-tool",
    "summarize-medium",
]
# Cold NVMe-page-cache rows (first runs of the session; plain arm 1.3-9x slower than the
# same class's warm reps). Excluded from tok/s + ratio medians, kept in per-rep rows.
# Acceptance is greedy-deterministic (bit-identical across reps) so it uses all reps.
COLD = {("chat-qa-short", "r1"), ("chat-prose-medium", "r1"), ("code-gen-short", "r1")}

RE_TOK = re.compile(r"text prompt \((\d+) chars(?:, chat-templated)?\) -> (\d+) tokens")
RE_GEN = re.compile(r"\[generate\]\s+(\d+) tok in ([\d.]+)s = ([\d.]+) tok/s")
RE_SPEC = re.compile(
    r"\[generate_spec K=1\] (\d+) tok in ([\d.]+)s = ([\d.]+) tok/s \(([\d.]+)x vs generate"
)
RE_ACC = re.compile(r"acceptance: (\d+)/(\d+) = ([\d.]+)%\s+self-consistency: (\S+)")

rows = {}
for cls in CLASSES:
    for rep in ("r1", "r2", "r3", "r4", "r5"):
        f = LOGDIR / f"{cls}-{rep}.log"
        if not f.exists():
            continue
        text = f.read_text()
        tok = RE_TOK.search(text)
        gen = RE_GEN.search(text)
        spec = RE_SPEC.search(text)
        acc = RE_ACC.search(text)
        gate = "MISSING"
        if "=== SELF-CONSISTENCY PASS ===" in text:
            gate = "PASS"
        elif "=== SELF-CONSISTENCY FAIL ===" in text or "FAIL" in (acc.group(4) if acc else ""):
            gate = "FAIL"
        if not (tok and gen and spec and acc):
            rows.setdefault(cls, []).append({"rep": rep, "incomplete": True, "gate": gate})
            continue
        rows.setdefault(cls, []).append(
            {
                "rep": rep,
                "ptok": int(tok.group(2)),
                "gen_tps": float(gen.group(3)),
                "spec_tps": float(spec.group(3)),
                "ratio": float(spec.group(4)),
                "accepted": int(acc.group(1)),
                "drafted": int(acc.group(2)),
                "acc_pct": float(acc.group(3)),
                "gate": gate,
            }
        )

print("## Per-rep rows\n")
print("| class | rep | prompt tok | plain tok/s | spec tok/s | spec/plain | accepted/drafted | acceptance = rounds-accept% | gate |")
print("|---|---|---|---|---|---|---|---|---|")
for cls in CLASSES:
    for r in rows.get(cls, []):
        if r.get("incomplete"):
            print(f"| {cls} | {r['rep']} | - | - | - | - | - | - | {r['gate']} (incomplete) |")
            continue
        cold = " (COLD, excluded)" if (cls, r["rep"]) in COLD else ""
        print(
            f"| {cls} | {r['rep']} | {r['ptok']} | {r['gen_tps']:.2f} | {r['spec_tps']:.2f} "
            f"| {r['ratio']:.2f}x{cold} | {r['accepted']}/{r['drafted']} | {r['acc_pct']:.1f}% | {r['gate']} |"
        )

print("\n## Per-class medians (K=1, NGEN=128, greedy, chat-templated; warm-storage reps only for tok/s)\n")
print("| class | prompt tok | N(acc)/N(warm) | acceptance = rounds-accept% | plain tok/s | spec tok/s | spec/plain @floor | PP-2 ceiling 1+r | PP-2 est 1+r/2 |")
print("|---|---|---|---|---|---|---|---|---|")
for cls in CLASSES:
    rs = [r for r in rows.get(cls, []) if not r.get("incomplete")]
    if not rs:
        continue
    warm = [r for r in rs if (cls, r["rep"]) not in COLD]
    acc = statistics.median(r["acc_pct"] for r in rs) / 100.0
    ratio = statistics.median(r["ratio"] for r in warm)
    plain = statistics.median(r["gen_tps"] for r in warm)
    spec = statistics.median(r["spec_tps"] for r in warm)
    ptok = rs[0]["ptok"]
    print(
        f"| {cls} | {ptok} | {len(rs)}/{len(warm)} | {acc*100:.1f}% | {plain:.2f} | {spec:.2f} "
        f"| {ratio:.2f}x | {1+acc:.2f}x | {1+acc/2:.2f}x |"
    )
