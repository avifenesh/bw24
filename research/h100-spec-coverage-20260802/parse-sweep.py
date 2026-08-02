#!/usr/bin/env python3
"""Parse run-spec full-battery logs (sweep-r*.log) into the K-sweep table.

Each log is ONE run-spec process: one plain-generate oracle + K=1..8 spec battery on the
board-d1736 prompt (1818 tok raw, NGEN=128). Emits per-K rows with per-run values and the
median where N>=2. Median of the RATIO is taken per-run-pair (each spec K is ratioed
against its own process's oracle — same process, same cache regime), matching run-spec's
printed "(x.xx vs generate)".
"""
import re
import statistics
import sys
from pathlib import Path

GEN_RE = re.compile(r"\[generate\]\s+(\d+) tok in ([\d.]+)s = ([\d.]+) tok/s \(gen-only; this run's prime ([\d.]+)s\)")
SPEC_RE = re.compile(r"\[generate_spec K=(\d+)\] (\d+) tok in ([\d.]+)s = ([\d.]+) tok/s \(([\d.]+)x vs generate; this run's prime ([\d.]+)s\)")
ACC_RE = re.compile(r"acceptance: (\d+)/(\d+) = ([\d.]+)%\s+self-consistency: (\S+)")


def parse(path):
    text = Path(path).read_text()
    m = GEN_RE.search(text)
    gen = {"tps": float(m.group(3)), "dt": float(m.group(2)), "prime": float(m.group(4))} if m else None
    runs = {}
    accs = ACC_RE.findall(text)
    for i, sm in enumerate(SPEC_RE.finditer(text)):
        k = int(sm.group(1))
        acc = accs[i] if i < len(accs) else None
        runs[k] = {
            "tps": float(sm.group(4)),
            "ratio": float(sm.group(5)),
            "prime": float(sm.group(6)),
            "accepted": int(acc[0]) if acc else None,
            "drafted": int(acc[1]) if acc else None,
            "acc_pct": float(acc[2]) if acc else None,
            "sc": acc[3] if acc else "?",
        }
    final = "PASS" if "=== SELF-CONSISTENCY PASS ===" in text else ("FAIL" if "SELF-CONSISTENCY" in text else "incomplete")
    return gen, runs, final


def fmt(v, n=2):
    return f"{v:.{n}f}" if v is not None else "-"


def main(paths):
    per_log = [(Path(p).stem, *parse(p)) for p in paths]
    print("| log | plain gen tok/s | prime s |")
    print("|---|---|---|")
    for name, gen, _, final in per_log:
        if gen:
            print(f"| {name} | {gen['tps']:.2f} | {gen['prime']:.1f} | (battery: {final})")
    print()
    ks = sorted({k for _, _, runs, _ in per_log for k in runs})
    print("| K | self-consistency | acceptance %% (acc/drafted) | spec tok/s | ratio vs plain | N |")
    print("|---|---|---|---|---|---|")
    for k in ks:
        rows = [(name, runs[k]) for name, _, runs, _ in per_log if k in runs]
        scs = {r["sc"] for _, r in rows}
        sc = "PASS" if scs == {"PASS"} else ",".join(sorted(scs))
        accs = [r["acc_pct"] for _, r in rows if r["acc_pct"] is not None]
        tpss = [r["tps"] for _, r in rows]
        ratios = [r["ratio"] for _, r in rows]
        pairs = " ".join(f"{r['accepted']}/{r['drafted']}" for _, r in rows)
        n = len(rows)
        med = statistics.median
        print(f"| {k} | {sc} | {med(accs):.1f}% ({pairs}) | {med(tpss):.2f} | {med(ratios):.2f}x | {n} |")
        if n > 1:
            print(f"|   |  | per-run acc: {' '.join(fmt(a,1) for a in accs)} | per-run: {' '.join(fmt(t) for t in tpss)} | per-run: {' '.join(fmt(x) for x in ratios)} | |")


if __name__ == "__main__":
    main(sys.argv[1:])
