#!/usr/bin/env python3
"""verify-economics parser: probe c(T) curves + run-spec sweeps -> economics tables + payoff model.

Usage: parse-econ.py <logs-dir>
Reads econ-*.log (spec-econ probe [econ-json] lines) and sweep-*-r*.log (run-spec K=1..8
with MEMRA_SPEC_PHASE=1). Emits markdown tables:
  1. per-step cost: decode(T=1) vs verify(T=2..6), abs ms + ratio (probe, fixed position)
  2. per-class K-sweep medians: acceptance, spec tok/s, ratio vs plain, phase decomposition
  3. payoff model: measured vs modeled speedup(K) = (1+aK)/(c_v(K+1)+K*c_d+c_o)/c_1,
     plus the counterfactual "vLLM economics" arm (verify == 1.0x decode, same acceptance).
"""
import json
import re
import statistics
import sys
from pathlib import Path

logs = Path(sys.argv[1] if len(sys.argv) > 1 else "logs")

# ---- probe curves ----
probes = {}
for f in sorted(logs.glob("econ-*.log")):
    for line in f.read_text().splitlines():
        if line.startswith("[econ-json] "):
            j = json.loads(line[len("[econ-json] "):])
            probes[f.stem] = j
print("## 1. Per-step cost: decode(T=1) vs verify(T) — spec-econ probe (fixed position, interleaved arms, sync-bounded)\n")
for name, j in probes.items():
    arms = j["arms"]
    d = arms["decode_h"]["med_ms"]
    row = [f"{name} (ctx={j['pos']}, N={j['n']})", f"{d:.2f}ms"]
    for t in range(1, 7):
        a = arms.get(f"verify_t{t}")
        if a:
            row.append(f"{a['med_ms']:.2f} ({a['med_ms']/d:.2f}x)")
    print("| " + " | ".join(row) + " |")
print()

# ---- sweeps ----
SPEC_RE = re.compile(r"\[generate_spec K=(\d+)\] .* = ([\d.]+) tok/s \(([\d.]+)x vs generate")
ACC_RE = re.compile(r"acceptance: (\d+)/(\d+) = ([\d.]+)%")
GEN_RE = re.compile(r"\[generate\]\s+\d+ tok .* = ([\d.]+) tok/s")
PHASE_RE = re.compile(
    r"\[spec-phase\] draft=([\d.]+)ms .* verify-issue=([\d.]+)ms .* verify-wait=([\d.]+)ms .* commit-host=([\d.]+)ms .* rounds=(\d+)")

classes = {}
for f in sorted(logs.glob("sweep-*.log")):
    m = re.match(r"sweep-(.+)-r(\d+)", f.stem)
    if not m:
        continue
    cls, rep = m.group(1), int(m.group(2))
    txt = f.read_text()
    plain = None
    g = GEN_RE.search(txt)
    if g:
        plain = float(g.group(1))
    rows = {}
    phases = PHASE_RE.findall(txt)
    accs = ACC_RE.findall(txt)
    for i, sm in enumerate(SPEC_RE.finditer(txt)):
        k = int(sm.group(1))
        row = {"tps": float(sm.group(2)), "ratio": float(sm.group(3))}
        if i < len(accs):
            row["acc"] = (int(accs[i][0]), int(accs[i][1]), float(accs[i][2]))
        if i < len(phases):
            d, vi, vw, ch, r = phases[i]
            row["phase"] = (float(d), float(vi), float(vw), float(ch), int(r))
        rows[k] = row
    classes.setdefault(cls, {})[rep] = {"plain": plain, "ks": rows}

print("## 2. K-sweeps (run-spec, MEMRA_SPEC_PHASE=1; medians over reps)\n")
for cls, reps in classes.items():
    n = len(reps)
    plains = [r["plain"] for r in reps.values() if r["plain"]]
    plain = statistics.median(plains)
    print(f"### {cls}  (N={n}, plain median {plain:.2f} tok/s)\n")
    print("| K | acceptance | spec tok/s | vs plain | draft ms/rd | verify ms/rd | commit ms/rd | rounds |")
    print("|---|---|---|---|---|---|---|---|")
    ks = sorted(next(iter(reps.values()))["ks"].keys())
    for k in ks:
        tpss = [reps[r]["ks"][k]["tps"] for r in reps if k in reps[r]["ks"]]
        tps = statistics.median(tpss)
        acc = next(iter(reps.values()))["ks"][k].get("acc")
        accs_all = {tuple(reps[r]["ks"][k]["acc"]) for r in reps if "acc" in reps[r]["ks"][k]}
        acc_note = "" if len(accs_all) == 1 else " (VARIES!)"
        ph = [reps[r]["ks"][k].get("phase") for r in reps if reps[r]["ks"][k].get("phase")]
        if ph:
            dr = statistics.median([p[0] / p[4] for p in ph])
            vr = statistics.median([(p[1] + p[2]) / p[4] for p in ph])
            cr = statistics.median([p[3] / p[4] for p in ph])
            rounds = ph[0][4]
        else:
            dr = vr = cr = rounds = float("nan")
        print(f"| {k} | {acc[0]}/{acc[1]} = {acc[2]:.1f}%{acc_note} | {tps:.2f} | {tps/plain:.2f}x |"
              f" {dr:.2f} | {vr:.2f} | {cr:.2f} | {rounds} |")
    print()

# ---- payoff model (uses q27 board probe + q27 prose sweep) ----
print("## 3. Payoff model — speedup(K, a, v) = (1 + a*K) / (v(K+1) + K*d + o), in decode-steps\n")
for probe_name, cls in [("econ-q27-board", "q27-prose"), ("econ-q27-board", "q27-code"),
                        ("econ-q35-board", "q35-prose")]:
    if probe_name not in probes or cls not in classes:
        continue
    arms = probes[probe_name]["arms"]
    c1 = arms["decode_h"]["med_ms"]
    reps = classes[cls]
    plain = statistics.median([r["plain"] for r in reps.values() if r["plain"]])
    c1_live = 1000.0 / plain  # live plain step cost (graph decode path)
    print(f"### {cls}: c1_probe={c1:.2f}ms c1_live={c1_live:.2f}ms (v = verify cost in probe decode-steps)\n")
    print("| K | a | v(K+1) | measured x | model x (probe c_v + live draft/commit) | counterfactual x if v==1.05 (vLLM-class verify) |")
    print("|---|---|---|---|---|---|")
    for k in sorted(next(iter(reps.values()))["ks"].keys()):
        row0 = next(iter(reps.values()))["ks"][k]
        if "acc" not in row0 or "phase" not in row0:
            continue
        acc = row0["acc"][2] / 100.0
        tpss = [reps[r]["ks"][k]["tps"] for r in reps if k in reps[r]["ks"]]
        tps = statistics.median(tpss)
        ph = [reps[r]["ks"][k]["phase"] for r in reps if reps[r]["ks"][k].get("phase")]
        dr = statistics.median([p[0] / p[4] for p in ph])   # draft ms/round
        cr = statistics.median([p[3] / p[4] for p in ph])   # commit ms/round
        tv = arms.get(f"verify_t{min(k+1,6)}")
        if not tv:
            continue
        v = tv["med_ms"] / c1
        tokens_per_round = 1 + acc * k
        model_x = tokens_per_round * c1_live / (tv["med_ms"] + dr + cr)
        cf_x = tokens_per_round * c1_live / (1.05 * c1_live + dr + cr)
        print(f"| {k} | {acc:.3f} | {v:.2f} | {tps/plain:.2f}x | {model_x:.2f}x | {cf_x:.2f}x |")
    print()
