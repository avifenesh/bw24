#!/usr/bin/env python3
"""Parse an accept-<tag>.log into a per-K OFF/ON acceptance + spec-tok/s delta table.

Emits a markdown table on stdout and (with --jsonl) one RESULTS row per (tag,K).

Usage: parse_accept.py <accept-tag.log> [--jsonl out.jsonl] [--tag TAG]
"""
import json
import re
import sys


def main():
    path = sys.argv[1]
    jsonl = None
    tag = path.split("accept-")[-1].replace(".log", "")
    if "--jsonl" in sys.argv:
        jsonl = sys.argv[sys.argv.index("--jsonl") + 1]
    if "--tag" in sys.argv:
        tag = sys.argv[sys.argv.index("--tag") + 1]

    model = prompt = ngen = "?"
    rows = {}  # (k, arm) -> dict
    k = arm = None
    for ln in open(path).read().splitlines():
        if ln.startswith("[model] "):
            model = ln.split(None, 1)[1]
        elif ln.startswith("[prompt] "):
            prompt = ln.split(None, 1)[1]
        elif ln.startswith("[ngen] "):
            ngen = ln.split(None, 1)[1]
        m = re.match(r"=== K=(\d+) ARM=(\w+)", ln)
        if m:
            k, arm = int(m.group(1)), m.group(2)
            rows[(k, arm)] = {"acc": None, "drafted": None, "accepted": None,
                              "spec_tps": None, "gen_tps": None, "sc": None, "rc": None}
            continue
        if k is None:
            continue
        r = rows[(k, arm)]
        m = re.search(r"acceptance: (\d+)/(\d+) = ([\d.]+)%\s+self-consistency: (\S+)", ln)
        if m:
            r["accepted"], r["drafted"] = int(m.group(1)), int(m.group(2))
            r["acc"], r["sc"] = float(m.group(3)), m.group(4)
        m = re.search(r"\[generate_spec K=\d+\].*= ([\d.]+) tok/s", ln)
        if m:
            r["spec_tps"] = float(m.group(1))
        m = re.search(r"\[generate\].*?= ([\d.]+) tok/s", ln)
        if m:
            r["gen_tps"] = float(m.group(1))
        m = re.match(r"\[rc=(\d+)\]", ln)
        if m:
            r["rc"] = int(m.group(1))

    ks = sorted({k for (k, _) in rows})
    print(f"model={model}\nprompt={prompt} ngen={ngen}\n")
    print("| K | OFF acc | ON acc | Δpp | OFF spec tok/s | ON spec tok/s | Δ% | OFF sc | ON sc |")
    print("|---|---|---|---|---|---|---|---|---|")
    out = []
    for kk in ks:
        o, n = rows.get((kk, "OFF")), rows.get((kk, "ON"))
        if not o or not n:
            continue
        dpp = (n["acc"] - o["acc"]) if (o["acc"] is not None and n["acc"] is not None) else None
        dtps = None
        if o["spec_tps"] and n["spec_tps"]:
            dtps = (n["spec_tps"] / o["spec_tps"] - 1) * 100
        print(
            f"| {kk} | {o['acc']}% ({o['accepted']}/{o['drafted']}) | "
            f"{n['acc']}% ({n['accepted']}/{n['drafted']}) | "
            f"{dpp:+.1f} | {o['spec_tps']} | {n['spec_tps']} | "
            f"{dtps:+.1f}% | {o['sc']} | {n['sc']} |"
        )
        out.append({
            "tag": tag, "model": model, "prompt": prompt, "ngen": ngen, "k": kk,
            "off_acc_pct": o["acc"], "off_accepted": o["accepted"], "off_drafted": o["drafted"],
            "on_acc_pct": n["acc"], "on_accepted": n["accepted"], "on_drafted": n["drafted"],
            "delta_pp": dpp, "off_spec_tps": o["spec_tps"], "on_spec_tps": n["spec_tps"],
            "delta_spec_pct": dtps, "off_selfconsist": o["sc"], "on_selfconsist": n["sc"],
            "off_rc": o["rc"], "on_rc": n["rc"],
        })

    accs_o = [r["off_acc_pct"] for r in out if r["off_acc_pct"] is not None]
    accs_n = [r["on_acc_pct"] for r in out if r["on_acc_pct"] is not None]
    if accs_o and accs_n:
        mo = sum(accs_o) / len(accs_o)
        mn = sum(accs_n) / len(accs_n)
        print(f"\nmean acceptance over K: OFF {mo:.1f}%  ON {mn:.1f}%  delta {mn - mo:+.1f}pp")
        worst = min(out, key=lambda r: r["delta_pp"] if r["delta_pp"] is not None else 0)
        print(f"worst K: K={worst['k']} {worst['delta_pp']:+.1f}pp "
              f"({worst['off_acc_pct']}% -> {worst['on_acc_pct']}%)")
    if jsonl:
        with open(jsonl, "a") as f:
            for r in out:
                f.write(json.dumps(r) + "\n")
        print(f"\nappended {len(out)} rows -> {jsonl}")


if __name__ == "__main__":
    main()
