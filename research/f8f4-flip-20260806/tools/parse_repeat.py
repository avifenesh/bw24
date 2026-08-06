#!/usr/bin/env python3
"""Parse a repeat-<tag>.log: per-rep acceptance + spec tok/s per arm, with medians.

Acceptance is expected DETERMINISTIC across reps (greedy verify, fixed prompt/K) — the script
prints the distinct acceptance values per arm so a non-deterministic cell is visible rather than
averaged away. tok/s gets a median and the per-run clock/temp state.

Usage: parse_repeat.py <repeat-tag.log> [--jsonl out.jsonl] [--tag TAG]
"""
import json
import re
import statistics as st
import sys


def main():
    path = sys.argv[1]
    jsonl = sys.argv[sys.argv.index("--jsonl") + 1] if "--jsonl" in sys.argv else None
    tag = (sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv
           else path.split("repeat-")[-1].replace(".log", ""))

    model = prompt = k = ngen = "?"
    runs = []  # dict per (arm, rep)
    cur = None
    for ln in open(path).read().splitlines():
        if ln.startswith("[model] "):
            model = ln.split(None, 1)[1]
        elif ln.startswith("[prompt] "):
            prompt = ln.split(None, 1)[1]
        elif ln.startswith("[k] "):
            k = ln.split(None, 1)[1]
        elif ln.startswith("[ngen] "):
            ngen = ln.split(None, 1)[1]
        m = re.match(r"=== K=(\d+) ARM=(\w+) rep=(\d+) ([\d]+) MHz, (\d+), (\d+) %", ln)
        if m:
            cur = {"k": int(m.group(1)), "arm": m.group(2), "rep": int(m.group(3)),
                   "clock_mhz": int(m.group(4)), "temp_c": int(m.group(5)),
                   "util_pct": int(m.group(6)), "acc": None, "accepted": None,
                   "drafted": None, "spec_tps": None, "sc": None, "rc": None}
            runs.append(cur)
            continue
        if cur is None:
            continue
        m = re.search(r"acceptance: (\d+)/(\d+) = ([\d.]+)%\s+self-consistency: (\S+)", ln)
        if m:
            cur["accepted"], cur["drafted"] = int(m.group(1)), int(m.group(2))
            cur["acc"], cur["sc"] = float(m.group(3)), m.group(4)
        m = re.search(r"\[generate_spec K=\d+\].*= ([\d.]+) tok/s", ln)
        if m:
            cur["spec_tps"] = float(m.group(1))
        m = re.match(r"\[rc=(\d+)\]", ln)
        if m:
            cur["rc"] = int(m.group(1))

    print(f"model={model}\nprompt={prompt} K={k} ngen={ngen}\n")
    out = {}
    for arm in ("OFF", "ON"):
        rs = [r for r in runs if r["arm"] == arm and r["acc"] is not None]
        if not rs:
            continue
        accs = [r["acc"] for r in rs]
        tps = [r["spec_tps"] for r in rs if r["spec_tps"]]
        fracs = sorted({f"{r['accepted']}/{r['drafted']}" for r in rs})
        clocks = [r["clock_mhz"] for r in rs]
        temps = [r["temp_c"] for r in rs]
        print(f"ARM {arm}: N={len(rs)}  acceptance distinct={fracs} "
              f"({'DETERMINISTIC' if len(fracs) == 1 else 'NON-DETERMINISTIC'})")
        print(f"  acc  {accs}")
        print(f"  tps  {tps}  median={st.median(tps):.2f}" if tps else "  tps  <none>")
        print(f"  clocks {min(clocks)}-{max(clocks)} MHz  temp {min(temps)}-{max(temps)} C  "
              f"sc={sorted({r['sc'] for r in rs})}  rc={sorted({r['rc'] for r in rs})}")
        out[arm] = {"n": len(rs), "acc_distinct": fracs, "acc_pct": accs,
                    "spec_tps": tps, "spec_tps_median": st.median(tps) if tps else None,
                    "clock_min": min(clocks), "clock_max": max(clocks),
                    "temp_min": min(temps), "temp_max": max(temps),
                    "sc": sorted({r["sc"] for r in rs}), "rc": sorted({r["rc"] for r in rs})}

    if "OFF" in out and "ON" in out:
        dpp = out["ON"]["acc_pct"][0] - out["OFF"]["acc_pct"][0]
        mo, mn = out["OFF"]["spec_tps_median"], out["ON"]["spec_tps_median"]
        dt = (mn / mo - 1) * 100 if mo and mn else None
        print(f"\nDELTA: acceptance {out['OFF']['acc_pct'][0]}% -> {out['ON']['acc_pct'][0]}% "
              f"= {dpp:+.1f}pp   spec tok/s median {mo:.2f} -> {mn:.2f} = {dt:+.1f}%")
        if jsonl:
            with open(jsonl, "a") as f:
                f.write(json.dumps({"tag": tag, "kind": "repeat", "model": model,
                                    "prompt": prompt, "k": int(k), "ngen": ngen,
                                    "delta_pp": dpp, "delta_spec_pct": dt,
                                    "arms": out}) + "\n")
            print(f"appended 1 row -> {jsonl}")


if __name__ == "__main__":
    main()
