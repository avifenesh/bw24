#!/usr/bin/env python3
"""Parse a regime-accept-<tag>.log into the per-prompt regime acceptance table.

Three passes are recorded (OFF, ON, OFF2). OFF vs OFF2 is the BOX-STABILITY control: the two
passes are the same configuration in different server processes, so any disagreement between
them bounds how much of an OFF-vs-ON delta could be drift rather than the arm. A cell is only
reported as an arm effect when OFF == OFF2.

Usage: parse_regime.py <regime-accept-tag.log> [--jsonl out.jsonl] [--tag TAG]
"""
import json
import re
import sys


def main():
    path = sys.argv[1]
    jsonl = sys.argv[sys.argv.index("--jsonl") + 1] if "--jsonl" in sys.argv else None
    tag = (sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv
           else path.split("regime-accept-")[-1].replace(".log", ""))

    model = draft = k = "?"
    rows = {}  # (arm, prompt) -> dict
    arm = prompt = None
    for ln in open(path).read().splitlines():
        if ln.startswith("[model] "):
            model = ln.split(None, 1)[1]
        elif ln.startswith("[draft] "):
            draft = ln.split(None, 1)[1]
        elif ln.startswith("[spec K] "):
            k = ln.split(None, 2)[2]
        m = re.match(r"=== PASS \d+ ARM (\w+)", ln)
        if m:
            arm = m.group(1)
            continue
        m = re.match(r"--- prompt (\S+)", ln)
        if m:
            prompt = m.group(1)
            continue
        if ln.startswith("USAGE ") and arm and prompt:
            u = json.loads(re.search(r"USAGE (\{.*?\}) SPEC", ln).group(1))
            s = json.loads(re.search(r"SPEC (\{.*\})$", ln).group(1))
            rows[(arm, prompt)] = {"usage": u, "spec": s}
        if ln.startswith("TEXT_SHA ") and arm and prompt:
            rows.setdefault((arm, prompt), {})["sha"] = ln.split()[1]

    prompts = []
    for (a, p) in rows:
        if p not in prompts:
            prompts.append(p)
    print(f"model={model}\ndraft={draft}\nK={k}\n")
    hdr = ("| prompt | pfx tok | OFF acc | ON acc | Dpp | OFF2 acc | box stable? | "
           "greedy text | OFF s | ON s |")
    print(hdr)
    print("|" + "---|" * 10)
    out = []
    for p in prompts:
        o, n, o2 = rows.get(("OFF", p)), rows.get(("ON", p)), rows.get(("OFF2", p))
        if not (o and n):
            continue
        def frac(r):
            return f"{r['spec']['accepted']}/{r['spec']['drafted']}"
        def pct(r):
            return r["spec"]["acceptance_rate"] * 100
        dpp = pct(n) - pct(o)
        stable = (o2 is not None and frac(o2) == frac(o))
        same_text = o.get("sha") == n.get("sha")
        print(f"| {p} | {o['usage']['prompt_tokens']} | {pct(o):.1f}% ({frac(o)}) | "
              f"{pct(n):.1f}% ({frac(n)}) | **{dpp:+.1f}** | "
              f"{pct(o2):.1f}% ({frac(o2)}) | {'YES' if stable else '**NO**'} | "
              f"{'IDENTICAL' if same_text else '**DIFFERS**'} | "
              f"{o['usage']['elapsed_s']:.3f} | {n['usage']['elapsed_s']:.3f} |")
        out.append({"prompt": p, "prompt_tokens": o["usage"]["prompt_tokens"],
                    "off": o["spec"], "on": n["spec"],
                    "off2": (o2 or {}).get("spec"), "delta_pp": dpp,
                    "box_stable": stable, "greedy_text_identical": same_text,
                    "off_elapsed_s": o["usage"]["elapsed_s"],
                    "on_elapsed_s": n["usage"]["elapsed_s"],
                    "off_sha": o.get("sha"), "on_sha": n.get("sha")})
    if out:
        mean = sum(r["delta_pp"] for r in out) / len(out)
        worst = min(out, key=lambda r: r["delta_pp"])
        print(f"\nmean Dpp over {len(out)} prompts = {mean:+.2f}pp   "
              f"worst = {worst['prompt']} {worst['delta_pp']:+.1f}pp")
        print(f"box-stability control: OFF==OFF2 on "
              f"{sum(1 for r in out if r['box_stable'])}/{len(out)} prompts")
        if jsonl:
            with open(jsonl, "a") as f:
                f.write(json.dumps({"tag": tag, "kind": "regime-accept", "model": model,
                                    "draft": draft, "k": k, "mean_delta_pp": mean,
                                    "worst_prompt": worst["prompt"],
                                    "worst_delta_pp": worst["delta_pp"],
                                    "rows": out}) + "\n")
            print(f"appended 1 row -> {jsonl}")


if __name__ == "__main__":
    main()
