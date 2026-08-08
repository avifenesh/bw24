#!/usr/bin/env python3
"""Compare the two arms' generated `tokens:` lines inside one argmax-<tag>.log.

Prints the cross-arm first-divergence index, the internal MATCH/MISMATCH tally per arm, and
(when a golden is given) whether each arm still matches the pinned golden prefix.

Usage: cmp_arms.py <argmax-tag.log> [golden.tokens]
"""
import re
import sys


def toks(line):
    return [int(x) for x in re.findall(r"-?\d+", line.split("[", 1)[1])]


def main():
    path = sys.argv[1]
    lines = open(path).read().splitlines()
    arms, cur = {}, None
    for ln in lines:
        m = re.match(r"=== ARM (\w+)", ln)
        if m:
            cur = m.group(1)
            arms[cur] = {"gen": None, "match": 0, "mismatch": 0, "maxdiff": []}
            continue
        if cur is None:
            continue
        a = arms[cur]
        if ln.startswith("tokens: ["):
            a["gen"] = toks(ln)
        elif "MATCH" in ln and "MISMATCH" not in ln:
            a["match"] += 1
            md = re.search(r"maxdiff=([\d.eE+-]+)", ln)
            if md:
                a["maxdiff"].append(md.group(1))
        elif "MISMATCH" in ln:
            a["mismatch"] += 1

    for name, a in arms.items():
        n = len(a["gen"]) if a["gen"] else 0
        print(
            f"ARM {name:3s}  internal MATCH={a['match']} MISMATCH={a['mismatch']}  "
            f"ngen={n}  logit_maxdiff={','.join(a['maxdiff'])}"
        )

    if "OFF" in arms and "ON" in arms and arms["OFF"]["gen"] and arms["ON"]["gen"]:
        o, n = arms["OFF"]["gen"], arms["ON"]["gen"]
        k = min(len(o), len(n))
        div = next((i for i in range(k) if o[i] != n[i]), None)
        if div is None and len(o) == len(n):
            print(f"CROSS-ARM: IDENTICAL ({len(o)} tokens)")
        elif div is None:
            print(f"CROSS-ARM: prefix-identical to {k}, lengths differ {len(o)} vs {len(n)}")
        else:
            print(
                f"CROSS-ARM: DIVERGE at generated index {div} "
                f"(OFF={o[div]} ON={n[div]}); identical prefix = {div}/{k} tokens"
            )

    if len(sys.argv) > 2:
        g = None
        for ln in open(sys.argv[2]).read().splitlines():
            if ln.startswith("tokens: ["):
                g = toks(ln)
        if g:
            for name, a in arms.items():
                if not a["gen"]:
                    continue
                ok = a["gen"][: len(g)] == g
                print(f"GOLDEN({len(g)}) vs ARM {name}: {'MATCH' if ok else 'MISMATCH'}")


if __name__ == "__main__":
    main()
