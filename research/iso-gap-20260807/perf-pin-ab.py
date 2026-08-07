#!/usr/bin/env python3
"""iso-gap perf A/B: the COST of the byte-equality pin (MEMRA_SERVE_B1FAST=0 MEMRA_SERVE_GS=0)
on solo greedy serving — the mission's 'if honest per-session selection costs perf, measure it'
clause, at serve level, fresh, N=5 interleaved.

WHY THIS SHAPE. The lane's finding: the staggered-depth receipt is carried by the solo<->batched
program flip at the co-residence boundary. 'Selection keyed on the session's own state' applied
here means ONE program family regardless of co-residents — i.e. the batched body always (the m=1
fused trunk and GraphSession replay are structurally solo-only; there is no per-session way to
keep them inside a batch). The pin is the deployment lever for byte-equality; this measures what
it costs where it bites hardest (c=1, the solo regime the fused trunk/graph exist for).

Arms interleaved A,B,A,B,... x5 pairs (H100-lane law: same-session adjacency, no cross-day
denominators), one server boot per run (the program families differ at boot), spec OFF both arms
(the flip class is the non-spec interactive path; spec's own gate is separately byte-exact).
Metric: completion_tokens / wall seconds of a fixed 512-token greedy request, single stream.
"""
import json, os, statistics, subprocess, sys, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
spec = importlib.util.spec_from_file_location(
    "sab", os.path.join(os.path.dirname(os.path.abspath(__file__)), "serve-ab.py"))
sab = importlib.util.module_from_spec(spec); spec.loader.exec_module(sab)

OUT = sab.OUT
N = 5
TOK = 512
ARMS = {
    "default": {},
    "pinned": {"MEMRA_SERVE_B1FAST": "0", "MEMRA_SERVE_GS": "0"},
}

def one_run(name, env, rep):
    p, lf = sab.boot(env, os.path.join(OUT, f"perfab-{name}-r{rep}-server.log"))
    try:
        # warm the weights/caches: one 32-token request, discarded
        sab.post("Warm up.", 32)
        t0 = time.monotonic()
        r = sab.post(sab.X_PROMPT, TOK)
        dt = time.monotonic() - t0
        toks = r.get("usage", {}).get("completion_tokens", 0)
        assert toks >= TOK - 2, f"short stream ({toks})"
        return toks / dt
    finally:
        sab.stop(p, lf)

def main():
    os.makedirs(OUT, exist_ok=True)
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    rows = {a: [] for a in ARMS}
    for rep in range(N):
        for name, env in ARMS.items():   # adjacent within each rep
            tps = one_run(name, env, rep)
            rows[name].append(tps)
            print(f"rep {rep} {name}: {tps:.2f} tok/s", flush=True)
    med = {a: statistics.median(v) for a, v in rows.items()}
    ratio = med["pinned"] / med["default"]
    res = {"n": N, "tok": TOK, "runs": rows, "medians": med,
           "pinned_over_default": ratio,
           "note": "solo (c=1) greedy serve, spec OFF, warm boot, rep-adjacent interleave"}
    print(json.dumps(res, indent=1))
    with open(os.path.join(OUT, "perfab-results.json"), "w") as f:
        json.dump(res, f, indent=1)

if __name__ == "__main__":
    main()
