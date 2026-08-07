#!/usr/bin/env python3
"""iso-gap serve A/B part 2: the MID-STREAM flip (X first, Y arrives later) — the exact
REF_LOAD shape, and the mechanism of the receipt's MOVING byte (1347 -> 2379 across runs).

Prediction from part 1 (H-B): X runs the solo program until Y's arrival tick, then flips to
the batched body; the output is a mixture whose divergence-from-O1 point tracks Y's ARRIVAL
TIME — re-running with the same delay may still move the byte (tick alignment jitter), and a
different delay MUST move it. The depth Y sits at stays irrelevant.

  O4a  X first, Y arrives at ~2.0s  (X a few hundred tokens deep)
  O4b  same delay, repeat            (timing jitter -> byte may move)
  O5   X first, Y arrives at ~6.0s  (byte MUST move later vs O4a)
Compare each against part 1's O1 (solo default reference).
"""
import json, os, sys, threading, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
spec = importlib.util.spec_from_file_location(
    "sab", os.path.join(os.path.dirname(os.path.abspath(__file__)), "serve-ab.py"))
sab = importlib.util.module_from_spec(spec); spec.loader.exec_module(sab)

OUT = sab.OUT

def run_arm_xfirst(name, delay):
    p, lf = sab.boot({}, os.path.join(OUT, f"serveab-{name}-server.log"))
    try:
        xr = {}
        xt = threading.Thread(target=lambda: xr.update(sab.post(sab.X_PROMPT, sab.X_TOKENS)))
        xt.start()
        time.sleep(delay)
        yr = sab.post(sab.Y_PROMPT, sab.Y_TOKENS)
        xt.join()
        text = sab.full_text(xr)
        assert len(text) > 200, f"{name}: near-empty stream"
        with open(os.path.join(OUT, f"serveab-{name}.txt"), "w") as f:
            f.write(text)
        print(f"  [{name}] X bytes={len(text)} Y completion={yr.get('usage',{}).get('completion_tokens',-1)}")
        return text
    finally:
        sab.stop(p, lf)

def main():
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    o1 = open(os.path.join(OUT, "serveab-O1.txt")).read()
    v = {}
    for name, delay in [("O4a", 2.0), ("O4b", 2.0), ("O5", 6.0)]:
        print(f"== arm {name} (X first, Y at {delay}s) ==")
        t = run_arm_xfirst(name, delay)
        d = sab.diff(o1, t)
        v[f"{name}_vs_O1"] = "IDENTICAL" if d is None else f"diverges at byte {d}"
    print(json.dumps(v, indent=1))
    with open(os.path.join(OUT, "serveab2-verdicts.json"), "w") as f:
        json.dump(v, f, indent=1)

if __name__ == "__main__":
    main()
