#!/usr/bin/env python3
"""Lane 1 q27 MTP spec sweep (darklanes-8x, GPU 1 ONLY).

Phase A (default): fixed-K grid — K in {2,3,4,5} x 3 prompt classes x N=3,
interleaved (run outer, class/K inner) so cross-K comparisons share the clock.
Each cell is one run-spec invocation: plain-generate oracle + generate_spec at K,
gen-only timing (prime subtracted by the binary), greedy self-consistency gate.

Appends one JSON row per run to sweep.jsonl; raw log per run under logs/
(evidence discipline: tee raw first, parse the log second).

Usage on box:  python3 sweep.py            # phase A grid
               python3 sweep.py knob NAME=VAL[,NAME=VAL] K CLASS [CLASS..]  # phase B cells, N=3
"""
import json
import os
import re
import subprocess
import sys
import time

HOME = os.path.expanduser("~")
LANE = f"{HOME}/lane1"
MODEL = "/opt/dlami/nvme/models/Qwen3.6-27B-Q4_K_M.gguf"
OUT = f"{LANE}/research/q27-mtp-20260801"
LOGD = f"{OUT}/logs"
JSONL = f"{OUT}/sweep.jsonl"
PROMPTS = {
    "short": f"{LANE}/research/e2e/prompts/p1-code-short.txt",
    "board2048": f"{LANE}/research/e2e/prompts/board-2048.txt",
    "agentic500": f"{OUT}/prompt-agentic-500w.txt",
}
NGEN = 256

os.makedirs(LOGD, exist_ok=True)


def run_cell(cls, k, run, extra=None, tag="fixedk"):
    flag_id = "-".join(f"{n.replace('MEMRA_','').lower()}{v}" for n, v in (extra or {}).items())
    lf = f"{LOGD}/{tag}{('-' + flag_id) if flag_id else ''}-{cls}-k{k}-r{run}.log"
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="1",
        MEMRA_SPEC_K=str(k),
        MEMRA_NGEN=str(NGEN),
        MEMRA_PROMPT_FILE=PROMPTS[cls],
    )
    if extra:
        env.update(extra)
    t0 = time.time()
    with open(lf, "w") as f:
        rc = subprocess.call(
            [f"{LANE}/target/release/run-spec", MODEL],
            stdout=f, stderr=subprocess.STDOUT, env=env, cwd=LANE,
        )
    txt = open(lf, errors="replace").read()
    row = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "lane": 1, "gpu": 1, "tag": tag,
        "prompt_class": cls, "k": k, "run": run, "ngen": NGEN, "rc": rc,
        "flags": extra or {}, "log": os.path.relpath(lf, OUT),
        "wall_s": round(time.time() - t0, 1),
    }
    m = re.search(r"\[generate\]\s+\d+ tok in [\d.]+s = ([\d.]+) tok/s", txt)
    if m:
        row["plain_tok_s"] = float(m.group(1))
    m = re.search(
        r"\[generate_spec K=%d\] \d+ tok in [\d.]+s = ([\d.]+) tok/s \(([\d.]+)x" % k, txt)
    if m:
        row["tok_s"] = float(m.group(1))
        row["speedup_vs_plain"] = float(m.group(2))
    m = re.search(r"acceptance: (\d+)/(\d+) = ([\d.]+)%\s+self-consistency: (\w+)", txt)
    if m:
        row["accepted"] = int(m.group(1))
        row["drafted"] = int(m.group(2))
        row["acceptance_pct"] = float(m.group(3))
        row["self_consistency"] = m.group(4)
    with open(JSONL, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)
    return row


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "knob":
        extra = dict(kv.split("=", 1) for kv in sys.argv[2].split(","))
        k = int(sys.argv[3])
        classes = sys.argv[4:] or list(PROMPTS)
        for run in (1, 2, 3):
            for cls in classes:
                run_cell(cls, k, run, extra=extra, tag="knob")
        return
    for run in (1, 2, 3):
        for cls in ("short", "board2048", "agentic500"):
            for k in (2, 3, 4, 5):
                run_cell(cls, k, run)


if __name__ == "__main__":
    main()
