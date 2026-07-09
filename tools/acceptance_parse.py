#!/usr/bin/env python3
"""Parse ONE run-spec invocation's merged stdout+stderr into a single JSONL acceptance row.

Contract: the caller runs run-spec with BW24_SPEC_K=<k> (exactly one K) and BW24_SPEC_STATS=1,
merges stderr into stdout (2>&1), and pipes it here. Metadata (arm/prompt/run/model/...) comes via
flags; the numeric acceptance/per-slot/tok-s fields are scraped from the output. The row is APPENDED
to --out. Used by acceptance_battery.sh and agent_loop_acceptance.sh — the bf16-vs-nvfp4 deliverable
(MTP-heal protocol, see HANDOVER "BW24 DUAL-SHAPE").

run-spec lines consumed (see crates/bw24-engine/src/bin/run_spec.rs + spec.rs [spec-stats]):
  [generate]   31 tok in 0.500s = 62.00 tok/s (gen-only; this run's prime 0.100s)
  [generate_spec K=3] 32 tok in 0.400s = 77.50 tok/s (1.25x vs generate; this run's prime 0.100s)
    acceptance: 27/40 = 67.5%   self-consistency: PASS (identical to generate)
  [spec-stats] rounds=14 full_accept=6 len_hist=[..] per_slot=[27/40=0.675 ..] total=27/40=0.675 ..
"""
import argparse, json, re, sys, time

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="JSONL file to append the row to")
    ap.add_argument("--arm", required=True, help="e.g. bf16-fullprec | nvfp4")
    ap.add_argument("--prompt", required=True, help="prompt id, e.g. p1 / p2 / p3 / agentloop-t3")
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--run", type=int, required=True)
    ap.add_argument("--model", default="")
    ap.add_argument("--ngen", type=int, default=0)
    ap.add_argument("--full-prec", action="store_true")
    ap.add_argument("--extra", default="", help="free-form note (e.g. arm env)")
    args = ap.parse_args()

    text = sys.stdin.read()

    row = {
        "arm": args.arm, "model": args.model, "prompt": args.prompt, "k": args.k,
        "run": args.run, "ngen": args.ngen, "full_prec": bool(args.full_prec),
        "accepted": None, "drafted": None, "acc_rate": None, "per_slot": None,
        "tok_s": None, "speedup": None, "gen_tok_s": None,
        "self_consistency": None, "extra": args.extra, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    m = re.search(r"acceptance:\s*(\d+)/(\d+)\s*=\s*([\d.]+)%", text)
    if m:
        row["accepted"], row["drafted"] = int(m.group(1)), int(m.group(2))
        row["acc_rate"] = float(m.group(3)) / 100.0
    m = re.search(r"self-consistency:\s*(PASS|FAIL)", text)
    if m:
        row["self_consistency"] = m.group(1)
    m = re.search(r"\[generate_spec K=\d+\][^=]*=\s*([\d.]+)\s*tok/s\s*\(([\d.]+)x", text)
    if m:
        row["tok_s"], row["speedup"] = float(m.group(1)), float(m.group(2))
    m = re.search(r"\[generate\]\s+\d+\s*tok[^=]*=\s*([\d.]+)\s*tok/s", text)
    if m:
        row["gen_tok_s"] = float(m.group(1))
    m = re.search(r"per_slot=\[([^\]]*)\]", text)
    if m:
        slots = []
        for tok in m.group(1).split():
            mm = re.search(r"=([\d.]+)$", tok)
            slots.append(float(mm.group(1)) if mm else None)
        row["per_slot"] = slots

    if row["acc_rate"] is None:
        # surface the failure but still emit a row so the battery is auditable
        row["error"] = "no acceptance line parsed (run-spec failed or crashed)"
        tail = "\n".join(text.strip().splitlines()[-4:])
        row["tail"] = tail
        sys.stderr.write(f"[acceptance_parse] WARN no acceptance for {args.arm}/{args.prompt}/K{args.k}/run{args.run}\n")

    with open(args.out, "a") as f:
        f.write(json.dumps(row) + "\n")

if __name__ == "__main__":
    main()
