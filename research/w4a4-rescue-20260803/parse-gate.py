#!/usr/bin/env python3
"""Parse a w4a4-gate raw log into one JSONL row per cell.

Parses the LOG, never the pipe: run-gate.sh tees the raw output first so a CUDA error or a panic
keeps its text, and this reads that file. Cells whose gate process died carry the quoted failure
line rather than a verdict, so a capacity failure is never silently scored as an exactness result.

usage: parse-gate.py <label> <stage>   e.g. parse-gate.py baseline2 baseline
"""
import json
import re
import sys
from pathlib import Path

LANE = Path("/home/avifenesh/projects/wt-w4a4/research/w4a4-rescue-20260803")

# Fields worth carrying into the summary row. The token streams and decoded text stay in the raw
# log: they are the evidence, but they are not a summary.
KEEP = [
    "cell", "model", "prompt_tokens", "ngen",
    "ref_prefill_argmax", "ref_decode_argmax", "ref_self_maxdiff", "ref_prime_argmax",
    "test_prefill_argmax", "test_decode_argmax", "test_self_maxdiff", "test_prime_argmax",
    "first_divergent_pos", "ref_token", "test_token",
    "cross_prefill_maxdiff", "cross_prime_maxdiff",
    "ref_logit_ref_id", "ref_logit_test_id", "test_logit_ref_id", "test_logit_test_id",
    "ref_margin", "test_margin", "div_row_maxdiff",
    "ref_entrypoint_floor_pos", "test_entrypoint_floor_pos",
    "ref_entry_noise_at_div", "margin_within_entry_noise",
]


def main() -> int:
    label, stage = sys.argv[1], sys.argv[2]
    log = LANE / "logs" / f"{label}-gate.log"
    out = LANE / f"{stage}.jsonl"

    rows = []
    header = re.compile(r"^#{10} (\S+) / (\S+) #{10}$")
    cur_model, cur_cell, buf = None, None, []

    def flush():
        if cur_model is None:
            return
        row = {"stage": stage, "model_tag": cur_model, "cell_tag": cur_cell}
        # The k the W4A4 arm ran under, echoed into the log by run-gate.sh. Recorded per row so a
        # summary file can never be read as belonging to the wrong sweep point.
        rk = next((l for l in buf if l.startswith("MEMRA_MMQ_RESIDUAL_K=")), None)
        if rk:
            row["residual_k"] = int(rk.split("=", 1)[1])
        payload = next((l for l in buf if l.startswith("JSONL ")), None)
        if payload:
            parsed = json.loads(payload[len("JSONL "):])
            row.update({k: parsed[k] for k in KEEP if k in parsed})
            # Divergence is measured on the SERVING stream (prime_cache + greedy decode), so keep
            # the generated ids for the diff and the length of the agreeing prefix.
            row["ngen_agree"] = (
                parsed["ngen"] if parsed["first_divergent_pos"] is None
                else parsed["first_divergent_pos"]
            )
            row["verdict"] = parsed["verdict"]
        else:
            # No JSONL line => the process died before emitting. Quote the failure, do not infer it.
            err = [l for l in buf if l.startswith("Error:") or "panicked at" in l]
            row["verdict"] = "DIED"
            row["failure_quoted"] = err[-1].strip() if err else "died, cause unknown — repro needed"
        rows.append(row)

    for line in log.read_text().splitlines():
        m = header.match(line)
        if m:
            flush()
            cur_model, cur_cell, buf = m.group(1), m.group(2), []
            continue
        buf.append(line)
    flush()

    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    for r in rows:
        extra = r.get("failure_quoted", f"agree {r.get('ngen_agree')}/{r.get('ngen')} tokens")
        print(f"{r['model_tag']:>4} {r['cell_tag']:<16} {r['verdict']:<10} {extra}")
    print(f"-> {out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
