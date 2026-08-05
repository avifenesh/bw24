#!/usr/bin/env python3
"""Parse load-serve JSON lines / memra /metrics blobs into one compact row.

Usage:
  parse.py point   < load-serve stdout
  parse.py metrics < curl /metrics
"""
import json
import sys


def main():
    mode = sys.argv[1]
    if mode == "point":
        for line in sys.stdin:
            line = line.strip()
            if not line.startswith("{"):
                continue
            d = json.loads(line)
            print("  agg=%.2f p50lat=%.3f p95lat=%.3f nok=%d nerr=%d ntok=%d" % (
                d["agg_tok_s"], d["lat_p50_s"] or 0, d["lat_p95_s"] or 0,
                d["n_ok"], d["n_err"], d["completion_tokens_total"]))
    elif mode == "metrics":
        d = json.load(sys.stdin)
        p50 = d.get("step_p50_ms")
        p99 = d.get("step_p99_ms")
        if p50:
            print("  step p50=%.3fms p99=%.3fms -> %.2f tok/s decode-only  tokens_out=%s" % (
                p50, p99, 1000.0 / p50, d.get("tokens_out")))
        else:
            print("  " + json.dumps(d))


if __name__ == "__main__":
    main()
