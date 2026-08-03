#!/usr/bin/env python3
"""Parse a run-perf.sh raw log into per-arm prefill medians.

Parses the LOG, never a pipe (a CUDA error or panic must keep its text). Every arm's samples are
printed with N, and an arm whose runs did not all produce a prefill line is reported as incomplete
rather than silently medianed over whatever survived.

usage: parse-perf.py <label>
"""
import re
import statistics
import sys
from pathlib import Path

LANE = Path("/home/avifenesh/projects/wt-w4a4/research/w4a4-rescue-20260803")


def main() -> int:
    label = sys.argv[1]
    log = LANE / "logs" / f"{label}-perf.log"

    arm_re = re.compile(r"^=== ARM (\S+) round (\d+) ===$")
    pp_re = re.compile(r"^prefill (\d+) tok in ([0-9.]+)s = ([0-9.]+) tok/s")

    samples: dict[str, list[float]] = {}
    rounds: dict[str, int] = {}
    cur = None
    for line in log.read_text().splitlines():
        m = arm_re.match(line)
        if m:
            cur = m.group(1)
            rounds[cur] = rounds.get(cur, 0) + 1
            continue
        m = pp_re.match(line)
        if m and cur:
            samples.setdefault(cur, []).append(float(m.group(3)))

    order = ["w4a8", "w4a4-k0", "w4a4-k16", "w4a4-k32"]
    base = None
    print(f"{'arm':<10} {'N':>2} {'median':>9} {'min':>9} {'max':>9} {'spread':>7} {'vs w4a8':>8}")
    for a in order:
        s = samples.get(a, [])
        if not s:
            print(f"{a:<10} -- NO SAMPLES ({rounds.get(a, 0)} runs launched)")
            continue
        med = statistics.median(s)
        if a == "w4a8":
            base = med
        spread = (max(s) - min(s)) / med * 100
        ratio = f"{med / base:.3f}x" if base else "--"
        flag = "" if len(s) == rounds.get(a, 0) else f"  INCOMPLETE {len(s)}/{rounds[a]}"
        print(f"{a:<10} {len(s):>2} {med:>9.1f} {min(s):>9.1f} {max(s):>9.1f} "
              f"{spread:>6.1f}% {ratio:>8}{flag}")
    print(f"\nraw: {log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
