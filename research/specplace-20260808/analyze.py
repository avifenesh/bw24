#!/usr/bin/env python3
"""Summarize the spec-placement S/N sweep from load-serve JSONL."""

from __future__ import annotations

import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path


LABEL_RE = re.compile(
    r"^(?P<model>q9|step35)-(?P<placement>sc|pp2)-"
    r"(?P<arm>S|N)-r(?P<rep>[1-9][0-9]*)-c(?P<c>[1-9][0-9]*)$"
)


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {Path(sys.argv[0]).name} POINTS.jsonl", file=sys.stderr)
        return 2

    path = Path(sys.argv[1])
    groups: dict[tuple[str, str, str, int], list[dict]] = defaultdict(list)
    bad_labels: list[str] = []
    total_errors = 0

    with path.open(encoding="utf-8") as src:
        for lineno, line in enumerate(src, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            match = LABEL_RE.fullmatch(row["label"])
            if match is None:
                bad_labels.append(f"{lineno}:{row['label']}")
                continue
            key = (
                match["model"],
                match["placement"],
                match["arm"],
                int(match["c"]),
            )
            groups[key].append(row)
            total_errors += int(row["n_err"]) + int(row["n_shed"])

    if bad_labels:
        print("unrecognized labels: " + ", ".join(bad_labels), file=sys.stderr)
        return 1

    print("| model | placement | c | spec ON median (min-max) | "
          "spec OFF median (min-max) | S/N | winner |")
    print("|---|---|---:|---:|---:|---:|---|")
    incomplete = False
    for model, placement in (("q9", "sc"), ("step35", "pp2")):
        for concurrency in (1, 2, 4):
            arms: dict[str, tuple[float, float, float, int]] = {}
            for arm in ("S", "N"):
                rows = groups.get((model, placement, arm, concurrency), [])
                values = [float(row["agg_tok_s"]) for row in rows]
                if len(values) < 3:
                    incomplete = True
                    arms[arm] = (float("nan"), float("nan"), float("nan"), len(values))
                else:
                    arms[arm] = (
                        statistics.median(values),
                        min(values),
                        max(values),
                        len(values),
                    )
            spec, plain = arms["S"], arms["N"]
            ratio = spec[0] / plain[0]
            winner = "spec ON" if ratio > 1.0 else "spec OFF"
            print(
                f"| {model} | {placement} | {concurrency} | "
                f"{spec[0]:.1f} ({spec[1]:.1f}-{spec[2]:.1f}), N={spec[3]} | "
                f"{plain[0]:.1f} ({plain[1]:.1f}-{plain[2]:.1f}), N={plain[3]} | "
                f"{ratio:.2f}x | {winner} |"
            )

    print()
    print(f"load errors + shed requests: {total_errors}")
    print(f"raw load points: {sum(len(rows) for rows in groups.values())}")
    if incomplete:
        print("FAIL: at least one cell has fewer than 3 runs", file=sys.stderr)
        return 1
    if total_errors:
        print("FAIL: load errors or shed requests observed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
