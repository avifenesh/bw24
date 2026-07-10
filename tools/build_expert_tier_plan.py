#!/usr/bin/env python3
"""Convert bw24 routing traces into frozen per-layer expert quantization plans."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any


FORMAT = "bw24-expert-tier-plan-v2"
RECIPES = ("uniform-nvfp4", "usage-pyramid", "reap50-plus25")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_layers(raw: str) -> list[int]:
    if "-" in raw:
        lo, hi = (int(x) for x in raw.split("-", 1))
        if lo > hi:
            raise ValueError("layer range is descending")
        return list(range(lo, hi + 1))
    return [int(x) for x in raw.split(",") if x]


def read_trace(paths: list[Path], layers: list[int], n_expert: int) -> tuple[dict[int, list[int]], int]:
    counts = {layer: [0] * n_expert for layer in layers}
    events = 0
    for path in paths:
        with path.open() as handle:
            for line_no, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                if line.startswith("{"):
                    row = json.loads(line)
                    layer = int(row["layer"])
                    if "experts" in row:
                        pairs = [(int(x["expert"]), int(x["count"])) for x in row["experts"]]
                    else:
                        pairs = [(int(x), 1) for x in row["selected"]]
                else:
                    fields = line.split(maxsplit=2)
                    if len(fields) != 3:
                        raise ValueError(f"{path}:{line_no}: expected 'LAYER TOKENS ID,ID,...'")
                    layer = int(fields[0])
                    pairs = [(int(x), 1) for x in fields[2].split(",") if x]
                if layer not in counts:
                    continue
                for expert, count in pairs:
                    if expert < 0 or expert >= n_expert:
                        raise ValueError(f"{path}:{line_no}: expert {expert} outside 0..{n_expert - 1}")
                    counts[layer][expert] += count
                events += 1
    return counts, events


def _take_ranked(ids: list[int], counts: list[int], n: int, hottest: bool) -> set[int]:
    ranked = sorted(ids, key=lambda ex: ((-counts[ex], ex) if hottest else (counts[ex], ex)))
    return set(ranked[:n])


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    layers = parse_layers(args.layers)
    if args.recipe == "uniform-nvfp4":
        if args.trace:
            raise ValueError("uniform-nvfp4 must not depend on calibration traces")
        if args.prune_unused:
            raise ValueError("uniform-nvfp4 cannot prune experts")
        counts = {layer: [0] * args.expert_count for layer in layers}
        events = 0
    else:
        counts, events = read_trace(args.trace, layers, args.expert_count)
        if events == 0:
            raise ValueError("no trace records matched the selected layers")
    assignments: list[dict[str, Any]] = []
    pruned: dict[str, list[int]] = {}
    summaries: dict[str, Any] = {}

    for layer in layers:
        c = counts[layer]
        inactive = {ex for ex, count in enumerate(c) if count == 0} if args.prune_unused else set()
        retained = [ex for ex in range(args.expert_count) if ex not in inactive]
        if len(retained) < args.top_k:
            raise ValueError(f"layer {layer}: pruning leaves {len(retained)} experts, below top_k={args.top_k}")
        tiers: dict[str, set[int]]
        if args.recipe == "uniform-nvfp4":
            tiers = {"NVFP4": set(retained)}
        elif args.recipe == "usage-pyramid":
            hot_n = max(1, math.ceil(len(retained) * args.hot_fraction))
            low_n = max(1, math.ceil(len(retained) * args.low_fraction))
            if hot_n + low_n > len(retained):
                low_n = len(retained) - hot_n
            hot = _take_ranked(retained, c, hot_n, hottest=True)
            remaining = [ex for ex in retained if ex not in hot]
            low = _take_ranked(remaining, c, low_n, hottest=False)
            tiers = {"NVFP4": hot, "Q2_K": low, "Q3_K": set(retained) - hot - low}
        else:
            if args.expert_count * 2 != args.original_expert_count:
                raise ValueError(
                    "reap50-plus25 expects the source to retain exactly 50% of original experts "
                    f"({args.expert_count} vs original {args.original_expert_count})"
                )
            if inactive:
                raise ValueError("reap50-plus25 does not apply an additional unused-expert prune")
            q2_n = round(args.original_expert_count * 0.25)
            low = _take_ranked(retained, c, q2_n, hottest=False)
            tiers = {"Q2_K": low, "NVFP4": set(retained) - low}

        for qtype in ("NVFP4", "Q3_K", "Q2_K"):
            ids = sorted(tiers.get(qtype, set()))
            if ids:
                assignments.append({"layer": layer, "experts": ids, "qtype": qtype})
        if inactive:
            pruned[str(layer)] = sorted(inactive)
        summaries[str(layer)] = {
            "assignments": sum(c),
            "observed_experts": sum(x > 0 for x in c),
            "pruned": len(inactive),
            "nvfp4": len(tiers.get("NVFP4", set())),
            "q3_k": len(tiers.get("Q3_K", set())),
            "q2_k": len(tiers.get("Q2_K", set())),
        }

    return {
        "format": FORMAT,
        "recipe": args.recipe,
        "description": {
            "uniform-nvfp4": "All retained experts NVFP4; no calibration or pruning",
            "usage-pyramid": "Top 25% NVFP4, middle 50% Q3_K, bottom 25% Q2_K, zero-count experts pruned",
            "reap50-plus25": "REAP50 source: least-used 25% of original bank Q2_K, remaining 25% NVFP4",
        }[args.recipe],
        "model": {
            "expert_count": args.expert_count,
            "original_expert_count": args.original_expert_count,
            "expert_used_count": args.top_k,
            "moe_layers": layers,
        },
        "policy": {
            "rank_metric": None if args.recipe == "uniform-nvfp4" else "router_selection_count",
            "hot_fraction": args.hot_fraction if args.recipe == "usage-pyramid" else None,
            "low_fraction": args.low_fraction if args.recipe == "usage-pyramid" else None,
            "prune_unused": args.prune_unused,
            "tie_break": "ascending expert id",
        },
        "calibration": {
            "trace_files": [
                {"path": str(path.resolve()), "sha256": sha256(path)} for path in args.trace
            ],
            "matched_events": events,
            "public_eval_data_used_for_selection": False,
        },
        "pruned_experts": pruned,
        "assignments": assignments,
        "layer_summary": summaries,
    }


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="bw24-tier-plan-") as tmp:
        root = Path(tmp)
        trace = root / "usage.trace"
        trace.write_text("1 1 0,0,1,2\n1 1 0,1,2,3\n")
        args = argparse.Namespace(
            trace=[trace], recipe="usage-pyramid", expert_count=5, original_expert_count=5,
            top_k=2, layers="1", hot_fraction=0.25, low_fraction=0.25, prune_unused=True,
        )
        plan = build_plan(args)
        summary = plan["layer_summary"]["1"]
        assert summary == {
            "assignments": 8, "observed_experts": 4, "pruned": 1,
            "nvfp4": 1, "q3_k": 2, "q2_k": 1,
        }
        assert plan["pruned_experts"] == {"1": [4]}
        uniform = build_plan(argparse.Namespace(
            trace=[], recipe="uniform-nvfp4", expert_count=4, original_expert_count=4,
            top_k=2, layers="1", hot_fraction=0.25, low_fraction=0.25,
            prune_unused=False,
        ))
        uniform_summary = uniform["layer_summary"]["1"]
        assert uniform_summary["nvfp4"] == 4
        assert uniform_summary["q3_k"] == 0 and uniform_summary["q2_k"] == 0
        assert uniform["calibration"]["trace_files"] == []
        reap_trace = root / "reap.trace"
        reap_trace.write_text("1 1 " + ",".join(str(x) for x in range(96)) + "\n")
        reap = build_plan(argparse.Namespace(
            trace=[reap_trace], recipe="reap50-plus25", expert_count=96,
            original_expert_count=192, top_k=8, layers="1",
            hot_fraction=0.25, low_fraction=0.25, prune_unused=False,
        ))
        reap_summary = reap["layer_summary"]["1"]
        assert reap_summary["q2_k"] == 48 and reap_summary["nvfp4"] == 48
        assert reap_summary["q3_k"] == 0 and reap_summary["pruned"] == 0
        print("expert tier plan self-test: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, action="append", default=[])
    parser.add_argument("--recipe", choices=RECIPES)
    parser.add_argument("--expert-count", type=int)
    parser.add_argument("--original-expert-count", type=int)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--layers", help="inclusive range (1-79) or comma-separated ids")
    parser.add_argument("--hot-fraction", type=float, default=0.25)
    parser.add_argument("--low-fraction", type=float, default=0.25)
    parser.add_argument("--prune-unused", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    required = (args.recipe, args.expert_count, args.original_expert_count, args.layers, args.out)
    if not all(x is not None for x in required):
        parser.error("--recipe, --expert-count, --original-expert-count, --layers, and --out are required")
    if args.recipe != "uniform-nvfp4" and not args.trace:
        parser.error("--trace is required for usage-ranked recipes")
    if not (0 < args.hot_fraction < 1 and 0 < args.low_fraction < 1):
        parser.error("tier fractions must be between 0 and 1")
    plan = build_plan(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
