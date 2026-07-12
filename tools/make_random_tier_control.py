#!/usr/bin/env python3
"""Create a seeded per-layer random control matched to a frozen tier plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def random_control(plan: dict, seed: int) -> dict:
    if plan.get("format") != "bw24-expert-tier-plan-v2":
        raise ValueError("input is not a v2 expert tier plan")
    by_layer: dict[int, Counter[str]] = defaultdict(Counter)
    for group in plan["assignments"]:
        by_layer[int(group["layer"])][group["qtype"]] += len(group["experts"])
    n_expert = int(plan["model"]["expert_count"])
    pruned = {int(layer): set(ids) for layer, ids in plan.get("pruned_experts", {}).items()}
    assignments = []
    for layer in [int(x) for x in plan["model"]["moe_layers"]]:
        ids = [ex for ex in range(n_expert) if ex not in pruned.get(layer, set())]
        random.Random((seed << 16) ^ layer).shuffle(ids)
        offset = 0
        for qtype in ("Q8_0", "NVFP4", "Q3_K", "Q2_K"):
            count = by_layer[layer][qtype]
            if count:
                assignments.append({
                    "layer": layer,
                    "experts": sorted(ids[offset : offset + count]),
                    "qtype": qtype,
                })
            offset += count
        if offset != len(ids):
            raise ValueError(f"layer {layer}: tier counts do not cover retained experts")
    out = dict(plan)
    out["recipe"] = "random-budget-control"
    out["description"] = f"Seeded random assignment matched to {plan.get('recipe')} tier counts"
    out["random_control"] = {
        "seed": seed,
        "matched_recipe": plan.get("recipe"),
        "matched_plan_sha256": None,
        "per_layer_seed_formula": "(seed << 16) XOR layer",
    }
    out["assignments"] = assignments
    out.pop("layer_summary", None)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raw = args.plan.read_bytes()
    control = random_control(json.loads(raw), args.seed)
    control["random_control"]["matched_plan_sha256"] = hashlib.sha256(raw).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(control, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
