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
RECIPES = (
    "uniform-nvfp4", "usage-pyramid", "reap50-plus25", "quartile-prune",
    "traffic-ladder",
)


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


def read_trace(
    paths: list[Path],
    layers: list[int],
    n_expert: int,
    top_k: int,
    expected_tokens: int | None,
) -> tuple[dict[int, list[int]], int, dict[int, int]]:
    counts = {layer: [0] * n_expert for layer in layers}
    tokens_by_layer = {layer: 0 for layer in layers}
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
                    tokens = int(row["tokens"])
                    if "experts" in row:
                        pairs = [(int(x["expert"]), int(x["count"])) for x in row["experts"]]
                    else:
                        pairs = [(int(x), 1) for x in row["selected"]]
                else:
                    fields = line.split(maxsplit=2)
                    if len(fields) != 3:
                        raise ValueError(f"{path}:{line_no}: expected 'LAYER TOKENS ID,ID,...'")
                    layer = int(fields[0])
                    tokens = int(fields[1])
                    pairs = [(int(x), 1) for x in fields[2].split(",") if x]
                assignments = sum(count for _, count in pairs)
                if assignments != tokens * top_k:
                    raise ValueError(
                        f"{path}:{line_no}: {assignments} assignments != "
                        f"tokens={tokens} * top_k={top_k}"
                    )
                if layer not in counts:
                    continue
                for expert, count in pairs:
                    if expert < 0 or expert >= n_expert:
                        raise ValueError(f"{path}:{line_no}: expert {expert} outside 0..{n_expert - 1}")
                    counts[layer][expert] += count
                tokens_by_layer[layer] += tokens
                events += 1
    if expected_tokens is not None:
        drift = {
            layer: tokens
            for layer, tokens in tokens_by_layer.items()
            if tokens != expected_tokens
        }
        if drift:
            preview = ", ".join(f"{layer}={tokens}" for layer, tokens in list(drift.items())[:8])
            raise ValueError(
                f"trace token coverage differs from expected {expected_tokens} per layer: {preview}"
            )
    return counts, events, tokens_by_layer


def read_mask(path: Path, layers: list[int], n_expert: int) -> dict[int, set[int]]:
    payload = json.loads(path.read_text())
    if payload.get("format") != "bw24-hy3-reap-mask-v1":
        raise ValueError(f"{path}: unsupported expert mask format {payload.get('format')!r}")
    specs = payload.get("layers")
    if not isinstance(specs, dict):
        raise ValueError(f"{path}: layers must be an object")
    result: dict[int, set[int]] = {}
    universe = set(range(n_expert))
    for layer in layers:
        spec = specs.get(str(layer))
        if not isinstance(spec, dict):
            raise ValueError(f"{path}: missing layer {layer}")
        retained = {int(expert) for expert in spec.get("retained_experts", [])}
        pruned = {int(expert) for expert in spec.get("pruned_experts", [])}
        if retained | pruned != universe or retained & pruned:
            raise ValueError(f"{path}: layer {layer} is not an exact partition of 0..{n_expert - 1}")
        result[layer] = pruned
    return result


def _take_ranked(ids: list[int], counts: list[int], n: int, hottest: bool) -> set[int]:
    ranked = sorted(ids, key=lambda ex: ((-counts[ex], ex) if hottest else (counts[ex], ex)))
    return set(ranked[:n])


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    layers = parse_layers(args.layers)
    if args.recipe == "traffic-ladder":
        if args.mask is not None or args.prune_unused:
            raise ValueError("traffic-ladder uses the full bank and its fixed cold-tail prune")
        if args.expert_count != args.original_expert_count:
            raise ValueError("traffic-ladder requires the full original expert bank")
        tier_counts = (args.q8_count, args.nvfp4_count, args.q2_count)
        if any(count is None for count in tier_counts):
            raise ValueError("traffic-ladder requires Q8_0, NVFP4, and Q2_K counts")
        if args.q8_count < 0 or args.nvfp4_count < 0 or args.q2_count <= 0:
            raise ValueError(
                "traffic-ladder requires a positive Q2_K count and non-negative "
                "Q8_0/NVFP4 counts"
            )
        if sum(tier_counts) > args.expert_count:
            raise ValueError("traffic-ladder tier counts exceed the expert bank")
        if sum(tier_counts) < args.top_k:
            raise ValueError("traffic-ladder retains fewer experts than top_k")
    if args.recipe == "quartile-prune":
        if args.mask is not None:
            raise ValueError("quartile-prune requires the full expert bank and does not accept a mask")
        if args.prune_unused:
            raise ValueError("quartile-prune uses a fixed coldest quartile prune, not --prune-unused")
        if args.expert_count != args.original_expert_count:
            raise ValueError(
                "quartile-prune requires expert_count to equal original_expert_count "
                f"({args.expert_count} vs {args.original_expert_count})"
            )
        if args.expert_count % 4 != 0:
            raise ValueError("quartile-prune requires an expert count divisible by 4")
    masked_pruned = (
        read_mask(args.mask, layers, args.expert_count)
        if args.mask is not None
        else {layer: set() for layer in layers}
    )
    if args.recipe == "uniform-nvfp4":
        if args.trace:
            raise ValueError("uniform-nvfp4 must not depend on calibration traces")
        if args.prune_unused:
            raise ValueError("uniform-nvfp4 cannot prune experts")
        counts = {layer: [0] * args.expert_count for layer in layers}
        events = 0
        tokens_by_layer = {layer: 0 for layer in layers}
    else:
        counts, events, tokens_by_layer = read_trace(
            args.trace,
            layers,
            args.expert_count,
            args.top_k,
            args.expected_tokens,
        )
        if events == 0:
            raise ValueError("no trace records matched the selected layers")
    assignments: list[dict[str, Any]] = []
    pruned: dict[str, list[int]] = {}
    summaries: dict[str, Any] = {}

    for layer in layers:
        c = counts[layer]
        inactive = set(masked_pruned[layer])
        if args.prune_unused:
            inactive.update(ex for ex, count in enumerate(c) if count == 0)
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
        elif args.recipe == "reap50-plus25":
            if args.mask is None:
                raise ValueError("reap50-plus25 requires a recovered REAP mask")
            if len(retained) * 2 != args.original_expert_count:
                raise ValueError(
                    "reap50-plus25 expects the source to retain exactly 50% of original experts "
                    f"({len(retained)} vs original {args.original_expert_count})"
                )
            if args.prune_unused:
                raise ValueError("reap50-plus25 does not apply an additional unused-expert prune")
            q2_n = round(args.original_expert_count * 0.25)
            low = _take_ranked(retained, c, q2_n, hottest=False)
            tiers = {"Q2_K": low, "NVFP4": set(retained) - low}
        elif args.recipe == "quartile-prune":
            quartile = len(retained) // 4
            if 3 * quartile < args.top_k:
                raise ValueError(
                    f"layer {layer}: quartile pruning leaves {3 * quartile} experts, "
                    f"below top_k={args.top_k}"
                )
            ranked = sorted(retained, key=lambda ex: (-c[ex], ex))
            tiers = {
                "NVFP4": set(ranked[:quartile]),
                "Q3_K": set(ranked[quartile : 2 * quartile]),
                "Q2_K": set(ranked[2 * quartile : 3 * quartile]),
            }
            inactive.update(ranked[3 * quartile :])
        elif args.recipe == "traffic-ladder":
            ranked = sorted(retained, key=lambda ex: (-c[ex], ex))
            q8_end = args.q8_count
            nvfp4_end = q8_end + args.nvfp4_count
            q2_end = nvfp4_end + args.q2_count
            tiers = {
                "Q8_0": set(ranked[:q8_end]),
                "NVFP4": set(ranked[q8_end:nvfp4_end]),
                "Q2_K": set(ranked[nvfp4_end:q2_end]),
            }
            inactive.update(ranked[q2_end:])
        else:
            raise ValueError(f"unknown recipe {args.recipe!r}")

        for qtype in ("Q8_0", "NVFP4", "Q3_K", "Q2_K"):
            ids = sorted(tiers.get(qtype, set()))
            if ids:
                assignments.append({"layer": layer, "experts": ids, "qtype": qtype})
        if inactive:
            pruned[str(layer)] = sorted(inactive)
        summaries[str(layer)] = {
            "assignments": sum(c),
            "observed_experts": sum(x > 0 for x in c),
            "pruned": len(inactive),
            "q8_0": len(tiers.get("Q8_0", set())),
            "nvfp4": len(tiers.get("NVFP4", set())),
            "q3_k": len(tiers.get("Q3_K", set())),
            "q2_k": len(tiers.get("Q2_K", set())),
        }

    descriptions = {
        "uniform-nvfp4": "All retained experts NVFP4; no calibration or pruning",
        "usage-pyramid": "Top 25% NVFP4, middle 50% Q3_K, bottom 25% Q2_K, zero-count experts pruned",
        "reap50-plus25": "REAP50 source: least-used 25% of original bank Q2_K, remaining 25% NVFP4",
        "quartile-prune": "Usage-ranked full bank: top 25% NVFP4, next 25% Q3_K, next 25% Q2_K, coldest 25% pruned",
    }
    if args.recipe == "traffic-ladder":
        descriptions["traffic-ladder"] = (
            f"Usage-ranked full bank: top {args.q8_count} Q8_0, next "
            f"{args.nvfp4_count} NVFP4, next {args.q2_count} Q2_K, coldest "
            f"{args.expert_count - args.q8_count - args.nvfp4_count - args.q2_count} pruned"
        )

    return {
        "format": FORMAT,
        "recipe": args.recipe,
        "description": descriptions[args.recipe],
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
            "fixed_tier_fractions": (
                {"NVFP4": 0.25, "Q3_K": 0.25, "Q2_K": 0.25}
                if args.recipe == "quartile-prune"
                else None
            ),
            "fixed_tier_counts": (
                {"Q8_0": args.q8_count, "NVFP4": args.nvfp4_count, "Q2_K": args.q2_count}
                if args.recipe == "traffic-ladder"
                else None
            ),
            "fixed_prune_fraction": 0.25 if args.recipe == "quartile-prune" else None,
            "fixed_prune_count": (
                args.expert_count - args.q8_count - args.nvfp4_count - args.q2_count
                if args.recipe == "traffic-ladder"
                else None
            ),
            "prune_unused": args.prune_unused,
            "expert_mask": str(args.mask.resolve()) if args.mask is not None else None,
            "tie_break": "ascending expert id",
        },
        "calibration": {
            "trace_files": [
                {"path": str(path.resolve()), "sha256": sha256(path)} for path in args.trace
            ],
            "mask_file": (
                {"path": str(args.mask.resolve()), "sha256": sha256(args.mask)}
                if args.mask is not None
                else None
            ),
            "matched_events": events,
            "expected_prompt_tokens_per_layer": args.expected_tokens,
            "matched_prompt_tokens_by_layer": {
                str(layer): tokens_by_layer[layer] for layer in layers
            },
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
        trace.write_text("1 2 0,0,1,2\n1 2 0,1,2,3\n")
        args = argparse.Namespace(
            trace=[trace], recipe="usage-pyramid", expert_count=5, original_expert_count=5,
            top_k=2, layers="1", hot_fraction=0.25, low_fraction=0.25, prune_unused=True,
            mask=None, expected_tokens=4,
        )
        plan = build_plan(args)
        summary = plan["layer_summary"]["1"]
        assert summary == {
            "assignments": 8, "observed_experts": 4, "pruned": 1,
            "q8_0": 0, "nvfp4": 1, "q3_k": 2, "q2_k": 1,
        }
        assert plan["pruned_experts"] == {"1": [4]}
        try:
            read_trace([trace], [1], 5, top_k=2, expected_tokens=5)
        except ValueError as exc:
            assert "token coverage" in str(exc)
        else:
            raise AssertionError("truncated trace coverage was accepted")
        malformed = root / "malformed.trace"
        malformed.write_text("1 2 0,1,2\n")
        try:
            read_trace([malformed], [1], 5, top_k=2, expected_tokens=None)
        except ValueError as exc:
            assert "assignments" in str(exc)
        else:
            raise AssertionError("malformed trace assignment count was accepted")
        uniform = build_plan(argparse.Namespace(
            trace=[], recipe="uniform-nvfp4", expert_count=4, original_expert_count=4,
            top_k=2, layers="1", hot_fraction=0.25, low_fraction=0.25,
            prune_unused=False, mask=None, expected_tokens=None,
        ))
        uniform_summary = uniform["layer_summary"]["1"]
        assert uniform_summary["nvfp4"] == 4
        assert uniform_summary["q3_k"] == 0 and uniform_summary["q2_k"] == 0
        assert uniform["calibration"]["trace_files"] == []
        reap_trace = root / "reap.trace"
        reap_trace.write_text("1 12 " + ",".join(str(x) for x in range(0, 192, 2)) + "\n")
        reap_mask = root / "reap-mask.json"
        reap_mask.write_text(json.dumps({
            "format": "bw24-hy3-reap-mask-v1",
            "layers": {"1": {
                "retained_experts": list(range(0, 192, 2)),
                "pruned_experts": list(range(1, 192, 2)),
            }},
        }))
        reap = build_plan(argparse.Namespace(
            trace=[reap_trace], recipe="reap50-plus25", expert_count=192,
            original_expert_count=192, top_k=8, layers="1",
            hot_fraction=0.25, low_fraction=0.25, prune_unused=False, mask=reap_mask,
            expected_tokens=12,
        ))
        reap_summary = reap["layer_summary"]["1"]
        assert reap_summary["q2_k"] == 48 and reap_summary["nvfp4"] == 48
        assert reap_summary["q3_k"] == 0 and reap_summary["pruned"] == 96
        reap_uniform = build_plan(argparse.Namespace(
            trace=[], recipe="uniform-nvfp4", expert_count=192,
            original_expert_count=192, top_k=8, layers="1",
            hot_fraction=0.25, low_fraction=0.25, prune_unused=False,
            mask=reap_mask, expected_tokens=None,
        ))
        assert reap_uniform["layer_summary"]["1"]["nvfp4"] == 96
        assert reap_uniform["pruned_experts"]["1"] == list(range(1, 192, 2))
        quartile_trace = root / "quartile.trace"
        quartile_trace.write_text(json.dumps({
            "layer": 1,
            "tokens": 18,
            "experts": [
                {"expert": expert, "count": count}
                for expert, count in enumerate(range(8, 0, -1))
            ],
        }) + "\n")
        quartile = build_plan(argparse.Namespace(
            trace=[quartile_trace], recipe="quartile-prune", expert_count=8,
            original_expert_count=8, top_k=2, layers="1",
            hot_fraction=0.25, low_fraction=0.25, prune_unused=False,
            mask=None, expected_tokens=18,
        ))
        quartile_summary = quartile["layer_summary"]["1"]
        assert quartile_summary == {
            "assignments": 36, "observed_experts": 8, "pruned": 2,
            "q8_0": 0, "nvfp4": 2, "q3_k": 2, "q2_k": 2,
        }
        assert quartile["assignments"] == [
            {"layer": 1, "experts": [0, 1], "qtype": "NVFP4"},
            {"layer": 1, "experts": [2, 3], "qtype": "Q3_K"},
            {"layer": 1, "experts": [4, 5], "qtype": "Q2_K"},
        ]
        assert quartile["pruned_experts"] == {"1": [6, 7]}
        traffic = build_plan(argparse.Namespace(
            trace=[quartile_trace], recipe="traffic-ladder", expert_count=8,
            original_expert_count=8, top_k=2, layers="1",
            hot_fraction=0.25, low_fraction=0.25, prune_unused=False,
            mask=None, expected_tokens=18, q8_count=1, nvfp4_count=2, q2_count=3,
        ))
        assert traffic["assignments"] == [
            {"layer": 1, "experts": [0], "qtype": "Q8_0"},
            {"layer": 1, "experts": [1, 2], "qtype": "NVFP4"},
            {"layer": 1, "experts": [3, 4, 5], "qtype": "Q2_K"},
        ]
        assert traffic["pruned_experts"] == {"1": [6, 7]}
        traffic_no_prune = build_plan(argparse.Namespace(
            trace=[quartile_trace], recipe="traffic-ladder", expert_count=8,
            original_expert_count=8, top_k=2, layers="1",
            hot_fraction=0.25, low_fraction=0.25, prune_unused=False,
            mask=None, expected_tokens=18, q8_count=1, nvfp4_count=2, q2_count=5,
        ))
        assert traffic_no_prune["assignments"] == [
            {"layer": 1, "experts": [0], "qtype": "Q8_0"},
            {"layer": 1, "experts": [1, 2], "qtype": "NVFP4"},
            {"layer": 1, "experts": [3, 4, 5, 6, 7], "qtype": "Q2_K"},
        ]
        assert traffic_no_prune["pruned_experts"] == {}
        assert traffic_no_prune["layer_summary"]["1"]["pruned"] == 0
        assert traffic_no_prune["policy"]["fixed_prune_count"] == 0
        traffic_q8_q2 = build_plan(argparse.Namespace(
            trace=[quartile_trace], recipe="traffic-ladder", expert_count=8,
            original_expert_count=8, top_k=2, layers="1",
            hot_fraction=0.25, low_fraction=0.25, prune_unused=False,
            mask=None, expected_tokens=18, q8_count=1, nvfp4_count=0, q2_count=7,
        ))
        assert traffic_q8_q2["assignments"] == [
            {"layer": 1, "experts": [0], "qtype": "Q8_0"},
            {"layer": 1, "experts": [1, 2, 3, 4, 5, 6, 7], "qtype": "Q2_K"},
        ]
        assert traffic_q8_q2["pruned_experts"] == {}
        assert traffic_q8_q2["layer_summary"]["1"] == {
            "assignments": 36, "observed_experts": 8, "pruned": 0,
            "q8_0": 1, "nvfp4": 0, "q3_k": 0, "q2_k": 7,
        }
        traffic_nvfp4_q2 = build_plan(argparse.Namespace(
            trace=[quartile_trace], recipe="traffic-ladder", expert_count=8,
            original_expert_count=8, top_k=2, layers="1",
            hot_fraction=0.25, low_fraction=0.25, prune_unused=False,
            mask=None, expected_tokens=18, q8_count=0, nvfp4_count=3, q2_count=5,
        ))
        assert traffic_nvfp4_q2["assignments"] == [
            {"layer": 1, "experts": [0, 1, 2], "qtype": "NVFP4"},
            {"layer": 1, "experts": [3, 4, 5, 6, 7], "qtype": "Q2_K"},
        ]
        assert traffic_nvfp4_q2["pruned_experts"] == {}
        assert traffic_nvfp4_q2["layer_summary"]["1"] == {
            "assignments": 36, "observed_experts": 8, "pruned": 0,
            "q8_0": 0, "nvfp4": 3, "q3_k": 0, "q2_k": 5,
        }
        print("expert tier plan self-test: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, action="append", default=[])
    parser.add_argument("--mask", type=Path, help="recovered original-id expert mask")
    parser.add_argument("--recipe", choices=RECIPES)
    parser.add_argument("--expert-count", type=int)
    parser.add_argument("--original-expert-count", type=int)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--expected-tokens",
        type=int,
        help="require every selected MoE layer to cover exactly this many prompt tokens",
    )
    parser.add_argument("--layers", help="inclusive range (1-79) or comma-separated ids")
    parser.add_argument("--hot-fraction", type=float, default=0.25)
    parser.add_argument("--low-fraction", type=float, default=0.25)
    parser.add_argument("--q8-count", type=int)
    parser.add_argument("--nvfp4-count", type=int)
    parser.add_argument("--q2-count", type=int)
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
