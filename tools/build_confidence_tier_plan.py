#!/usr/bin/env python3
"""Build a globally ranked expert plan from teacher-forced confidence and router weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator


FORMAT = "bw24-expert-tier-plan-v2"
CONFIDENCE_FORMAT = "bw24-token-confidence-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_layers(raw: str) -> list[int]:
    if "-" in raw:
        lo, hi = (int(value) for value in raw.split("-", 1))
        if lo > hi:
            raise ValueError("layer range is descending")
        return list(range(lo, hi + 1))
    return [int(value) for value in raw.split(",") if value]


def load_requests(path: Path) -> list[dict[str, Any]]:
    requests = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not requests:
        raise ValueError("request corpus is empty")
    seen: set[str] = set()
    for record in requests:
        trace_id = str(record["ordinal"])
        if trace_id in seen:
            raise ValueError(f"duplicate request ordinal {trace_id}")
        seen.add(trace_id)
        if int(record["prompt_tokens"]) != len(record["prompt_ids"]):
            raise ValueError(f"request {trace_id}: prompt token count does not match prompt_ids")
        if len(record["prompt_ids"]) < 2:
            raise ValueError(f"request {trace_id}: at least two prompt tokens are required")
    return requests


def load_confidence(
    path: Path,
    requests: list[dict[str, Any]],
    low_fraction: float,
    high_fraction: float,
) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, dict[str, int]]]:
    expected = {
        (str(record["ordinal"]), position)
        for record in requests
        for position in range(len(record["prompt_ids"]) - 1)
    }
    request_stratum = {str(record["ordinal"]): str(record["stratum"]) for record in requests}
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open() as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("format") != CONFIDENCE_FORMAT:
                raise ValueError(f"{path}:{line_no}: unsupported confidence format")
            trace_id = str(row.get("trace_id"))
            key = (trace_id, int(row["input_position"]))
            if key not in expected:
                raise ValueError(f"{path}:{line_no}: unexpected token key {key}")
            if key in rows:
                raise ValueError(f"{path}:{line_no}: duplicate token key {key}")
            row["trace_id"] = trace_id
            row["stratum"] = request_stratum[trace_id]
            rows[key] = row
            by_stratum[row["stratum"]].append(row)
    missing = expected - rows.keys()
    if missing:
        raise ValueError(f"confidence trace is missing {len(missing)} token records")

    band_counts: dict[str, dict[str, int]] = {}
    for stratum, items in by_stratum.items():
        ranked = sorted(
            items,
            key=lambda row: (
                float(row["reference_logprob"]),
                row["trace_id"],
                int(row["input_position"]),
            ),
        )
        low_n = max(1, math.ceil(len(ranked) * low_fraction))
        high_n = max(1, math.ceil(len(ranked) * high_fraction))
        if low_n + high_n > len(ranked):
            raise ValueError(f"confidence bands overlap for stratum {stratum}")
        for row in ranked[:low_n]:
            row["band"] = "low"
        for row in ranked[low_n : len(ranked) - high_n]:
            row["band"] = "middle"
        for row in ranked[len(ranked) - high_n :]:
            row["band"] = "high"
        band_counts[stratum] = {
            "tokens": len(ranked),
            "low": low_n,
            "middle": len(ranked) - low_n - high_n,
            "high": high_n,
        }
    return rows, band_counts


def trace_lines(path: Path) -> Iterator[tuple[int, str]]:
    with path.open() as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                yield line_no, line.strip()


def parse_weight_row(
    path: Path,
    line_no: int,
    line: str,
    expected_layer: int,
    expert_count: int,
    top_k: int,
) -> list[tuple[int, float]]:
    fields = line.split(maxsplit=2)
    if len(fields) != 3:
        raise ValueError(f"{path}:{line_no}: expected 'LAYER 1 ID:WEIGHT,...'")
    layer, tokens = int(fields[0]), int(fields[1])
    if layer != expected_layer or tokens != 1:
        raise ValueError(
            f"{path}:{line_no}: expected layer={expected_layer} tokens=1, got {layer}/{tokens}"
        )
    pairs: list[tuple[int, float]] = []
    for raw in fields[2].split(","):
        expert_s, weight_s = raw.split(":", 1)
        expert, weight = int(expert_s), float(weight_s)
        if expert < 0 or expert >= expert_count:
            raise ValueError(f"{path}:{line_no}: expert {expert} is out of range")
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"{path}:{line_no}: invalid router weight {weight}")
        pairs.append((expert, weight))
    if len(pairs) != top_k or len({expert for expert, _ in pairs}) != top_k:
        raise ValueError(f"{path}:{line_no}: expected {top_k} distinct routed experts")
    return pairs


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    layers = parse_layers(args.layers)
    if not layers:
        raise ValueError("at least one MoE layer is required")
    if not (0 < args.low_fraction < 1 and 0 < args.high_fraction < 1):
        raise ValueError("confidence fractions must be between zero and one")
    if args.low_fraction + args.high_fraction > 1:
        raise ValueError("confidence bands overlap")
    if not (0 <= args.rescue_weight <= 1):
        raise ValueError("rescue weight must be between zero and one")
    tier_counts = (args.q8_count, args.nvfp4_count, args.q2_count)
    if args.q8_count <= 0 or args.nvfp4_count < 0 or args.q2_count <= 0:
        raise ValueError("Q8/Q2 counts must be positive and NVFP4 count non-negative")
    if sum(tier_counts) != args.expert_count:
        raise ValueError("the first confidence experiment is no-prune and must assign every expert")

    requests = load_requests(args.requests)
    confidence, band_counts = load_confidence(
        args.confidence_trace, requests, args.low_fraction, args.high_fraction
    )
    low_mass = {(layer, expert): 0.0 for layer in layers for expert in range(args.expert_count)}
    high_mass = {(layer, expert): 0.0 for layer in layers for expert in range(args.expert_count)}
    route_counts = {(layer, expert): 0 for layer in layers for expert in range(args.expert_count)}
    lines = trace_lines(args.weight_trace)
    routed_tokens = 0
    for request in requests:
        trace_id = str(request["ordinal"])
        for position in range(len(request["prompt_ids"])):
            token = confidence.get((trace_id, position))
            for layer in layers:
                try:
                    line_no, line = next(lines)
                except StopIteration as exc:
                    raise ValueError("weighted route trace ended early") from exc
                pairs = parse_weight_row(
                    args.weight_trace, line_no, line, layer, args.expert_count, args.top_k
                )
                for expert, weight in pairs:
                    route_counts[(layer, expert)] += 1
                    if token is None or not bool(token["top1_correct"]):
                        continue
                    if token["band"] == "low":
                        uncertainty = min(
                            args.uncertainty_cap,
                            1.0 / (float(token["top1_top2_margin"]) + args.margin_epsilon),
                        )
                        low_mass[(layer, expert)] += uncertainty * weight
                    elif token["band"] == "high":
                        high_mass[(layer, expert)] += weight
            routed_tokens += 1
    try:
        extra_line_no, _ = next(lines)
    except StopIteration:
        pass
    else:
        raise ValueError(f"weighted route trace has extra data starting at line {extra_line_no}")

    max_low = max(low_mass.values())
    if max_low <= 0:
        raise ValueError("no correct low-confidence rescue mass was observed")
    specialization = {
        key: low_mass[key] / (low_mass[key] + high_mass[key] + 1e-12) for key in low_mass
    }
    scores = {
        key: args.rescue_weight * (low_mass[key] / max_low)
        + (1.0 - args.rescue_weight) * specialization[key]
        for key in low_mass
    }
    ranked = sorted(scores, key=lambda key: (-scores[key], key[0], key[1]))
    n_layers = len(layers)
    q8_end = args.q8_count * n_layers
    nvfp4_end = q8_end + args.nvfp4_count * n_layers
    tier_for: dict[tuple[int, int], str] = {}
    for key in ranked[:q8_end]:
        tier_for[key] = "Q8_0"
    for key in ranked[q8_end:nvfp4_end]:
        tier_for[key] = "NVFP4"
    for key in ranked[nvfp4_end:]:
        tier_for[key] = "Q2_K"

    assignments: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for layer in layers:
        layer_tiers: dict[str, list[int]] = {}
        for qtype in ("Q8_0", "NVFP4", "Q2_K"):
            experts = [expert for expert in range(args.expert_count) if tier_for[(layer, expert)] == qtype]
            layer_tiers[qtype] = experts
            if experts:
                assignments.append({"layer": layer, "experts": experts, "qtype": qtype})
        summaries[str(layer)] = {
            "assignments": sum(route_counts[(layer, expert)] for expert in range(args.expert_count)),
            "observed_experts": sum(route_counts[(layer, expert)] > 0 for expert in range(args.expert_count)),
            "pruned": 0,
            "q8_0": len(layer_tiers["Q8_0"]),
            "nvfp4": len(layer_tiers["NVFP4"]),
            "q3_k": 0,
            "q2_k": len(layer_tiers["Q2_K"]),
        }

    expert_scores = [
        {
            "layer": layer,
            "expert": expert,
            "score": scores[(layer, expert)],
            "low_rescue_mass": low_mass[(layer, expert)],
            "high_easy_mass": high_mass[(layer, expert)],
            "specialization": specialization[(layer, expert)],
        }
        for layer, expert in ranked
    ]
    return {
        "format": FORMAT,
        "recipe": "confidence-ladder",
        "description": "Global hard-token rescue ranking with exact no-prune format totals",
        "model": {
            "expert_count": args.expert_count,
            "original_expert_count": args.expert_count,
            "expert_used_count": args.top_k,
            "moe_layers": layers,
        },
        "policy": {
            "rank_metric": "confidence_rescue_plus_hard_easy_specialization",
            "global_allocation": True,
            "rescue_weight": args.rescue_weight,
            "low_fraction_per_stratum": args.low_fraction,
            "high_fraction_per_stratum": args.high_fraction,
            "uncertainty_cap": args.uncertainty_cap,
            "margin_epsilon": args.margin_epsilon,
            "matched_total_counts": {
                "Q8_0": args.q8_count * n_layers,
                "NVFP4": args.nvfp4_count * n_layers,
                "Q2_K": args.q2_count * n_layers,
            },
            "tie_break": "ascending layer then expert id",
        },
        "calibration": {
            "requests": {"path": str(args.requests.resolve()), "sha256": sha256(args.requests)},
            "confidence_trace": {
                "path": str(args.confidence_trace.resolve()),
                "sha256": sha256(args.confidence_trace),
            },
            "weighted_route_trace": {
                "path": str(args.weight_trace.resolve()),
                "sha256": sha256(args.weight_trace),
            },
            "requests_count": len(requests),
            "routed_tokens": routed_tokens,
            "confidence_band_counts": band_counts,
            "public_eval_data_used_for_selection": False,
        },
        "score_diagnostics": {
            "experts": expert_scores,
            "top_experts": expert_scores[: min(64, len(expert_scores))],
        },
        "pruned_experts": {},
        "assignments": assignments,
        "layer_summary": summaries,
    }


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="bw24-confidence-plan-") as tmp:
        root = Path(tmp)
        requests = root / "requests.jsonl"
        confidence = root / "confidence.jsonl"
        weights = root / "weights.trace"
        requests.write_text("\n".join(json.dumps(row) for row in [
            {"ordinal": 0, "stratum": "code", "prompt_tokens": 3, "prompt_ids": [10, 11, 12]},
            {"ordinal": 1, "stratum": "code", "prompt_tokens": 3, "prompt_ids": [20, 21, 22]},
        ]) + "\n")
        confidence.write_text("\n".join(json.dumps({
            "format": CONFIDENCE_FORMAT,
            "trace_id": str(trace_id),
            "input_position": position,
            "reference_logprob": logprob,
            "top1_correct": True,
            "top1_top2_margin": margin,
        }) for trace_id, position, logprob, margin in [
            (0, 0, -4.0, 0.1), (0, 1, -2.0, 0.5),
            (1, 0, -1.0, 1.0), (1, 1, -0.1, 4.0),
        ]) + "\n")
        route_experts = [2, 1, 0, 0, 1, 1]
        weight_rows = []
        for expert in route_experts:
            for layer in (1, 2):
                weight_rows.append(f"{layer} 1 {expert}:1.000000000")
        weights.write_text("\n".join(weight_rows) + "\n")
        args = argparse.Namespace(
            requests=requests, confidence_trace=confidence, weight_trace=weights,
            layers="1-2", expert_count=3, top_k=1, q8_count=1, nvfp4_count=0,
            q2_count=2, low_fraction=0.25, high_fraction=0.25,
            rescue_weight=0.5, uncertainty_cap=10.0, margin_epsilon=0.1,
        )
        plan = build_plan(args)
        plan_path = root / "plan.json"
        plan_path.write_text(json.dumps(plan))
        from prepare_mixed_expert_repack import load_assignments

        _, expanded, pruned = load_assignments(plan_path)
        assert len(expanded) == 2 * 3 * 3
        assert all(not experts for experts in pruned.values())
        q8 = {
            (group["layer"], expert)
            for group in plan["assignments"] if group["qtype"] == "Q8_0"
            for expert in group["experts"]
        }
        assert q8 == {(1, 2), (2, 2)}
        assert plan["policy"]["matched_total_counts"] == {"Q8_0": 2, "NVFP4": 0, "Q2_K": 4}
        assert plan["pruned_experts"] == {}
        assert len(plan["score_diagnostics"]["experts"]) == 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--confidence-trace", type=Path, required=True)
    parser.add_argument("--weight-trace", type=Path, required=True)
    parser.add_argument("--layers", default="1-79")
    parser.add_argument("--expert-count", type=int, default=192)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--q8-count", type=int, required=True)
    parser.add_argument("--nvfp4-count", type=int, required=True)
    parser.add_argument("--q2-count", type=int, required=True)
    parser.add_argument("--low-fraction", type=float, default=0.25)
    parser.add_argument("--high-fraction", type=float, default=0.25)
    parser.add_argument("--rescue-weight", type=float, default=0.5)
    parser.add_argument("--uncertainty-cap", type=float, default=10.0)
    parser.add_argument("--margin-epsilon", type=float, default=0.1)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        print("confidence tier plan self-test: PASS")
        return
    args = parse_args()
    if args.self_test:
        self_test()
        print("confidence tier plan self-test: PASS")
        return
    plan = build_plan(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out} ({sha256(args.out)})")


if __name__ == "__main__":
    main()
