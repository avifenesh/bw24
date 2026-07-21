#!/usr/bin/env python3
"""Measure adjacent-layer Hy3 route prediction on held-out decode passes."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_decode_passes(path: Path, n_used: int) -> list[dict[int, list[int]]]:
    passes: list[dict[int, list[int]]] = []
    current: dict[int, list[int]] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        layer_raw, tokens_raw, routes_raw = line.split()
        layer = int(layer_raw)
        tokens = int(tokens_raw)
        routes = [int(value) for value in routes_raw.split(",")]
        if len(routes) != tokens * n_used:
            raise ValueError(
                f"line {line_number}: {len(routes)} routes != {tokens} * {n_used}"
            )
        if tokens != 1:
            continue
        if current and (layer in current or layer < max(current)):
            passes.append(current)
            current = {}
        current[layer] = routes
    if current:
        passes.append(current)
    if not passes:
        raise ValueError("trace has no one-token decode passes")
    layer_set = set(passes[0])
    if any(set(decode_pass) != layer_set for decode_pass in passes):
        raise ValueError("decode passes do not share one complete layer set")
    return passes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--train", type=int, default=25)
    parser.add_argument("--test", type=int, default=25)
    parser.add_argument("--n-used", type=int, default=8)
    parser.add_argument("--expert-count", type=int, default=192)
    args = parser.parse_args()

    passes = parse_decode_passes(args.trace, args.n_used)
    if len(passes) < args.train + args.test:
        raise ValueError(
            f"need {args.train + args.test} decode passes, found {len(passes)}"
        )
    train = passes[: args.train]
    test = passes[args.train : args.train + args.test]
    layers = sorted(train[0])
    transitions: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)
    for decode_pass in train:
        for source_layer, target_layer in zip(layers, layers[1:]):
            for source in decode_pass[source_layer]:
                transitions[source_layer, source].update(decode_pass[target_layer])

    hits = {1: 0, 2: 0, 4: 0}
    cases = 0
    previous = train[-1]
    for decode_pass in test:
        for source_layer, target_layer in zip(layers, layers[1:]):
            score: Counter[int] = Counter()
            for source in decode_pass[source_layer]:
                score.update(transitions[source_layer, source])
            temporal = previous[target_layer]
            temporal_rank = {expert: rank for rank, expert in enumerate(temporal)}
            ranked = sorted(
                temporal_rank,
                key=lambda expert: (-score[expert], temporal_rank[expert], expert),
            )
            actual = set(decode_pass[target_layer])
            for width in hits:
                hits[width] += sum(expert in actual for expert in ranked[:width])
            cases += 1
        previous = decode_pass

    result = {
        "format": "bw24-hy3-route-transition-analysis-v1",
        "trace": str(args.trace),
        "decode_passes": len(passes),
        "train_passes": args.train,
        "test_passes": args.test,
        "layers_per_pass": len(layers),
        "adjacent_layer_cases": cases,
        "n_used": args.n_used,
        "expert_count": args.expert_count,
        "predictor": "previous-target-set ranked by summed adjacent transition counts",
        "widths": {
            str(width): {
                "hits": hits[width],
                "precision": hits[width] / (cases * width),
                "recall": hits[width] / (cases * args.n_used),
            }
            for width in hits
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
