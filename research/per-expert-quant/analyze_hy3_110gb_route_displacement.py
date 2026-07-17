#!/usr/bin/env python3
"""Compare matched private route traces for a base and restored-expert arm.

This analysis consumes only private prompts, route traces, and the frozen
private-coverage selection receipt.  It does not read public capability scores
and is suitable for diagnosing top-k competition before constructing another
arm.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
from typing import Any


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_requests(path: pathlib.Path) -> list[dict[str, Any]]:
    requests = [json.loads(line) for line in path.read_text().splitlines() if line]
    if not requests:
        raise SystemExit("private request set is empty")
    return requests


def load_restored(path: pathlib.Path) -> set[tuple[int, int]]:
    receipt = json.loads(path.read_text())
    if receipt.get("public_capability_results_used") is not False:
        raise SystemExit("selection receipt is not explicitly private-only")
    return {(int(item["layer"]), int(item["expert"])) for item in receipt["selected"]}


def parse_trace(
    path: pathlib.Path, requests: list[dict[str, Any]], top_k: int
) -> dict[tuple[int, int, int], list[tuple[int, float]]]:
    by_tokens: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for request in requests:
        by_tokens[int(request["prompt_tokens"])].append(request)
    seen_tokens: collections.Counter[int] = collections.Counter()
    rows: dict[tuple[int, int, int], list[tuple[int, float]]] = {}
    for raw in path.read_text().splitlines():
        layer_text, tokens_text, packed = raw.split(maxsplit=2)
        layer = int(layer_text)
        prompt_tokens = int(tokens_text)
        candidates = by_tokens[prompt_tokens]
        occurrence = seen_tokens[(layer, prompt_tokens)]
        if occurrence >= len(candidates):
            raise SystemExit(f"unmatched trace row for layer={layer} tokens={prompt_tokens}")
        request = candidates[occurrence]
        seen_tokens[(layer, prompt_tokens)] += 1
        values = []
        for item in packed.split(","):
            expert_text, weight_text = item.split(":", maxsplit=1)
            values.append((int(expert_text), float(weight_text)))
        expected = prompt_tokens * top_k
        if len(values) != expected:
            raise SystemExit(
                f"trace width mismatch layer={layer}: {len(values)} != {expected}"
            )
        ordinal = int(request["ordinal"])
        for token in range(prompt_tokens):
            start = token * top_k
            rows[(ordinal, layer, token)] = values[start : start + top_k]
    return rows


def summarize_counter(counter: collections.Counter[Any], limit: int = 30) -> list[dict[str, Any]]:
    return [
        {"key": list(key) if isinstance(key, tuple) else key, "count": count}
        for key, count in counter.most_common(limit)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=pathlib.Path, required=True)
    parser.add_argument("--base-trace", type=pathlib.Path, required=True)
    parser.add_argument("--restored-trace", type=pathlib.Path, required=True)
    parser.add_argument("--selection-receipt", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    requests = load_requests(args.requests)
    restored = load_restored(args.selection_receipt)
    base = parse_trace(args.base_trace, requests, args.top_k)
    candidate = parse_trace(args.restored_trace, requests, args.top_k)
    if base.keys() != candidate.keys():
        raise SystemExit("route traces do not have identical paired rows")

    request_by_ordinal = {int(item["ordinal"]): item for item in requests}
    strata: dict[str, dict[str, Any]] = collections.defaultdict(
        lambda: {
            "token_layer_positions": 0,
            "changed_positions": 0,
            "changed_slots": 0,
            "restored_entries": 0,
            "restored_entry_weight": 0.0,
            "indirect_base_entries": 0,
            "indirect_base_entry_weight": 0.0,
            "base_exits": 0,
            "base_exit_weight": 0.0,
            "restored_experts": collections.Counter(),
            "indirect_base_experts": collections.Counter(),
            "displaced_base_experts": collections.Counter(),
            "restored_displaced_pairs": collections.Counter(),
        }
    )
    global_pairs: collections.Counter[tuple[int, int, int]] = collections.Counter()
    restored_experts: collections.Counter[tuple[int, int]] = collections.Counter()
    displaced_experts: collections.Counter[tuple[int, int]] = collections.Counter()

    for (ordinal, layer, _token), base_values in base.items():
        candidate_values = candidate[(ordinal, layer, _token)]
        stratum = str(request_by_ordinal[ordinal]["stratum"])
        summary = strata[stratum]
        summary["token_layer_positions"] += 1
        base_map = dict(base_values)
        candidate_map = dict(candidate_values)
        entered = [expert for expert in candidate_map if expert not in base_map]
        exited = [expert for expert in base_map if expert not in candidate_map]
        if not entered and not exited:
            continue
        if len(entered) != len(exited):
            raise SystemExit("paired top-k change did not conserve slot count")
        summary["changed_positions"] += 1
        summary["changed_slots"] += len(entered)
        for expert in entered:
            if (layer, expert) in restored:
                summary["restored_entries"] += 1
                summary["restored_entry_weight"] += candidate_map[expert]
                summary["restored_experts"][(layer, expert)] += 1
                restored_experts[(layer, expert)] += 1
            else:
                # Earlier restored-expert dispatches alter the hidden state, so
                # later layers may change between two already-active experts.
                summary["indirect_base_entries"] += 1
                summary["indirect_base_entry_weight"] += candidate_map[expert]
                summary["indirect_base_experts"][(layer, expert)] += 1
        for expert in exited:
            summary["base_exits"] += 1
            summary["base_exit_weight"] += base_map[expert]
            summary["displaced_base_experts"][(layer, expert)] += 1
            displaced_experts[(layer, expert)] += 1
        for entering in entered:
            if (layer, entering) not in restored:
                continue
            # Pair by closest router weight; this is diagnostic attribution only.
            leaving = min(exited, key=lambda expert: abs(base_map[expert] - candidate_map[entering]))
            summary["restored_displaced_pairs"][(layer, entering, leaving)] += 1
            global_pairs[(layer, entering, leaving)] += 1

    rendered_strata: dict[str, Any] = {}
    for stratum, summary in sorted(strata.items()):
        positions = summary["token_layer_positions"]
        rendered_strata[stratum] = {
            "token_layer_positions": positions,
            "changed_positions": summary["changed_positions"],
            "changed_position_rate": summary["changed_positions"] / positions,
            "changed_slots": summary["changed_slots"],
            "restored_entries": summary["restored_entries"],
            "restored_entry_weight": summary["restored_entry_weight"],
            "indirect_base_entries": summary["indirect_base_entries"],
            "indirect_base_entry_weight": summary["indirect_base_entry_weight"],
            "base_exits": summary["base_exits"],
            "base_exit_weight": summary["base_exit_weight"],
            "top_restored_experts": summarize_counter(summary["restored_experts"]),
            "top_indirect_base_experts": summarize_counter(
                summary["indirect_base_experts"]
            ),
            "top_displaced_base_experts": summarize_counter(
                summary["displaced_base_experts"]
            ),
            "top_restored_displaced_pairs": summarize_counter(
                summary["restored_displaced_pairs"]
            ),
        }

    output = {
        "format": "bw24-hy3-private-route-displacement-v1",
        "public_capability_results_used": False,
        "top_k": args.top_k,
        "paired_token_layer_positions": len(base),
        "restored_expert_count": len(restored),
        "inputs": {
            "requests": {"path": str(args.requests), "sha256": sha256(args.requests)},
            "base_trace": {"path": str(args.base_trace), "sha256": sha256(args.base_trace)},
            "restored_trace": {
                "path": str(args.restored_trace),
                "sha256": sha256(args.restored_trace),
            },
            "selection_receipt": {
                "path": str(args.selection_receipt),
                "sha256": sha256(args.selection_receipt),
            },
        },
        "strata": rendered_strata,
        "top_restored_experts": summarize_counter(restored_experts),
        "top_displaced_base_experts": summarize_counter(displaced_experts),
        "top_restored_displaced_pairs": summarize_counter(global_pairs),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
