#!/usr/bin/env python3
"""Prove that a runtime route trace never selects a physically pruned expert."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import sys
import tempfile


OUTPUT_FORMAT = "bw24-pruned-route-trace-gate-v1"


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(
    manifest_path: pathlib.Path,
    trace_path: pathlib.Path,
    expected_tokens: int,
    layers: list[int],
    top_k: int,
) -> dict:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != "bw24-expert-overlay-v2":
        raise ValueError("unsupported artifact manifest format")
    plan = manifest.get("plan", {})
    if plan.get("format") != "bw24-expert-tier-plan-v2":
        raise ValueError("artifact has no embedded expert tier plan")
    expert_count = int(plan.get("model", {}).get("expert_count", 0))
    if expert_count <= 0:
        raise ValueError("artifact plan has no positive expert count")
    pruned = {
        int(layer): {int(expert) for expert in experts}
        for layer, experts in manifest.get("pruned_experts", {}).items()
    }
    plan_pruned = {
        int(layer): {int(expert) for expert in experts}
        for layer, experts in plan.get("pruned_experts", {}).items()
    }
    if pruned != plan_pruned:
        raise ValueError("manifest and embedded plan prune masks differ")

    expected_route_rows = expected_tokens * len(layers)
    trace_rows = 0
    route_rows = 0
    selected = 0
    tokens_by_layer = {layer: 0 for layer in layers}
    cycle_tokens: int | None = None
    selected_pruned: list[dict[str, int]] = []
    with trace_path.open() as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            expected_layer = layers[trace_rows % len(layers)]
            fields = line.split(maxsplit=2)
            if len(fields) != 3:
                raise ValueError(f"{trace_path}:{line_no}: malformed route row")
            layer, row_tokens = int(fields[0]), int(fields[1])
            if layer != expected_layer or row_tokens <= 0:
                raise ValueError(
                    f"{trace_path}:{line_no}: layer/tokens={layer}/{row_tokens}, "
                    f"expected layer {expected_layer} with a positive token count"
                )
            layer_index = trace_rows % len(layers)
            if layer_index == 0:
                cycle_tokens = row_tokens
            elif row_tokens != cycle_tokens:
                raise ValueError(
                    f"{trace_path}:{line_no}: token count {row_tokens} differs from "
                    f"{cycle_tokens} within one request cycle"
                )
            pairs = []
            for raw in fields[2].split(","):
                expert_s, weight_s = raw.split(":", 1)
                expert, weight = int(expert_s), float(weight_s)
                if not math.isfinite(weight) or weight < 0:
                    raise ValueError(f"{trace_path}:{line_no}: invalid route weight")
                pairs.append((expert, weight))
            if len(pairs) != row_tokens * top_k:
                raise ValueError(
                    f"{trace_path}:{line_no}: route pairs={len(pairs)}, "
                    f"expected {row_tokens * top_k}"
                )
            for token_index in range(row_tokens):
                token_pairs = pairs[token_index * top_k:(token_index + 1) * top_k]
                if len({expert for expert, _ in token_pairs}) != top_k:
                    raise ValueError(
                        f"{trace_path}:{line_no}: token {token_index} does not have "
                        f"{top_k} distinct experts"
                    )
                for expert, _ in token_pairs:
                    if expert < 0 or expert >= expert_count:
                        raise ValueError(
                            f"{trace_path}:{line_no}: expert {expert} outside "
                            f"0..{expert_count - 1}"
                        )
                    if expert in pruned.get(layer, set()):
                        selected_pruned.append({
                            "line": line_no,
                            "token": token_index,
                            "layer": layer,
                            "expert": expert,
                        })
                    selected += 1
            tokens_by_layer[layer] += row_tokens
            route_rows += row_tokens
            trace_rows += 1
    if trace_rows % len(layers):
        raise ValueError(f"trace rows={trace_rows} do not end on a complete layer cycle")
    if any(tokens != expected_tokens for tokens in tokens_by_layer.values()):
        raise ValueError(
            f"per-layer token coverage differs from expected {expected_tokens}: "
            f"{tokens_by_layer}"
        )
    if route_rows != expected_route_rows:
        raise ValueError(f"route rows={route_rows}, expected {expected_route_rows}")
    return {
        "format": OUTPUT_FORMAT,
        "passed": not selected_pruned,
        "expected_tokens": expected_tokens,
        "layers": layers,
        "top_k": top_k,
        "trace_rows": trace_rows,
        "route_rows": route_rows,
        "selected_experts": selected,
        "selected_pruned_experts": selected_pruned,
        "evidence": {
            "manifest": {"path": str(manifest_path.resolve()), "sha256": sha256(manifest_path)},
            "route_trace": {"path": str(trace_path.resolve()), "sha256": sha256(trace_path)},
        },
    }


def parse_layers(raw: str) -> list[int]:
    if "-" in raw:
        start, end = (int(value) for value in raw.split("-", 1))
        if start > end:
            raise ValueError("descending layer range")
        return list(range(start, end + 1))
    layers = [int(value) for value in raw.split(",") if value]
    if not layers:
        raise ValueError("no layers selected")
    return layers


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="bw24-pruned-route-") as tmp:
        root = pathlib.Path(tmp)
        plan = {
            "format": "bw24-expert-tier-plan-v2",
            "model": {"expert_count": 4},
            "pruned_experts": {"1": [2], "2": [3]},
        }
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({
            "format": "bw24-expert-overlay-v2",
            "plan": plan,
            "pruned_experts": plan["pruned_experts"],
        }))
        trace = root / "routes.trace"
        trace.write_text(
            "1 2 0:0.6,1:0.4,1:0.7,3:0.3\n"
            "2 2 0:0.7,2:0.3,1:0.8,2:0.2\n"
        )
        result = validate(manifest, trace, 2, [1, 2], 2)
        assert result["passed"] and result["trace_rows"] == 2 and result["route_rows"] == 4
        trace.write_text(
            "1 2 0:0.6,2:0.4,1:0.7,3:0.3\n"
            "2 2 0:0.7,2:0.3,1:0.8,2:0.2\n"
        )
        assert not validate(manifest, trace, 2, [1, 2], 2)["passed"]
        trace.write_text(
            "1 2 0:0.6,4:0.4,1:0.7,3:0.3\n"
            "2 2 0:0.7,2:0.3,1:0.8,2:0.2\n"
        )
        try:
            validate(manifest, trace, 2, [1, 2], 2)
        except ValueError as exc:
            assert "outside 0..3" in str(exc)
        else:
            raise AssertionError("out-of-range routed expert was accepted")


def main() -> None:
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        print("pruned route trace gate self-test: PASS")
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--trace", type=pathlib.Path, required=True)
    parser.add_argument("--expected-tokens", type=int, required=True)
    parser.add_argument("--layers", default="1-79")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    result = validate(
        args.manifest, args.trace, args.expected_tokens, parse_layers(args.layers), args.top_k
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if not result["passed"]:
        raise SystemExit("runtime selected at least one pruned expert")
    print(f"wrote {args.output} passed=true rows={result['route_rows']}")


if __name__ == "__main__":
    main()
