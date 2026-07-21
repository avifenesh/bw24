#!/usr/bin/env python3
"""Merge disjoint expert-overlay layer fragments into one validated, self-contained manifest."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from prepare_mixed_expert_repack import (
    OVERLAY_FORMAT,
    PROJECTIONS,
    _write_safetensors,
    _install_tensor_overrides,
    _layers_from_model,
    canonical_json_sha256,
    load_assignments,
    prepare,
    sha256_file,
)


TENSOR_RE = re.compile(
    r"^blk\.(?P<layer>\d+)\.ffn_(?P<projection>gate|up|down)_exps\."
    r"(?P<expert>\d+)\.weight$"
)


def merge(args: argparse.Namespace) -> dict[str, Any]:
    if not args.fragment:
        raise ValueError("at least one fragment is required")
    fragments = [(path, json.loads(path.read_text())) for path in args.fragment]
    first = fragments[0][1]
    if first.get("format") != OVERLAY_FORMAT:
        raise ValueError("unsupported fragment format")
    common_keys = (
        "format", "source_dir", "quant_source_dir", "quality", "plan", "plan_sha256",
        "plan_canonical_sha256", "pruned_experts", "source_fingerprints",
        "fallback_fingerprints", "external_quantizer", "importance_sidecars",
    )
    common = {key: first.get(key) for key in common_keys}
    if first.get("plan_canonical_sha256") != canonical_json_sha256(first["plan"]):
        raise ValueError("fragment plan canonical hash differs")
    plan_path = args.plan.resolve()
    plan, assignments, _ = load_assignments(plan_path)
    if first.get("plan_sha256") != sha256_file(plan_path) or first.get("plan") != plan:
        raise ValueError("fragment plan differs from the authoritative plan")
    expected_layers = _layers_from_model(plan["model"])
    seen_layers: set[int] = set()
    tensors: dict[str, Any] = {}
    tier_summary: dict[str, dict[str, int]] = {}
    receipts = []
    for path, fragment in fragments:
        if fragment.get("format") != OVERLAY_FORMAT:
            raise ValueError(f"{path}: unsupported fragment format")
        if any(fragment.get(key) != value for key, value in common.items()):
            raise ValueError(f"{path}: common fragment metadata differs")
        layers = [int(layer) for layer in fragment.get("fragment_layers", [])]
        if not layers or seen_layers.intersection(layers):
            raise ValueError(f"{path}: empty or overlapping layer coverage")
        seen_layers.update(layers)
        for name, spec in fragment.get("tensors", {}).items():
            match = TENSOR_RE.fullmatch(name)
            if match is None or int(match.group("layer")) not in layers or name in tensors:
                raise ValueError(f"{path}: invalid, out-of-fragment, or duplicate tensor {name}")
            payload = args.out_dir / spec["file"]
            offset, size = int(spec.get("offset", 0)), int(spec["bytes"])
            if not payload.is_file() or offset < 0 or size <= 0 or payload.stat().st_size < offset + size:
                raise ValueError(f"{path}: invalid tensor payload extent for {name}")
            tensors[name] = spec
        for qtype, row in fragment.get("tier_summary", {}).items():
            target = tier_summary.setdefault(qtype, {"experts": 0, "projections": 0, "bytes": 0})
            for key in target:
                target[key] += int(row[key])
        if int(fragment.get("artifact_bytes", -1)) != sum(
            int(spec["bytes"]) for spec in fragment.get("tensors", {}).values()
        ):
            raise ValueError(f"{path}: fragment artifact byte total differs")
        receipts.append({
            "path": str(path.resolve()), "sha256": sha256_file(path), "layers": layers,
        })
    if seen_layers != set(expected_layers):
        raise ValueError(f"fragment layers={sorted(seen_layers)}, expected={expected_layers}")

    expected_names = {
        f"blk.{layer}.ffn_{projection}_exps.{expert}.weight"
        for layer, expert, projection in assignments
    }
    if set(tensors) != expected_names:
        missing, extra = expected_names - tensors.keys(), tensors.keys() - expected_names
        raise ValueError(f"tensor coverage mismatch: missing={len(missing)} extra={len(extra)}")
    artifact_bytes = sum(int(spec["bytes"]) for spec in tensors.values())
    if artifact_bytes != sum(row["bytes"] for row in tier_summary.values()):
        raise ValueError("merged tier and tensor byte totals differ")
    manifest = dict(first)
    for key in ("fragment_layers", "tensors", "tier_summary", "artifact_bytes", "payload_bytes"):
        manifest.pop(key, None)
    manifest["created_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest["fragments"] = receipts
    manifest["tensors"] = dict(sorted(tensors.items()))
    manifest["tier_summary"] = dict(sorted(tier_summary.items()))
    manifest["artifact_bytes"] = artifact_bytes
    _install_tensor_overrides(
        args.out_dir,
        args.tensor_overrides.resolve() if args.tensor_overrides is not None else None,
        manifest,
    )
    manifest["payload_bytes"] = artifact_bytes + int(
        manifest.get("tensor_overrides", {}).get("bytes", 0)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    return manifest


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="bw24-overlay-fragments-") as tmp:
        root = Path(tmp)
        plan_path = root / "plan.json"
        plan = {
            "format": "bw24-expert-tier-plan-v2",
            "model": {"expert_count": 2, "moe_layers": [1, 2]},
            "pruned_experts": {"1": [1], "2": []},
            "assignments": [
                {"layer": 1, "experts": [0], "qtype": "Q2_K"},
                {"layer": 2, "experts": [0, 1], "qtype": "Q2_K"},
            ],
        }
        plan_path.write_text(json.dumps(plan))
        common = {
            "format": OVERLAY_FORMAT, "source_dir": "/source", "quant_source_dir": "/source",
            "quality": "test", "plan": plan, "plan_sha256": sha256_file(plan_path),
            "plan_canonical_sha256": canonical_json_sha256(plan),
            "pruned_experts": {"1": [1]}, "source_fingerprints": {},
            "fallback_fingerprints": {},
        }
        fragments = []
        for layer, experts in ((1, [0]), (2, [0, 1])):
            blob = root / f"layer-{layer}.bin"
            blob.write_bytes(bytes([layer]) * (len(experts) * 3 * 4))
            tensors = {}
            offset = 0
            for expert in experts:
                for projection in ("gate", "up", "down"):
                    tensors[f"blk.{layer}.ffn_{projection}_exps.{expert}.weight"] = {
                        "file": blob.name, "offset": offset, "qtype": "Q2_K",
                        "ne": [1, 1], "bytes": 4,
                    }
                    offset += 4
            fragment = dict(common)
            fragment.update({
                "fragment_layers": [layer], "tensors": tensors,
                "tier_summary": {"Q2_K": {
                    "experts": len(experts), "projections": len(experts) * 3, "bytes": offset,
                }},
                "artifact_bytes": offset, "payload_bytes": offset,
            })
            path = root / f"fragment-{layer}.json"
            path.write_text(json.dumps(fragment)); fragments.append(path)
        args = argparse.Namespace(
            fragment=fragments, plan=plan_path, out_dir=root,
            tensor_overrides=None, output=root / "manifest.json",
        )
        result = merge(args)
        assert len(result["tensors"]) == 9
        assert result["artifact_bytes"] == 36
        assert result["tier_summary"]["Q2_K"]["experts"] == 3

    # Integration gate: the same plan built sequentially or as disjoint layer fragments must
    # produce byte-identical expert payloads and identical final tensor/tier inventories.
    with tempfile.TemporaryDirectory(prefix="bw24-overlay-fragment-integration-") as tmp:
        root = Path(tmp)
        source, sequential, parallel = root / "source", root / "sequential", root / "parallel"
        source.mkdir()
        shard = "model.safetensors"
        tensors = {}
        weight_map = {}
        for layer in (1, 2):
            for expert in (0, 1):
                for projection in PROJECTIONS:
                    name = f"model.layers.{layer}.mlp.experts.{expert}.{projection}_proj.weight"
                    values = np.sin(
                        np.arange(512, dtype=np.float32) * (1 + layer + expert) / 19
                    ).reshape(2, 256)
                    raw = (values.view(np.uint32) >> 16).astype("<u2").tobytes()
                    tensors[name] = ([2, 256], raw)
                    weight_map[name] = shard
        _write_safetensors(source / shard, tensors)
        (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
        (source / "config.json").write_text("{}\n")
        plan_path = root / "real-plan.json"
        plan_path.write_text(json.dumps({
            "format": "bw24-expert-tier-plan-v2",
            "model": {"expert_count": 2, "moe_layers": [1, 2]},
            "pruned_experts": {},
            "assignments": [
                {"layer": 1, "experts": [0, 1], "qtype": "Q2_K"},
                {"layer": 2, "experts": [0, 1], "qtype": "NVFP4"},
            ],
        }))
        base_args = dict(
            source_dir=str(source), fallback_dir=None, plan=str(plan_path), max_work_mb=8,
            resume=False, workers=1, tensor_overrides=None,
        )
        prepare(SimpleNamespace(out_dir=str(sequential), layers=None, manifest_fragment=None, **base_args))
        fragment_paths = []
        for layer in (1, 2):
            fragment = root / f"real-fragment-{layer}.json"
            prepare(SimpleNamespace(
                out_dir=str(parallel), layers=str(layer), manifest_fragment=str(fragment), **base_args
            ))
            fragment_paths.append(fragment)
        merged = merge(SimpleNamespace(
            fragment=fragment_paths, plan=plan_path, out_dir=parallel,
            tensor_overrides=None, output=parallel / "manifest.json",
        ))
        reference = json.loads((sequential / "manifest.json").read_text())
        assert merged["tensors"] == reference["tensors"]
        assert merged["tier_summary"] == reference["tier_summary"]
        assert merged["artifact_bytes"] == reference["artifact_bytes"]
        for payload in sorted((sequential / "experts").glob("*.bin")):
            assert payload.read_bytes() == (parallel / "experts" / payload.name).read_bytes()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fragment", type=Path, action="append", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tensor-overrides", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        print("expert overlay fragment merge self-test: PASS")
        return
    args = parse_args()
    result = merge(args)
    print(
        f"wrote {args.output} sha256={sha256_file(args.output)} "
        f"tensors={len(result['tensors'])} artifact_bytes={result['artifact_bytes']}"
    )


if __name__ == "__main__":
    main()
