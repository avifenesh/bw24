#!/usr/bin/env python3
"""Validate a tiered-expert artifact's frozen plan, byte ranges, and expert coverage."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections import defaultdict
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate(root: Path, verify_sources: bool) -> dict[str, int]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != "bw24-expert-overlay-v2":
        raise ValueError("artifact is not a bw24-expert-overlay-v2")
    plan = manifest.get("plan", {})
    if plan.get("format") != "bw24-expert-tier-plan-v2":
        raise ValueError("embedded plan is missing or has the wrong format")
    if manifest.get("plan_sha256") is None:
        raise ValueError("manifest has no plan hash")
    if plan.get("calibration", {}).get("public_eval_data_used_for_selection") is True:
        raise ValueError("plan declares public eval data was used for expert selection")

    model = plan["model"]
    n_expert = int(model["expert_count"])
    layers = [int(x) for x in model["moe_layers"]]
    layer_set = set(layers)
    pruned: dict[int, set[int]] = {}
    for raw_layer, raw_ids in plan.get("pruned_experts", {}).items():
        layer = int(raw_layer)
        ids = [int(expert) for expert in raw_ids]
        if layer not in layer_set:
            raise ValueError(f"prune mask contains layer {layer} outside the model")
        if len(ids) != len(set(ids)):
            raise ValueError(f"layer {layer}: duplicate expert in prune mask")
        if any(expert < 0 or expert >= n_expert for expert in ids):
            raise ValueError(f"layer {layer}: prune mask contains an expert outside 0..{n_expert - 1}")
        pruned[layer] = set(ids)
    manifest_pruned = {
        int(layer): set(ids) for layer, ids in manifest.get("pruned_experts", {}).items()
    }
    if manifest_pruned != pruned:
        raise ValueError("manifest prune mask differs from its embedded plan")

    allowed_qtypes = {"Q2_K", "Q3_K", "NVFP4"}
    expected_experts = {
        (layer, expert)
        for layer in layers
        for expert in range(n_expert)
        if expert not in pruned.get(layer, set())
    }
    assigned_qtypes: dict[tuple[int, int], str] = {}
    for assignment_index, assignment in enumerate(plan.get("assignments", [])):
        layer = int(assignment["layer"])
        qtype = assignment["qtype"]
        experts = [int(expert) for expert in assignment["experts"]]
        if layer not in layer_set:
            raise ValueError(f"assignment {assignment_index}: layer {layer} is outside the model")
        if qtype not in allowed_qtypes:
            raise ValueError(f"assignment {assignment_index}: forbidden expert qtype {qtype}")
        if len(experts) != len(set(experts)):
            raise ValueError(f"assignment {assignment_index}: duplicate expert id")
        for expert in experts:
            key = (layer, expert)
            if expert < 0 or expert >= n_expert:
                raise ValueError(
                    f"assignment {assignment_index}: expert {expert} outside 0..{n_expert - 1}"
                )
            if expert in pruned.get(layer, set()):
                raise ValueError(f"assignment {assignment_index}: pruned expert {layer}:{expert}")
            if key in assigned_qtypes:
                raise ValueError(f"overlapping assignments for expert {layer}:{expert}")
            assigned_qtypes[key] = qtype
    assigned_experts = set(assigned_qtypes)
    if assigned_experts != expected_experts:
        raise ValueError(
            f"expert assignment mismatch: missing={len(expected_experts - assigned_experts)} "
            f"extra={len(assigned_experts - expected_experts)}"
        )

    expected_qtypes = {
        f"blk.{layer}.ffn_{proj}_exps.{expert}.weight": assigned_qtypes[(layer, expert)]
        for layer, expert in expected_experts
        for proj in ("gate", "up", "down")
    }
    tensors = manifest.get("tensors", {})
    if set(tensors) != set(expected_qtypes):
        raise ValueError(
            f"expert coverage mismatch: missing={len(set(expected_qtypes) - set(tensors))} "
            f"extra={len(set(tensors) - set(expected_qtypes))}"
        )

    ranges: dict[Path, list[tuple[int, int, str]]] = defaultdict(list)
    qtypes: dict[str, int] = defaultdict(int)
    total = 0
    for name, rec in tensors.items():
        qtype = rec["qtype"]
        if qtype not in allowed_qtypes:
            raise ValueError(f"{name}: forbidden expert qtype {qtype}")
        if qtype != expected_qtypes[name]:
            raise ValueError(
                f"{name}: qtype {qtype} differs from plan assignment {expected_qtypes[name]}"
            )
        path = root / rec["file"]
        start = int(rec.get("offset", 0))
        end = start + int(rec["bytes"])
        if not path.is_file() or end > path.stat().st_size:
            raise ValueError(f"{name}: byte range [{start},{end}) exceeds {path}")
        ranges[path].append((start, end, name))
        qtypes[qtype] += 1
        total += int(rec["bytes"])
    for path, spans in ranges.items():
        spans.sort()
        for left, right in zip(spans, spans[1:]):
            if left[1] > right[0]:
                raise ValueError(f"overlapping tensor ranges in {path}: {left[2]} and {right[2]}")
    if total != int(manifest.get("artifact_bytes", -1)):
        raise ValueError(f"artifact byte total {total} != manifest {manifest.get('artifact_bytes')}")

    if verify_sources:
        for key, base_key in (("source_fingerprints", "quant_source_dir"), ("fallback_fingerprints", "source_dir")):
            base = Path(manifest[base_key])
            for name, rec in manifest.get(key, {}).items():
                path = base / name
                if not path.is_file() or path.stat().st_size != rec["bytes"] or sha256(path) != rec["sha256"]:
                    raise ValueError(f"source fingerprint mismatch: {path}")
    return {
        "layers": len(layers),
        "retained_experts": len(expected_experts),
        "pruned_experts": sum(len(x) for x in pruned.values()),
        "expert_projections": len(tensors),
        "artifact_bytes": total,
        **{qtype.lower() + "_projections": count for qtype, count in qtypes.items()},
    }


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="bw24-artifact-validator-") as tmp:
        root = Path(tmp)
        (root / "experts.bin").write_bytes(b"abcdef")
        tensors = {}
        for offset, proj in enumerate(("gate", "up", "down")):
            tensors[f"blk.1.ffn_{proj}_exps.0.weight"] = {
                "file": "experts.bin", "offset": offset * 2, "bytes": 2,
                "qtype": "NVFP4",
            }
        plan = {
            "format": "bw24-expert-tier-plan-v2",
            "model": {"expert_count": 2, "moe_layers": [1]},
            "pruned_experts": {"1": [1]},
            "assignments": [{"layer": 1, "experts": [0], "qtype": "NVFP4"}],
            "calibration": {"public_eval_data_used_for_selection": False},
        }
        manifest = {
            "format": "bw24-expert-overlay-v2", "plan": plan, "plan_sha256": "test",
            "pruned_experts": {"1": [1]}, "artifact_bytes": 6, "tensors": tensors,
        }
        (root / "manifest.json").write_text(json.dumps(manifest))
        summary = validate(root, False)
        assert summary["retained_experts"] == 1 and summary["artifact_bytes"] == 6
        tensors["blk.1.ffn_up_exps.0.weight"]["qtype"] = "Q3_K"
        (root / "manifest.json").write_text(json.dumps(manifest))
        try:
            validate(root, False)
        except ValueError as exc:
            assert "differs from plan assignment" in str(exc)
        else:
            raise AssertionError("manifest qtype mismatch was accepted")
        print("artifact validator self-test: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path, nargs="?")
    parser.add_argument("--verify-sources", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.artifact is None:
        parser.error("artifact is required unless --self-test is used")
    summary = validate(args.artifact.resolve(), args.verify_sources)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
