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


def canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def legacy_pretty_json_sha256(value: object) -> str:
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
    expected_plan_hash = manifest.get("plan_canonical_sha256", manifest["plan_sha256"])
    actual_plan_hash = (
        canonical_json_sha256(plan)
        if "plan_canonical_sha256" in manifest
        else legacy_pretty_json_sha256(plan)
    )
    if actual_plan_hash != expected_plan_hash:
        raise ValueError(
            f"embedded plan hash mismatch: {actual_plan_hash} != {expected_plan_hash}"
        )
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

    qtype_geometry = {
        "Q8_0": (32, 34),
        "Q2_K": (256, 84),
        "Q3_K": (256, 110),
        "NVFP4": (64, 36),
    }
    allowed_qtypes = set(qtype_geometry)
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
        ne = [int(x) for x in rec.get("ne", [])]
        if len(ne) != 2:
            raise ValueError(f"{name}: expected two-dimensional ne metadata")
        block, type_size = qtype_geometry[qtype]
        if ne[0] % block:
            raise ValueError(f"{name}: ne[0]={ne[0]} is not aligned for {qtype}")
        expected_row_bytes = ne[0] // block * type_size
        if int(rec.get("row_bytes", -1)) != expected_row_bytes:
            raise ValueError(
                f"{name}: row_bytes {rec.get('row_bytes')} != {expected_row_bytes} for {qtype}"
            )
        expected_bytes = ne[1] * expected_row_bytes
        if int(rec["bytes"]) != expected_bytes:
            raise ValueError(f"{name}: bytes {rec['bytes']} != {expected_bytes} for {qtype}")
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
        (root / "experts.bin").write_bytes(bytes(range(102)))
        tensors = {}
        for offset, proj in enumerate(("gate", "up", "down")):
            tensors[f"blk.1.ffn_{proj}_exps.0.weight"] = {
                "file": "experts.bin", "offset": offset * 34, "bytes": 34,
                "qtype": "Q8_0", "ne": [32, 1], "row_bytes": 34,
            }
        plan = {
            "format": "bw24-expert-tier-plan-v2",
            "model": {"expert_count": 2, "moe_layers": [1]},
            "pruned_experts": {"1": [1]},
            "assignments": [{"layer": 1, "experts": [0], "qtype": "Q8_0"}],
            "calibration": {"public_eval_data_used_for_selection": False},
        }
        manifest = {
            "format": "bw24-expert-overlay-v2", "plan": plan, "plan_sha256": "test",
            "plan_canonical_sha256": canonical_json_sha256(plan),
            "pruned_experts": {"1": [1]}, "artifact_bytes": 102, "tensors": tensors,
        }
        (root / "manifest.json").write_text(json.dumps(manifest))
        summary = validate(root, False)
        assert summary["retained_experts"] == 1 and summary["artifact_bytes"] == 102

        plan["assignments"][0]["qtype"] = "Q3_K"
        (root / "manifest.json").write_text(json.dumps(manifest))
        try:
            validate(root, False)
        except ValueError as exc:
            assert "embedded plan hash mismatch" in str(exc)
        else:
            raise AssertionError("tampered embedded plan was accepted")
        plan["assignments"][0]["qtype"] = "Q8_0"

        manifest["plan_canonical_sha256"] = "0" * 64
        (root / "manifest.json").write_text(json.dumps(manifest))
        try:
            validate(root, False)
        except ValueError as exc:
            assert "embedded plan hash mismatch" in str(exc)
        else:
            raise AssertionError("tampered embedded-plan hash was accepted")
        manifest["plan_canonical_sha256"] = canonical_json_sha256(plan)

        del manifest["plan_canonical_sha256"]
        manifest["plan_sha256"] = legacy_pretty_json_sha256(plan)
        (root / "manifest.json").write_text(json.dumps(manifest))
        summary = validate(root, False)
        assert summary["retained_experts"] == 1 and summary["artifact_bytes"] == 102

        plan["assignments"][0]["qtype"] = "Q3_K"
        (root / "manifest.json").write_text(json.dumps(manifest))
        try:
            validate(root, False)
        except ValueError as exc:
            assert "embedded plan hash mismatch" in str(exc)
        else:
            raise AssertionError("tampered legacy embedded plan was accepted")
        plan["assignments"][0]["qtype"] = "Q8_0"
        manifest["plan_canonical_sha256"] = canonical_json_sha256(plan)

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
