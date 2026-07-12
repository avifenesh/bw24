#!/usr/bin/env python3
"""Validate a tiered-expert artifact's frozen plan, byte ranges, and expert coverage."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections import defaultdict
from math import prod
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
    projections = ("gate", "up", "down")
    expected_experts = {
        (layer, expert)
        for layer in layers
        for expert in range(n_expert)
        if expert not in pruned.get(layer, set())
    }
    expected_projections = {
        (layer, expert, projection)
        for layer, expert in expected_experts
        for projection in projections
    }
    assigned_qtypes: dict[tuple[int, int, str], str] = {}
    for assignment_index, assignment in enumerate(plan.get("assignments", [])):
        layer = int(assignment["layer"])
        qtype = assignment["qtype"]
        experts = [int(expert) for expert in assignment["experts"]]
        assignment_projections = assignment.get("projections", list(projections))
        if layer not in layer_set:
            raise ValueError(f"assignment {assignment_index}: layer {layer} is outside the model")
        if qtype not in allowed_qtypes:
            raise ValueError(f"assignment {assignment_index}: forbidden expert qtype {qtype}")
        if len(experts) != len(set(experts)):
            raise ValueError(f"assignment {assignment_index}: duplicate expert id")
        if (
            not assignment_projections
            or len(assignment_projections) != len(set(assignment_projections))
            or any(projection not in projections for projection in assignment_projections)
        ):
            raise ValueError(
                f"assignment {assignment_index}: projections must be distinct and drawn from "
                f"{projections}"
            )
        for expert in experts:
            if expert < 0 or expert >= n_expert:
                raise ValueError(
                    f"assignment {assignment_index}: expert {expert} outside 0..{n_expert - 1}"
                )
            if expert in pruned.get(layer, set()):
                raise ValueError(f"assignment {assignment_index}: pruned expert {layer}:{expert}")
            for projection in assignment_projections:
                key = (layer, expert, projection)
                if key in assigned_qtypes:
                    raise ValueError(
                        f"overlapping assignments for expert projection "
                        f"{layer}:{expert}:{projection}"
                    )
                assigned_qtypes[key] = qtype
    assigned_projections = set(assigned_qtypes)
    if assigned_projections != expected_projections:
        raise ValueError(
            f"expert projection assignment mismatch: "
            f"missing={len(expected_projections - assigned_projections)} "
            f"extra={len(assigned_projections - expected_projections)}"
        )

    expected_qtypes = {
        f"blk.{layer}.ffn_{proj}_exps.{expert}.weight": assigned_qtypes[(layer, expert, proj)]
        for layer, expert in expected_experts
        for proj in projections
    }
    tensors = manifest.get("tensors", {})
    override_names: set[str] = set()
    override_bytes = 0
    override_metadata = manifest.get("tensor_overrides")
    if override_metadata is not None:
        receipt_path = Path(override_metadata["receipt_path"])
        if not receipt_path.is_file() or sha256(receipt_path) != override_metadata["receipt_sha256"]:
            raise ValueError("tensor override receipt is missing or its hash differs")
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("format") != "bw24-tensor-overrides-v1":
            raise ValueError("tensor override receipt has the wrong format")
        receipt_blob = receipt.get("blob", {})
        receipt_tensors = receipt.get("tensors", {})
        if not isinstance(receipt_tensors, dict) or not receipt_tensors:
            raise ValueError("tensor override receipt has no tensors")
        override_names = set(receipt_tensors)
        override_bytes = int(override_metadata["bytes"])
        if (
            int(override_metadata["tensor_count"]) != len(override_names)
            or override_bytes != int(receipt_blob.get("bytes", -1))
            or override_metadata["blob_sha256"] != receipt_blob.get("sha256")
        ):
            raise ValueError("tensor override metadata differs from its receipt")
        installed_rel = Path("overrides") / f"{override_metadata['blob_sha256']}.bin"
        installed = root / installed_rel
        if (
            not installed.is_file()
            or installed.stat().st_size != override_bytes
            or sha256(installed) != override_metadata["blob_sha256"]
        ):
            raise ValueError("installed tensor override blob is missing or differs")
        allowed_suffixes = (".ffn_gate_inp.weight", ".exp_probs_b.bias")
        for name, receipt_record in receipt_tensors.items():
            record = tensors.get(name)
            ne = receipt_record.get("ne")
            if (
                not name.startswith("blk.")
                or not name.endswith(allowed_suffixes)
                or not isinstance(ne, list)
                or not ne
                or any(int(value) <= 0 for value in ne)
                or receipt_record.get("qtype") != "F32"
                or record is None
            ):
                raise ValueError(f"invalid tensor override {name}")
            size = int(receipt_record["bytes"])
            offset = int(receipt_record["offset"])
            if size != prod(int(value) for value in ne) * 4 or offset < 0:
                raise ValueError(f"invalid F32 tensor override extent for {name}")
            expected_record = {
                "source": receipt_record.get("source", "healed-router"),
                "file": str(installed_rel),
                "offset": offset,
                "qtype": "F32",
                "ne": [int(value) for value in ne],
                "bytes": size,
            }
            if record != expected_record or offset + size > override_bytes:
                raise ValueError(f"installed tensor override record differs for {name}")

    expected_tensor_names = set(expected_qtypes) | override_names
    if set(tensors) != expected_tensor_names:
        raise ValueError(
            f"tensor coverage mismatch: missing={len(expected_tensor_names - set(tensors))} "
            f"extra={len(set(tensors) - expected_tensor_names)}"
        )

    ranges: dict[Path, list[tuple[int, int, str]]] = defaultdict(list)
    qtypes: dict[str, int] = defaultdict(int)
    total = 0
    for name, rec in tensors.items():
        if name in override_names:
            path = root / rec["file"]
            start = int(rec.get("offset", 0))
            ranges[path].append((start, start + int(rec["bytes"]), name))
            continue
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
    if total + override_bytes != int(manifest.get("payload_bytes", total)):
        raise ValueError(
            f"payload byte total {total + override_bytes} != manifest "
            f"{manifest.get('payload_bytes')}"
        )

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
            "assignments": [
                {"layer": 1, "experts": [0], "projections": [proj], "qtype": "Q8_0"}
                for proj in ("gate", "up", "down")
            ],
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

        plan["assignments"].append(
            {"layer": 1, "experts": [0], "projections": ["gate"], "qtype": "Q8_0"}
        )
        manifest["plan_canonical_sha256"] = canonical_json_sha256(plan)
        (root / "manifest.json").write_text(json.dumps(manifest))
        try:
            validate(root, False)
        except ValueError as exc:
            assert "overlapping assignments for expert projection 1:0:gate" in str(exc)
        else:
            raise AssertionError("overlapping projection assignment was accepted")
        plan["assignments"].pop()
        manifest["plan_canonical_sha256"] = canonical_json_sha256(plan)

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
        tensors["blk.1.ffn_up_exps.0.weight"]["qtype"] = "Q8_0"

        override_blob = root / "overrides" / ("a" * 64 + ".bin")
        override_blob.parent.mkdir()
        override_blob.write_bytes(b"\0" * 8)
        blob_hash = sha256(override_blob)
        installed = override_blob.with_name(blob_hash + ".bin")
        override_blob.rename(installed)
        override_name = "blk.1.exp_probs_b.bias"
        override_receipt = root / "router-overrides.json"
        override_receipt.write_text(json.dumps({
            "format": "bw24-tensor-overrides-v1",
            "blob": {"path": str(installed), "bytes": 8, "sha256": blob_hash},
            "tensors": {override_name: {
                "source": "healed-router", "offset": 0, "qtype": "F32",
                "ne": [2], "bytes": 8,
            }},
        }))
        tensors[override_name] = {
            "source": "healed-router", "file": f"overrides/{blob_hash}.bin",
            "offset": 0, "qtype": "F32", "ne": [2], "bytes": 8,
        }
        manifest["tensor_overrides"] = {
            "receipt_path": str(override_receipt),
            "receipt_sha256": sha256(override_receipt),
            "blob_sha256": blob_hash, "bytes": 8, "tensor_count": 1,
        }
        manifest["payload_bytes"] = 110
        (root / "manifest.json").write_text(json.dumps(manifest))
        summary = validate(root, False)
        assert summary["retained_experts"] == 1 and summary["artifact_bytes"] == 102
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
