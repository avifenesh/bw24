#!/usr/bin/env python3
"""Create deterministic repack-only importance sidecars for a frozen allocation plan.

These uniform sidecars do not select experts or qtypes.  They only let the exact pinned ggml
quantizer encode already-frozen IQ3/IQ4/Q4 states when the historical private sidecars are gone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(
    plan_path: Path,
    sidecar_dir: Path,
    output_map: Path,
    output_plan: Path,
    library: Path,
    library_sha256: str,
    source_commit: str,
    hidden_size: int,
    intermediate_size: int,
) -> None:
    plan = json.loads(plan_path.read_text())
    layers = [int(value) for value in plan["model"]["moe_layers"]]
    experts = int(plan["model"]["expert_count"])
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    receipts = {}
    for layer in layers:
        path = sidecar_dir / f"layer-{layer:03d}.npz"
        if not path.exists():
            np.savez_compressed(
                path,
                input=np.ones((experts, hidden_size), dtype=np.float32),
                down=np.ones((experts, intermediate_size), dtype=np.float32),
            )
        with np.load(path) as payload:
            if payload["input"].shape != (experts, hidden_size):
                raise ValueError(f"invalid input importance shape: {path}")
            if payload["down"].shape != (experts, intermediate_size):
                raise ValueError(f"invalid down importance shape: {path}")
        receipts[str(layer)] = {
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "input_shape": [experts, hidden_size],
            "down_shape": [experts, intermediate_size],
            "construction": "uniform repack-only; not used for allocation",
        }
    provenance = {
        qtype: {
            "library_path": str(library.resolve()),
            "library_sha256": library_sha256,
            "llama_cpp_commit": source_commit,
            "importance": "uniform repack-only",
        }
        for qtype in ("IQ3_S", "IQ4_XS", "Q4_K")
    }
    result = {
        "format": "bw24-hy3-repack-importance-v1",
        "purpose": "encode frozen qtypes only; never select experts or precision",
        "public_eval_data_used_for_selection": False,
        "source_plan": {"path": str(plan_path.resolve()), "sha256": sha256(plan_path)},
        "importance_sidecars": receipts,
        "measurement": {"exact_quantizer_implementation": provenance},
    }
    output_map.parent.mkdir(parents=True, exist_ok=True)
    output_map.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    plan["calibration"] = dict(plan.get("calibration", {}))
    plan["calibration"]["quant_sensitivity"] = {
        "path": str(output_map.resolve()),
        "sha256": sha256(output_map),
        "role": "repack-only uniform importance; allocation was already frozen",
    }
    output_plan.parent.mkdir(parents=True, exist_ok=True)
    output_plan.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="bw24-repack-importance-") as tmp:
        root = Path(tmp)
        plan = root / "plan.json"
        plan.write_text(json.dumps({
            "format": "bw24-expert-tier-plan-v2",
            "model": {"moe_layers": [1, 2], "expert_count": 3},
            "calibration": {"public_eval_data_used_for_selection": False},
        }))
        lib = root / "lib.so"
        lib.write_bytes(b"test")
        output_map = root / "importance.json"
        output_plan = root / "bound-plan.json"
        build(plan, root / "sidecars", output_map, output_plan, lib, "a" * 64, "b" * 40, 4, 2)
        result = json.loads(output_map.read_text())
        assert set(result["importance_sidecars"]) == {"1", "2"}
        bound = json.loads(output_plan.read_text())
        assert bound["calibration"]["quant_sensitivity"]["sha256"] == sha256(output_map)


def main() -> None:
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        print("Hy3 repack importance self-test: PASS")
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--sidecar-dir", type=Path, required=True)
    parser.add_argument("--out-map", type=Path, required=True)
    parser.add_argument("--out-plan", type=Path, required=True)
    parser.add_argument("--ggml-lib", type=Path, required=True)
    parser.add_argument("--ggml-lib-sha256", required=True)
    parser.add_argument("--ggml-source-commit", required=True)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--intermediate-size", type=int, default=1536)
    args = parser.parse_args()
    if sha256(args.ggml_lib) != args.ggml_lib_sha256:
        raise SystemExit("ggml library hash mismatch")
    build(
        args.plan,
        args.sidecar_dir,
        args.out_map,
        args.out_plan,
        args.ggml_lib,
        args.ggml_lib_sha256,
        args.ggml_source_commit,
        args.hidden_size,
        args.intermediate_size,
    )
    print(
        f"wrote map={args.out_map} sha256={sha256(args.out_map)} "
        f"plan={args.out_plan} sha256={sha256(args.out_plan)}"
    )


if __name__ == "__main__":
    main()
