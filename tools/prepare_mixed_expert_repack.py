#!/usr/bin/env python3
"""Prepare a sparse bw24 overlay that quantizes only explicitly selected MoE experts.

The base Hugging Face checkpoint remains untouched. Selected BF16 expert projections are written
as GGUF-layout Q4_K rows; every tensor absent from the overlay resolves from ``source_dir`` at
runtime. Run bw24 with ``BW24_FULL_PREC=1`` so unselected BF16 experts stay BF16.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import struct
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from hy3_mlx_to_q4k import (
    Q4K_BLOCK,
    Q4K_BYTES,
    SafeTensorDir,
    sha256_file,
    trim_process_memory,
    write_q4k_from_bf16,
)


PLAN_FORMAT = "bw24-per-expert-quant-plan-v1"
OVERLAY_FORMAT = "bw24-expert-overlay-v1"
PROJECTIONS = ("gate", "up", "down")


def load_assignments(path: Path) -> tuple[dict[str, Any], list[tuple[int, int, str, str]]]:
    plan = json.loads(path.read_text())
    if plan.get("format") != PLAN_FORMAT:
        raise ValueError(f"{path}: format must be {PLAN_FORMAT!r}")
    if plan.get("default_qtype", "BF16") != "BF16":
        raise ValueError("v1 overlays require default_qtype=BF16")
    expanded: dict[tuple[int, int, str], str] = {}
    for i, group in enumerate(plan.get("assignments", [])):
        layer = int(group["layer"])
        experts = [int(x) for x in group.get("experts", [])]
        projections = group.get("projections", list(PROJECTIONS))
        qtype = group.get("qtype", "Q4_K")
        if layer < 0 or not experts or any(expert < 0 for expert in experts):
            raise ValueError(f"assignment {i}: layer and expert ids must be non-negative")
        if qtype != "Q4_K":
            raise ValueError(f"assignment {i}: v1 producer supports Q4_K, got {qtype}")
        if not projections or any(proj not in PROJECTIONS for proj in projections):
            raise ValueError(f"assignment {i}: projections must be drawn from {PROJECTIONS}")
        for expert in experts:
            for proj in projections:
                key = (layer, expert, proj)
                if key in expanded:
                    raise ValueError(f"assignment {i}: duplicate selection {key}")
                expanded[key] = qtype
    if not expanded:
        raise ValueError("plan selects no experts")
    assignments = [(layer, expert, proj, qtype)
                   for (layer, expert, proj), qtype in sorted(expanded.items())]
    return plan, assignments


def source_expert_name(store: SafeTensorDir, layer: int, expert: int, proj: str) -> str:
    ordinary = f"model.layers.{layer}.mlp.experts.{expert}.{proj}_proj.weight"
    mixtral_w = {"gate": "w1", "down": "w2", "up": "w3"}[proj]
    minimax = f"model.layers.{layer}.block_sparse_moe.experts.{expert}.{mixtral_w}.weight"
    candidates = [ordinary, minimax]
    candidates += [f"model.language_model.{name[len('model.') :]}" for name in candidates]
    candidates += [f"language_model.{name}" for name in candidates]
    for name in candidates:
        if name in store.weight_map:
            return name
    raise KeyError(f"no source tensor for layer={layer} expert={expert} projection={proj}")


def prepare(args: argparse.Namespace) -> None:
    source_dir = Path(args.source_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    plan_path = Path(args.plan).resolve()
    plan, assignments = load_assignments(plan_path)
    if not (source_dir / "model.safetensors.index.json").exists():
        raise FileNotFoundError("v1 producer requires model.safetensors.index.json in source_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "experts").mkdir(exist_ok=True)
    store = SafeTensorDir(source_dir)
    max_work = int(args.max_work_mb) << 20
    manifest: dict[str, Any] = {
        "format": OVERLAY_FORMAT,
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "source_dir": str(source_dir),
        "quality": "unverified - pending remote-machine correctness and public eval gates",
        "runtime_env": {"BW24_FULL_PREC": "1"},
        "plan": plan,
        "plan_sha256": sha256_file(plan_path),
        "source_fingerprints": {},
        "tensors": {},
    }
    for name in ("config.json", "model.safetensors.index.json"):
        path = source_dir / name
        if path.exists():
            manifest["source_fingerprints"][name] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }

    try:
        for layer, expert, proj, qtype in assignments:
            source_name = source_expert_name(store, layer, expert, proj)
            info = store.info(source_name)
            if info.dtype != "BF16" or len(info.shape) != 2:
                raise ValueError(f"{source_name}: expected 2D BF16, got {info.dtype} {info.shape}")
            out_f, in_f = info.shape
            if in_f % Q4K_BLOCK:
                raise ValueError(f"{source_name}: in_features={in_f} is not Q4_K aligned")
            row_bytes = in_f // Q4K_BLOCK * Q4K_BYTES
            expected = out_f * row_bytes
            rel = Path("experts") / f"blk{layer}-{proj}-expert{expert}.q4k"
            out_path = out_dir / rel
            if not (args.resume and out_path.exists() and out_path.stat().st_size == expected):
                with out_path.open("wb") as out:
                    shape, written = write_q4k_from_bf16(store, source_name, out, max_work)
                if shape != [out_f, in_f] or written != expected:
                    raise RuntimeError(
                        f"{source_name}: wrote shape={shape} bytes={written}, expected {info.shape} {expected}"
                    )
            mapped = f"blk.{layer}.ffn_{proj}_exps.{expert}.weight"
            manifest["tensors"][mapped] = {
                "source": source_name,
                "file": str(rel),
                "qtype": qtype,
                "ne": [in_f, out_f],
                "row_bytes": row_bytes,
                "bytes": expected,
                "sha256": sha256_file(out_path),
            }
            print(f"{qtype:5s} {source_name} -> {rel} ({expected / 1e6:.2f} MB)", flush=True)
            store.drop_cached_shards()
            trim_process_memory()
    finally:
        store.close()

    manifest_path = out_dir / "manifest.json"
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    tmp.replace(manifest_path)
    print(f"wrote {manifest_path} ({len(manifest['tensors'])} quantized projections)")


def _write_safetensors(path: Path, tensors: dict[str, tuple[list[int], bytes]]) -> None:
    header: dict[str, Any] = {}
    body = bytearray()
    for name, (shape, raw) in tensors.items():
        start = len(body)
        body.extend(raw)
        header[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [start, len(body)]}
    encoded = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + body)


def self_test() -> None:
    root = Path(tempfile.mkdtemp(prefix="bw24-mixed-expert-test-"))
    try:
        source = root / "source"
        out = root / "overlay"
        source.mkdir()
        name = "model.layers.0.mlp.experts.1.gate_proj.weight"
        values = np.tile(
            np.linspace(-1.0, 1.0, Q4K_BLOCK, dtype=np.float32), 2
        )
        bf16 = (values.view(np.uint32) >> 16).astype("<u2").tobytes()
        shard = "model-00001-of-00001.safetensors"
        _write_safetensors(source / shard, {name: ([2, Q4K_BLOCK], bf16)})
        (source / "model.safetensors.index.json").write_text(json.dumps({
            "metadata": {}, "weight_map": {name: shard},
        }))
        (source / "config.json").write_text("{}")
        plan = root / "plan.json"
        plan.write_text(json.dumps({
            "format": PLAN_FORMAT,
            "default_qtype": "BF16",
            "assignments": [{"layer": 0, "experts": [1], "projections": ["gate"], "qtype": "Q4_K"}],
        }))
        prepare(SimpleNamespace(
            source_dir=str(source), out_dir=str(out), plan=str(plan), max_work_mb=8, resume=False,
        ))
        manifest = json.loads((out / "manifest.json").read_text())
        mapped = "blk.0.ffn_gate_exps.1.weight"
        assert manifest["format"] == OVERLAY_FORMAT
        assert list(manifest["tensors"]) == [mapped]
        rec = manifest["tensors"][mapped]
        assert rec["qtype"] == "Q4_K" and rec["bytes"] == 2 * Q4K_BYTES
        assert len(rec["sha256"]) == 64
        assert (out / rec["file"]).stat().st_size == 2 * Q4K_BYTES
        print("mixed expert overlay self-test: PASS")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("source_dir")
    prep.add_argument("out_dir")
    prep.add_argument("--plan", required=True)
    prep.add_argument("--max-work-mb", type=int, default=512)
    prep.add_argument("--resume", action="store_true")
    sub.add_parser("test")
    args = parser.parse_args()
    if args.cmd == "prepare":
        prepare(args)
    else:
        self_test()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
