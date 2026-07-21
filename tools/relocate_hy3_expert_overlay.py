#!/usr/bin/env python3
"""Create a lightweight runtime view of a receipt-bound Hy3 expert overlay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path


EXPERT_SOURCE_RE = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<projection>gate|up|down)_proj\.weight$"
)
EXPERT_OVERLAY_RE = re.compile(
    r"^blk\.(?P<layer>\d+)\.ffn_(?P<projection>gate|up|down)_exps\."
    r"(?P<expert>\d+)\.weight$"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_source_fingerprints(manifest: dict[str, object], source: Path) -> None:
    fingerprints: dict[str, dict[str, object]] = {}
    for key in ("source_fingerprints", "fallback_fingerprints"):
        fingerprints.update(manifest.get(key, {}))
    if not fingerprints:
        raise ValueError("overlay manifest has no source fingerprints")
    for name, expected in fingerprints.items():
        path = source / name
        if not path.is_file():
            raise ValueError(f"source fingerprint file is missing: {name}")
        if path.stat().st_size != int(expected["bytes"]) or sha256(path) != expected["sha256"]:
            raise ValueError(f"source fingerprint mismatch: {name}")


def _symlink_file(source: Path, target: Path) -> None:
    os.symlink(source.resolve(), target)


def _new_output_path(output: Path) -> Path:
    """Resolve only the parent; never follow an existing final-component symlink."""
    if output.is_symlink():
        raise ValueError(f"output path must not be a symlink: {output}")
    return output.parent.resolve() / output.name


def _safe_shard_name(value: object) -> str:
    shard = str(value)
    path = Path(shard)
    if (
        not shard
        or path.is_absolute()
        or len(path.parts) != 1
        or path.name != shard
        or shard in {".", ".."}
    ):
        raise ValueError(f"unsafe source shard name: {shard!r}")
    return shard


def prepare_sparse_source_view(overlay: Path, source: Path, output: Path) -> dict[str, object]:
    """Materialize a tiny HF source view for an expert-complete v2 overlay.

    The original index remains byte-identical. Shards containing any non-expert tensor are real;
    shards containing routed experts only are valid empty safetensors placeholders. The overlay
    must supply every retained routed-expert projection and mask every omitted expert, so an
    accidental fallback lookup still fails closed instead of reading fabricated weights.
    """
    overlay = overlay.resolve()
    source = source.resolve()
    output = _new_output_path(output)
    manifest_path = overlay / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != "bw24-expert-overlay-v2":
        raise ValueError("sparse source views require a bw24 expert overlay v2")
    _verify_source_fingerprints(manifest, source)

    index_path = source / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("source index has no weight_map")

    overlay_sources: set[str] = set()
    for destination, record in manifest.get("tensors", {}).items():
        match = EXPERT_OVERLAY_RE.fullmatch(str(destination))
        if match is None:
            continue
        if not isinstance(record, dict) or not isinstance(record.get("source"), str):
            raise ValueError(f"overlay expert record has no source: {destination}")
        expected_source = (
            f"model.layers.{int(match.group('layer'))}.mlp.experts."
            f"{int(match.group('expert'))}.{match.group('projection')}_proj.weight"
        )
        source_name = record["source"]
        if source_name != expected_source:
            raise ValueError(
                f"overlay expert destination/source mismatch: {destination} -> {source_name}; "
                f"expected {expected_source}"
            )
        if source_name in overlay_sources:
            raise ValueError(f"duplicate overlay expert source: {source_name}")
        overlay_sources.add(source_name)
    source_experts: dict[str, tuple[int, int, str, str]] = {}
    non_expert_shards: set[str] = set()
    all_shards: set[str] = set()
    for name, shard in weight_map.items():
        name, shard = str(name), _safe_shard_name(shard)
        all_shards.add(shard)
        match = EXPERT_SOURCE_RE.fullmatch(name)
        if match is None:
            non_expert_shards.add(shard)
            continue
        source_experts[name] = (
            int(match.group("layer")),
            int(match.group("expert")),
            match.group("projection"),
            shard,
        )

    unknown_sources = sorted(overlay_sources - source_experts.keys())
    if unknown_sources:
        raise ValueError(f"overlay contains unknown expert sources: {unknown_sources[:3]}")
    pruned = {
        int(layer): {int(expert) for expert in experts}
        for layer, experts in manifest.get("pruned_experts", {}).items()
    }
    managed_layers = set(pruned)
    managed_layers.update(
        layer for name, (layer, _expert, _projection, _shard) in source_experts.items()
        if name in overlay_sources
    )
    uncovered: list[str] = []
    unmanaged_expert_shards: set[str] = set()
    for name, (layer, expert, _projection, shard) in source_experts.items():
        if layer not in managed_layers:
            unmanaged_expert_shards.add(shard)
        elif name not in overlay_sources and expert not in pruned.get(layer, set()):
            uncovered.append(name)
    if uncovered:
        raise ValueError(f"overlay neither supplies nor prunes expert sources: {uncovered[:3]}")

    real_shards = non_expert_shards | unmanaged_expert_shards
    expert_only_shards = all_shards - real_shards
    for shard in sorted(real_shards):
        if not (source / shard).is_file():
            raise ValueError(f"required source shard is missing: {shard}")
    for name, shard in weight_map.items():
        if str(shard) in expert_only_shards and EXPERT_SOURCE_RE.fullmatch(str(name)) is None:
            raise AssertionError(f"non-expert tensor {name} was assigned to placeholder shard {shard}")

    output.mkdir(parents=True, exist_ok=False)
    for path in source.iterdir():
        if path.is_file() and path.suffix != ".safetensors":
            _symlink_file(path, output / path.name)
    for shard in sorted(real_shards):
        _symlink_file(source / shard, output / shard)

    empty_path = output / ".expert-only.empty.safetensors"
    empty_header = b"{}"
    empty_path.write_bytes(len(empty_header).to_bytes(8, "little") + empty_header)
    for shard in sorted(expert_only_shards):
        os.symlink(empty_path.name, output / shard)

    real_bytes = sum((source / shard).stat().st_size for shard in real_shards)
    receipt = {
        "format": "bw24-sparse-hf-source-view-v1",
        "overlay_manifest_sha256": sha256(manifest_path),
        "source": str(source),
        "source_index_sha256": sha256(index_path),
        "source_tensors": len(weight_map),
        "overlay_expert_sources": len(overlay_sources),
        "real_non_expert_shards": sorted(non_expert_shards),
        "real_unmanaged_expert_shards": sorted(unmanaged_expert_shards),
        "real_source_shards": sorted(real_shards),
        "real_source_file_bytes": real_bytes,
        "placeholder_expert_only_shards": sorted(expert_only_shards),
    }
    (output / "sparse-source-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    return receipt


def relocate(overlay: Path, source: Path, output: Path) -> dict[str, object]:
    overlay = overlay.resolve()
    source = source.resolve()
    output = _new_output_path(output)
    manifest_path = overlay / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("format") not in {"bw24-expert-overlay-v1", "bw24-expert-overlay-v2"}:
        raise ValueError("input is not a bw24 expert overlay")
    if not (overlay / "experts").is_dir():
        raise ValueError("overlay experts directory is missing")

    _verify_source_fingerprints(manifest, source)
    fingerprints: dict[str, dict[str, object]] = {}
    for key in ("source_fingerprints", "fallback_fingerprints"):
        fingerprints.update(manifest.get(key, {}))

    output.mkdir(parents=True, exist_ok=False)
    os.symlink(overlay / "experts", output / "experts", target_is_directory=True)
    manifest["source_dir"] = str(source)
    manifest["quant_source_dir"] = str(source)
    runtime_manifest = json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n"
    (output / "manifest.json").write_bytes(runtime_manifest)
    receipt = {
        "format": "bw24-relocated-expert-overlay-v1",
        "overlay": str(overlay),
        "published_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "runtime_manifest_sha256": hashlib.sha256(runtime_manifest).hexdigest(),
        "source": str(source),
        "source_fingerprints": fingerprints,
    }
    (output / "relocation-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    return receipt


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="bw24-relocate-overlay-") as tmp:
        root = Path(tmp)
        overlay = root / "overlay"
        source = root / "source"
        overlay.joinpath("experts").mkdir(parents=True)
        source.mkdir()
        source.joinpath("config.json").write_text("{}\n")
        source.joinpath("model.safetensors.index.json").write_text("{}\n")
        fingerprints = {
            name: {"bytes": (source / name).stat().st_size, "sha256": sha256(source / name)}
            for name in ("config.json", "model.safetensors.index.json")
        }
        overlay.joinpath("manifest.json").write_text(
            json.dumps(
                {
                    "format": "bw24-expert-overlay-v2",
                    "source_dir": "/build/source",
                    "quant_source_dir": "/build/source",
                    "source_fingerprints": fingerprints,
                }
            )
        )
        output = root / "runtime"
        receipt = relocate(overlay, source, output)
        runtime = json.loads(output.joinpath("manifest.json").read_text())
        assert output.joinpath("experts").is_symlink()
        assert runtime["source_dir"] == runtime["quant_source_dir"] == str(source.resolve())
        assert receipt["published_manifest_sha256"] == sha256(overlay / "manifest.json")
        source.joinpath("config.json").write_text("tampered\n")
        try:
            relocate(overlay, source, root / "rejected")
        except ValueError as error:
            assert "fingerprint mismatch" in str(error)
        else:
            raise AssertionError("tampered source was accepted")

        sparse_overlay = root / "sparse-overlay"
        sparse_source = root / "sparse-source"
        sparse_overlay.joinpath("experts").mkdir(parents=True)
        sparse_source.mkdir()
        sparse_source.joinpath("config.json").write_text("{}\n")
        dense_shard = "model-00001-of-00003.safetensors"
        expert_shard = "model-00002-of-00003.safetensors"
        draft_shard = "model-00003-of-00003.safetensors"
        sparse_source.joinpath(dense_shard).write_bytes(b"dense")
        sparse_source.joinpath(draft_shard).write_bytes(b"draft")
        source_names = {
            "model.layers.1.mlp.experts.0.gate_proj.weight",
            "model.layers.1.mlp.experts.0.up_proj.weight",
            "model.layers.1.mlp.experts.0.down_proj.weight",
        }
        weight_map = {"model.norm.weight": dense_shard}
        for expert in range(2):
            for projection in ("gate", "up", "down"):
                weight_map[
                    f"model.layers.1.mlp.experts.{expert}.{projection}_proj.weight"
                ] = expert_shard
        for projection in ("gate", "up", "down"):
            weight_map[
                f"model.layers.2.mlp.experts.0.{projection}_proj.weight"
            ] = draft_shard
        sparse_source.joinpath("model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map}, sort_keys=True)
        )
        sparse_fingerprints = {
            name: {
                "bytes": (sparse_source / name).stat().st_size,
                "sha256": sha256(sparse_source / name),
            }
            for name in ("config.json", "model.safetensors.index.json")
        }
        sparse_tensors = {
            f"blk.1.ffn_{name.rsplit('.', 2)[-2].removesuffix('_proj')}_exps.0.weight": {
                "source": name
            }
            for name in sorted(source_names)
        }
        # Non-expert overrides may carry provenance strings in `source`; sparse expert coverage
        # must ignore them rather than treating them as missing HF expert tensors.
        sparse_tensors["blk.1.ffn_gate_inp.weight"] = {"source": "healed-router"}
        sparse_manifest = {
            "format": "bw24-expert-overlay-v2",
            "source_fingerprints": sparse_fingerprints,
            "fallback_fingerprints": sparse_fingerprints,
            "pruned_experts": {"1": [1]},
            "tensors": sparse_tensors,
        }
        sparse_overlay.joinpath("manifest.json").write_text(json.dumps(sparse_manifest))
        sparse_view = root / "sparse-view"
        sparse_receipt = prepare_sparse_source_view(sparse_overlay, sparse_source, sparse_view)
        assert sparse_receipt["real_non_expert_shards"] == [dense_shard]
        assert sparse_receipt["real_unmanaged_expert_shards"] == [draft_shard]
        assert sparse_view.joinpath(dense_shard).is_symlink()
        assert sparse_view.joinpath(draft_shard).resolve() == sparse_source / draft_shard
        assert sparse_view.joinpath(expert_shard).is_symlink()
        assert sparse_view.joinpath(expert_shard).resolve().stat().st_size == 10

        swapped_overlay = root / "swapped-overlay"
        swapped_overlay.joinpath("experts").mkdir(parents=True)
        swapped_manifest = json.loads(json.dumps(sparse_manifest))
        expert_destinations = sorted(
            name for name in swapped_manifest["tensors"] if EXPERT_OVERLAY_RE.fullmatch(name)
        )
        first, second = expert_destinations[:2]
        swapped_manifest["tensors"][first]["source"], swapped_manifest["tensors"][second]["source"] = (
            swapped_manifest["tensors"][second]["source"],
            swapped_manifest["tensors"][first]["source"],
        )
        swapped_overlay.joinpath("manifest.json").write_text(json.dumps(swapped_manifest))
        try:
            prepare_sparse_source_view(
                swapped_overlay,
                sparse_source,
                root / "swapped-view",
            )
        except ValueError as error:
            assert "destination/source mismatch" in str(error)
        else:
            raise AssertionError("swapped expert sources were accepted")

        unsafe_source = root / "unsafe-source"
        unsafe_source.mkdir()
        unsafe_source.joinpath("config.json").write_text("{}\n")
        unsafe_weight_map = dict(weight_map)
        unsafe_weight_map[next(iter(source_names))] = "../escaped.safetensors"
        unsafe_source.joinpath("model.safetensors.index.json").write_text(
            json.dumps({"weight_map": unsafe_weight_map}, sort_keys=True)
        )
        unsafe_fingerprints = {
            name: {
                "bytes": (unsafe_source / name).stat().st_size,
                "sha256": sha256(unsafe_source / name),
            }
            for name in ("config.json", "model.safetensors.index.json")
        }
        unsafe_overlay = root / "unsafe-overlay"
        unsafe_overlay.joinpath("experts").mkdir(parents=True)
        unsafe_manifest = json.loads(json.dumps(sparse_manifest))
        unsafe_manifest["source_fingerprints"] = unsafe_fingerprints
        unsafe_manifest["fallback_fingerprints"] = unsafe_fingerprints
        unsafe_overlay.joinpath("manifest.json").write_text(json.dumps(unsafe_manifest))
        try:
            prepare_sparse_source_view(unsafe_overlay, unsafe_source, root / "unsafe-view")
        except ValueError as error:
            assert "unsafe source shard name" in str(error)
        else:
            raise AssertionError("path-traversing source shard was accepted")
        assert not root.joinpath("escaped.safetensors").exists()

        redirected = root / "redirected"
        redirected.mkdir()
        output_link = root / "output-link"
        output_link.symlink_to(redirected, target_is_directory=True)
        for operation in (
            lambda: prepare_sparse_source_view(sparse_overlay, sparse_source, output_link),
            lambda: relocate(sparse_overlay, sparse_source, output_link),
        ):
            try:
                operation()
            except ValueError as error:
                assert "output path must not be a symlink" in str(error)
            else:
                raise AssertionError("output symlink was followed")

        incomplete_overlay = root / "incomplete-overlay"
        incomplete_overlay.joinpath("experts").mkdir(parents=True)
        incomplete_manifest = json.loads(sparse_overlay.joinpath("manifest.json").read_text())
        incomplete_manifest["tensors"].pop(sorted(incomplete_manifest["tensors"])[0])
        incomplete_overlay.joinpath("manifest.json").write_text(json.dumps(incomplete_manifest))
        try:
            prepare_sparse_source_view(
                incomplete_overlay,
                sparse_source,
                root / "incomplete-view",
            )
        except ValueError as error:
            assert "neither supplies nor prunes" in str(error)
        else:
            raise AssertionError("incomplete overlay was accepted for a sparse source view")
    print("relocate Hy3 expert overlay self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("overlay", type=Path, nargs="?")
    parser.add_argument("source", type=Path, nargs="?")
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument(
        "--sparse-source-view",
        type=Path,
        help="create and use a verified source view with placeholders for expert-only shards",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if None in (args.overlay, args.source, args.output):
        parser.error("overlay, source, and output are required")
    source = args.source
    if args.sparse_source_view is not None:
        prepare_sparse_source_view(args.overlay, source, args.sparse_source_view)
        source = args.sparse_source_view
    print(json.dumps(relocate(args.overlay, source, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
