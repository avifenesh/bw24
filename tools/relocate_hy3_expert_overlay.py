#!/usr/bin/env python3
"""Create a lightweight runtime view of a receipt-bound Hy3 expert overlay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relocate(overlay: Path, source: Path, output: Path) -> dict[str, object]:
    overlay = overlay.resolve()
    source = source.resolve()
    output = output.resolve()
    manifest_path = overlay / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("format") not in {"bw24-expert-overlay-v1", "bw24-expert-overlay-v2"}:
        raise ValueError("input is not a bw24 expert overlay")
    if not (overlay / "experts").is_dir():
        raise ValueError("overlay experts directory is missing")

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
    print("relocate Hy3 expert overlay self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("overlay", type=Path, nargs="?")
    parser.add_argument("source", type=Path, nargs="?")
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if None in (args.overlay, args.source, args.output):
        parser.error("overlay, source, and output are required")
    print(json.dumps(relocate(args.overlay, args.source, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
