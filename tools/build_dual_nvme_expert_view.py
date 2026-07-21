#!/usr/bin/env python3
"""Build a byte-verified expert view striped across the source and one second filesystem."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any


CHUNK_BYTES = 16 * 1024 * 1024


def payload_name(raw: Any) -> str:
    if not isinstance(raw, str):
        raise ValueError("manifest payload file must be a string")
    parts = raw.split("/")
    if (
        len(parts) < 2
        or parts[0] != "experts"
        or any(part in {"", ".", ".."} for part in parts)
        or Path(raw).is_absolute()
    ):
        raise ValueError(f"unsafe expert payload path: {raw!r}")
    return raw


def contained_path(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve(strict=False)
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError(f"expert payload escapes artifact root: {relative!r}")
    return candidate


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def link_exact(target: Path, link: Path) -> None:
    target = target.resolve()
    if os.path.lexists(link):
        if link.is_symlink() and link.resolve() == target:
            return
        raise FileExistsError(f"refusing to replace existing path: {link}")
    link.symlink_to(target, target_is_directory=target.is_dir())


def copy_verified(source: Path, destination: Path) -> str:
    if destination.exists():
        if not destination.is_file() or destination.stat().st_size != source.stat().st_size:
            raise FileExistsError(f"invalid partial destination: {destination}")
    else:
        partial = destination.with_name(destination.name + ".partial")
        if partial.exists():
            raise FileExistsError(f"remove or inspect stale partial copy: {partial}")
        with source.open("rb") as src, partial.open("xb") as dst:
            shutil.copyfileobj(src, dst, CHUNK_BYTES)
            dst.flush()
            os.fsync(dst.fileno())
        shutil.copystat(source, partial, follow_symlinks=True)
        partial.replace(destination)
    source_hash = sha256(source)
    destination_hash = sha256(destination)
    if destination_hash != source_hash:
        raise OSError(f"copy hash mismatch: {source} -> {destination}")
    return source_hash


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_runtime", type=Path)
    parser.add_argument("dual_nvme_view", type=Path)
    parser.add_argument(
        "--allow-same-device",
        action="store_true",
        help="permit a single-filesystem view for functional testing only",
    )
    args = parser.parse_args()

    source_runtime = args.source_runtime.resolve()
    output = args.dual_nvme_view.absolute()
    manifest_path = source_runtime / "manifest.json"
    manifest: dict[str, Any] = json.loads(manifest_path.read_text())
    files = sorted({payload_name(value["file"]) for value in manifest["tensors"].values()})
    if not files:
        raise ValueError("manifest must reference at least one experts/ payload file")

    sources = {name: contained_path(source_runtime, name) for name in files}
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing expert payloads: {missing[:3]}")

    output.mkdir(parents=True, exist_ok=True)
    source_device = source_runtime.stat().st_dev
    output_device = output.stat().st_dev
    if source_device == output_device and not args.allow_same_device:
        raise ValueError(
            "source_runtime and dual_nvme_view are on the same filesystem; "
            "choose a second NVMe mount or pass --allow-same-device for functional testing"
        )

    # Longest-processing-time partition balances physical payload bytes across the two devices.
    bins: list[list[str]] = [[], []]
    totals = [0, 0]
    for name in sorted(files, key=lambda item: (-sources[item].stat().st_size, item)):
        device = 0 if totals[0] <= totals[1] else 1
        bins[device].append(name)
        totals[device] += sources[name].stat().st_size

    experts = output / "experts"
    experts.mkdir(exist_ok=True)
    for entry in source_runtime.iterdir():
        if entry.name == "experts":
            continue
        link_exact(entry, output / entry.name)

    copied_hashes: dict[str, str] = {}
    secondary = set(bins[1])
    for index, name in enumerate(files, 1):
        destination = contained_path(output, name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if name in secondary:
            copied_hashes[name] = copy_verified(sources[name], destination)
        else:
            link_exact(sources[name], destination)
        if index % 25 == 0 or index == len(files):
            print(f"prepared {index}/{len(files)} expert files", flush=True)

    receipt = {
        "format": "bw24-dual-nvme-expert-view-v1",
        "source_runtime": str(source_runtime),
        "view": str(output),
        "source_device": source_device,
        "secondary_device": output_device,
        "device_bytes": totals,
        "source_device_files": sorted(bins[0]),
        "secondary_device_files": sorted(bins[1]),
        "secondary_sha256": copied_hashes,
        "manifest_sha256": sha256(manifest_path),
    }
    receipt_path = output / "dual-nvme-view.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(
        f"dual-NVMe view ready: {len(bins[0])} source files / {len(bins[1])} copied files, "
        f"{totals[0] / 1e9:.2f} GB / {totals[1] / 1e9:.2f} GB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
