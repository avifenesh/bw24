#!/usr/bin/env python3
"""Complete a dual-NVMe expert mirror and emit an inode-to-alternate-path map."""

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


def copy_verified(source: Path, destination: Path) -> str:
    if destination.exists():
        if not destination.is_file() or destination.stat().st_size != source.stat().st_size:
            raise FileExistsError(f"invalid existing mirror file: {destination}")
    else:
        partial = destination.with_name(destination.name + ".partial")
        if partial.exists():
            raise FileExistsError(f"inspect or remove stale partial: {partial}")
        with source.open("rb") as src, partial.open("xb") as dst:
            shutil.copyfileobj(src, dst, CHUNK_BYTES)
            dst.flush()
            os.fsync(dst.fileno())
        shutil.copystat(source, partial, follow_symlinks=True)
        partial.replace(destination)
    source_hash = sha256(source)
    if sha256(destination) != source_hash:
        raise OSError(f"mirror hash mismatch: {source} -> {destination}")
    return source_hash


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dual_nvme_view", type=Path)
    parser.add_argument("mirror_root", type=Path)
    args = parser.parse_args()

    view = args.dual_nvme_view.resolve()
    mirror = args.mirror_root.absolute()
    manifest_path = view / "manifest.json"
    dual_receipt_path = view / "dual-nvme-view.json"
    manifest: dict[str, Any] = json.loads(manifest_path.read_text())
    dual_receipt: dict[str, Any] = json.loads(dual_receipt_path.read_text())
    expected_manifest = dual_receipt["manifest_sha256"]
    if sha256(manifest_path) != expected_manifest:
        raise ValueError("dual-NVMe receipt does not match manifest.json")

    source_runtime = Path(dual_receipt["source_runtime"]).resolve()
    files = sorted({payload_name(value["file"]) for value in manifest["tensors"].values()})
    if not files:
        raise ValueError("manifest must reference only experts/ payload files")

    mirror.mkdir(parents=True, exist_ok=True)
    (mirror / "experts").mkdir(exist_ok=True)
    map_rows: dict[tuple[int, int], Path] = {}
    copied_hashes: dict[str, str] = {}
    hardlinked = 0
    copied = 0
    total_bytes = 0

    for index, name in enumerate(files, 1):
        view_path = view / name
        source_path = contained_path(source_runtime, name)
        destination = contained_path(mirror, name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not view_path.exists() or not source_path.is_file():
            raise FileNotFoundError(f"missing expert payload for {name}")

        # Files already copied into the dual view are on the root NVMe. Hard-link those into the
        # complete mirror; copy only the view entries that still resolve to the /data NVMe.
        view_stat = view_path.stat()
        source_stat = source_path.stat()
        if (view_stat.st_dev, view_stat.st_ino) != (source_stat.st_dev, source_stat.st_ino):
            if destination.exists():
                destination_stat = destination.stat()
                if (destination_stat.st_dev, destination_stat.st_ino) != (
                    view_stat.st_dev,
                    view_stat.st_ino,
                ):
                    raise FileExistsError(f"mirror hard link differs from dual view: {destination}")
            else:
                os.link(view_path, destination, follow_symlinks=True)
            hardlinked += 1
        else:
            copied_hashes[name] = copy_verified(source_path, destination)
            copied += 1

        mirror_stat = destination.stat()
        view_stat = view_path.stat()
        if view_stat.st_size != mirror_stat.st_size:
            raise OSError(f"mirror size mismatch for {name}")
        if view_stat.st_dev == mirror_stat.st_dev:
            alternate = source_path
        else:
            alternate = destination
        alternate_stat = alternate.stat()
        if alternate_stat.st_dev == view_stat.st_dev:
            raise OSError(f"alternate for {name} is not on the other filesystem")
        key = (view_stat.st_dev, view_stat.st_ino)
        prior = map_rows.setdefault(key, alternate.resolve())
        if prior != alternate.resolve():
            raise ValueError(f"conflicting alternate paths for inode {key}")
        total_bytes += view_stat.st_size
        if index % 20 == 0 or index == len(files):
            print(f"prepared {index}/{len(files)} mirrored expert files", flush=True)

    map_path = mirror / "inode-alternates.tsv"
    for path in map_rows.values():
        if "\t" in str(path) or "\n" in str(path):
            raise ValueError(f"mirror path cannot contain tab or newline: {path}")
    map_text = "".join(
        f"{device}\t{inode}\t{path}\n"
        for (device, inode), path in sorted(map_rows.items())
    )
    map_path.write_text(map_text)
    receipt = {
        "format": "bw24-expert-mirror-map-v1",
        "dual_nvme_view": str(view),
        "mirror_root": str(mirror),
        "manifest_sha256": expected_manifest,
        "map_sha256": hashlib.sha256(map_text.encode()).hexdigest(),
        "files": len(files),
        "payload_bytes": total_bytes,
        "hardlinked_files": hardlinked,
        "copied_files": copied,
        "copied_sha256": copied_hashes,
    }
    (mirror / "mirror-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"mirror ready: {hardlinked} hard links, {copied} verified copies, "
        f"{total_bytes / 1e9:.2f} GB; map={map_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
