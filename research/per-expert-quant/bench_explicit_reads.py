#!/usr/bin/env python3
"""Read-only storage probe for the proposed bounded spill backend.

This deliberately stops before CUDA: each worker reuses one fixed-size userspace buffer and issues
exact preadv calls. It answers whether explicit multi-megabyte reads at bounded queue depth can
outperform page-fault-driven mmap on the artifact's real files.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import pathlib
import resource
import time


ADVICE = {
    "normal": getattr(os, "POSIX_FADV_NORMAL", 0),
    "random": getattr(os, "POSIX_FADV_RANDOM", 1),
    "sequential": getattr(os, "POSIX_FADV_SEQUENTIAL", 2),
}


def parse_size(value: str) -> int:
    suffixes = {"k": 1 << 10, "m": 1 << 20, "g": 1 << 30}
    text = value.strip().lower()
    multiplier = suffixes.get(text[-1:], 1)
    if multiplier != 1:
        text = text[:-1]
    size = int(text) * multiplier
    if size <= 0:
        raise argparse.ArgumentTypeError("size must be positive")
    return size


def advise(fd: int, advice: int) -> None:
    if not hasattr(os, "posix_fadvise"):
        raise RuntimeError("os.posix_fadvise is required for a controlled cold-cache probe")
    os.posix_fadvise(fd, 0, 0, advice)


def evict(paths: list[pathlib.Path]) -> None:
    dontneed = getattr(os, "POSIX_FADV_DONTNEED", 4)
    for path in paths:
        fd = os.open(path, os.O_RDONLY)
        try:
            advise(fd, dontneed)
        finally:
            os.close(fd)


def read_file(path: pathlib.Path, chunk_bytes: int, advice: int) -> tuple[int, int]:
    expected = path.stat().st_size
    fd = os.open(path, os.O_RDONLY)
    buf = bytearray(chunk_bytes)
    view = memoryview(buf)
    total = 0
    checksum = 0
    try:
        advise(fd, advice)
        while total < expected:
            want = min(chunk_bytes, expected - total)
            done = 0
            while done < want:
                if hasattr(os, "preadv"):
                    got = os.preadv(fd, [view[done:want]], total + done)
                    if got == 0:
                        raise EOFError(f"short read: {path} at {total + done} of {expected}")
                else:
                    data = os.pread(fd, want - done, total + done)
                    got = len(data)
                    if got == 0:
                        raise EOFError(f"short read: {path} at {total + done} of {expected}")
                    view[done : done + got] = data
                done += got
            checksum ^= view[0] ^ view[want - 1]
            total += want
    finally:
        os.close(fd)
    return total, checksum


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=pathlib.Path)
    parser.add_argument("--glob", default="*.bin")
    parser.add_argument("--files", type=int, default=32, help="sorted-file prefix; 0 means all")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk", type=parse_size, default=parse_size("4m"))
    parser.add_argument("--advice", choices=sorted(ADVICE), default="random")
    parser.add_argument("--no-evict", action="store_true")
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.files < 0:
        parser.error("--files must be non-negative")
    paths = sorted(args.directory.glob(args.glob))
    if args.files:
        paths = paths[: args.files]
    if not paths:
        parser.error("no input files matched")
    if any(not path.is_file() for path in paths):
        parser.error("every input must be a regular file")

    logical_bytes = sum(path.stat().st_size for path in paths)
    if not args.no_evict:
        evict(paths)

    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(
            pool.map(lambda path: read_file(path, args.chunk, ADVICE[args.advice]), paths)
        )
    wall_s = time.monotonic() - start
    read_bytes = sum(item[0] for item in results)
    if read_bytes != logical_bytes:
        raise RuntimeError(f"read {read_bytes} bytes, expected {logical_bytes}")

    result = {
        "advice": args.advice,
        "bounded_buffer_bytes": args.workers * args.chunk,
        "chunk_bytes": args.chunk,
        "checksum": sum(item[1] for item in results),
        "directory": str(args.directory.resolve()),
        "evicted_before_read": not args.no_evict,
        "file_count": len(paths),
        "logical_bytes": logical_bytes,
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "started_utc": started_utc,
        "throughput_mib_s": read_bytes / (1 << 20) / wall_s,
        "wall_s": wall_s,
        "workers": args.workers,
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
