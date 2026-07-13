#!/usr/bin/env python3
"""Strict ctypes bridge to pinned ggml IQ4_XS and Q4_K quantizers.

Quantization is delegated to an explicitly hashed libggml-base build.  The bridge never falls
back to an in-tree approximation: candidate bytes, dequantized sensitivity measurements, healing,
and final artifacts therefore use the same upstream implementation.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


EXTERNAL_QTYPES = {
    "IQ4_XS": (256, 136),
    "Q4_K": (256, 144),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class GgmlQuantBridge:
    def __init__(self, library: Path, expected_sha256: str, source_commit: str):
        self.library = library.resolve(strict=True)
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            raise ValueError("expected libggml SHA-256 must be 64 lowercase hex characters")
        if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
            raise ValueError("ggml source commit must be a full 40-character git hash")
        actual = sha256_file(self.library)
        if actual != expected_sha256:
            raise ValueError(f"libggml hash mismatch: {actual} != {expected_sha256}")
        self.library_sha256 = actual
        self.source_commit = source_commit
        self._lib = ctypes.CDLL(str(self.library))
        self._quantizers: dict[str, Any] = {}
        self._dequantizers: dict[str, Any] = {}
        for qtype, suffix in (("IQ4_XS", "iq4_xs"), ("Q4_K", "q4_K")):
            quantize = getattr(self._lib, f"quantize_{suffix}")
            quantize.argtypes = (
                ctypes.POINTER(ctypes.c_float), ctypes.c_void_p,
                ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.c_float),
            )
            quantize.restype = ctypes.c_size_t
            dequantize = getattr(self._lib, f"dequantize_row_{suffix}")
            dequantize.argtypes = (
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64,
            )
            dequantize.restype = None
            self._quantizers[qtype] = quantize
            self._dequantizers[qtype] = dequantize

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "implementation": "pinned libggml-base exact C quantizer",
            "library_path": str(self.library),
            "library_sha256": self.library_sha256,
            "llama_cpp_commit": self.source_commit,
        }

    def quantize(
        self, rows: np.ndarray, qtype: str, importance: np.ndarray | None,
    ) -> bytes:
        if qtype not in EXTERNAL_QTYPES:
            raise ValueError(f"unsupported bridge qtype {qtype}")
        source = np.ascontiguousarray(rows, dtype=np.float32)
        if source.ndim != 2 or not np.isfinite(source).all():
            raise ValueError("quantization source must be a finite two-dimensional float array")
        nrows, cols = source.shape
        block, type_size = EXTERNAL_QTYPES[qtype]
        if cols % block:
            raise ValueError(f"{qtype} requires in_features divisible by {block}, got {cols}")
        weights = None
        weight_pointer = ctypes.POINTER(ctypes.c_float)()
        if importance is not None:
            weights = np.ascontiguousarray(importance, dtype=np.float32)
            if weights.shape != (cols,) or not np.isfinite(weights).all():
                raise ValueError(f"{qtype} importance must be a finite vector of length {cols}")
            if np.any(weights < 0) or not np.any(weights > 0):
                raise ValueError(f"{qtype} importance must be non-negative and non-zero")
            weight_pointer = weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        expected = nrows * (cols // block) * type_size
        destination = np.empty(expected, dtype=np.uint8)
        written = self._quantizers[qtype](
            source.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            destination.ctypes.data_as(ctypes.c_void_p),
            nrows,
            cols,
            weight_pointer,
        )
        if written != expected:
            raise RuntimeError(f"{qtype} wrote {written} bytes, expected {expected}")
        return destination.tobytes()

    def dequantize(self, raw: bytes, rows: int, cols: int, qtype: str) -> np.ndarray:
        if qtype not in EXTERNAL_QTYPES:
            raise ValueError(f"unsupported bridge qtype {qtype}")
        block, type_size = EXTERNAL_QTYPES[qtype]
        if rows <= 0 or cols <= 0 or cols % block:
            raise ValueError(f"invalid {qtype} matrix shape {(rows, cols)}")
        expected = rows * (cols // block) * type_size
        if len(raw) != expected:
            raise ValueError(f"{qtype} payload has {len(raw)} bytes, expected {expected}")
        encoded = np.frombuffer(raw, dtype=np.uint8)
        restored = np.empty(rows * cols, dtype=np.float32)
        # Both upstream row decoders iterate independent 256-value blocks.  Passing the complete
        # contiguous matrix extent is equivalent to row-by-row decoding and avoids Python calls per
        # output row; the exact equivalence is covered by self_test below.
        self._dequantizers[qtype](
            encoded.ctypes.data_as(ctypes.c_void_p),
            restored.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            rows * cols,
        )
        return restored.reshape(rows, cols)

    def quant_dequant(
        self, rows: np.ndarray, qtype: str, importance: np.ndarray | None,
    ) -> tuple[bytes, np.ndarray]:
        raw = self.quantize(rows, qtype, importance)
        return raw, self.dequantize(raw, *rows.shape, qtype)


def load_importance_sidecar(
    receipt: dict[str, Any], expert_count: int, hidden_size: int, intermediate_size: int,
) -> dict[str, np.ndarray]:
    path = Path(receipt["path"])
    if (
        not path.is_file()
        or path.stat().st_size != int(receipt["bytes"])
        or sha256_file(path) != receipt["sha256"]
    ):
        raise ValueError(f"importance sidecar is missing or changed: {path}")
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != {"input", "down"}:
            raise ValueError(f"{path}: importance sidecar must contain input and down")
        result = {
            "input": np.array(payload["input"], dtype=np.float32, copy=True),
            "down": np.array(payload["down"], dtype=np.float32, copy=True),
        }
    expected = {
        "input": (expert_count, hidden_size),
        "down": (expert_count, intermediate_size),
    }
    for name, values in result.items():
        if values.shape != expected[name] or not np.isfinite(values).all():
            raise ValueError(f"{path}: {name} importance has invalid shape or values")
        if np.any(values < 0) or np.any(np.sum(values, axis=1) <= 0):
            raise ValueError(f"{path}: {name} importance must be non-negative and non-zero per expert")
    return result


def self_test(library: Path, expected_sha256: str, source_commit: str) -> None:
    bridge = GgmlQuantBridge(library, expected_sha256, source_commit)
    rng = np.random.default_rng(19)
    rows = rng.normal(size=(3, 512)).astype(np.float32)
    importance = np.square(rng.normal(size=512)).astype(np.float32) + 1e-8
    for qtype in EXTERNAL_QTYPES:
        raw = bridge.quantize(rows, qtype, importance)
        restored = bridge.dequantize(raw, *rows.shape, qtype)
        assert restored.shape == rows.shape and np.isfinite(restored).all()
        assert raw == bridge.quantize(rows.copy(), qtype, importance.copy())
        # Verify the one-call matrix decode against the upstream row ABI itself.
        block, type_size = EXTERNAL_QTYPES[qtype]
        row_bytes = rows.shape[1] // block * type_size
        expected = np.empty_like(restored)
        encoded = np.frombuffer(raw, dtype=np.uint8)
        for row in range(len(rows)):
            bridge._dequantizers[qtype](
                ctypes.c_void_p(encoded.ctypes.data + row * row_bytes),
                expected[row].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                rows.shape[1],
            )
        assert np.array_equal(restored, expected), qtype


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--ggml-lib", type=Path, required=True)
    parser.add_argument("--ggml-lib-sha256", required=True)
    parser.add_argument("--ggml-source-commit", required=True)
    args = parser.parse_args()
    if not args.self_test:
        raise SystemExit("only --self-test is supported")
    self_test(args.ggml_lib, args.ggml_lib_sha256, args.ggml_source_commit)
    print("ggml quant bridge self-test: PASS")


if __name__ == "__main__":
    main()
