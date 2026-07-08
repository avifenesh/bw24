#!/usr/bin/env python3
"""CPU-only Hy3 REAP50 MLX-affine -> bw24 Q4_K preparation tool.

This intentionally avoids torch, mlx, safetensors, and GPU APIs.  It reads
safetensors with mmap + json, streams rows in bounded chunks, and writes a
bw24-oriented repack directory:

  manifest.json                  machine-readable loader plan
  tensors/*.q4k                   GGUF block_q4_K row-major bytes
  tensors/*.f32                   dequantized router/other affine non-4bit
  tensors/*.bin                   copied non-quant small tensors
  experts/blkL-proj-ExOxI.q4k      stacked expert bytes, expert-axis slowest

The inventory subcommand separately writes a source tensor dtype/shape/shard TSV.

Quality is unverified here: this is a container/byte conversion step only.
"""

from __future__ import annotations

import argparse
import ctypes
import datetime as _dt
import gc
import hashlib
import json
import mmap
import os
import re
import shutil
import struct
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np


Q4K_BLOCK = 256
Q4K_BYTES = 144
DTYPE_SIZES = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U64": 8,
    "U32": 4,
    "U16": 2,
    "U8": 1,
    "BOOL": 1,
}


@dataclass
class TensorInfo:
    name: str
    dtype: str
    shape: list[int]
    data_offsets: tuple[int, int]
    shard: str

    @property
    def nbytes(self) -> int:
        return self.data_offsets[1] - self.data_offsets[0]

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n


class Shard:
    def __init__(self, path: Path):
        self.path = path
        self._fh: BinaryIO = path.open("rb")
        self._mmap = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        hlen = struct.unpack_from("<Q", self._mmap, 0)[0]
        header = json.loads(self._mmap[8 : 8 + hlen])
        self.data_base = 8 + hlen
        self.infos: dict[str, TensorInfo] = {}
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            self.infos[name] = TensorInfo(
                name=name,
                dtype=meta["dtype"],
                shape=[int(x) for x in meta["shape"]],
                data_offsets=(int(meta["data_offsets"][0]), int(meta["data_offsets"][1])),
                shard=path.name,
            )

    def bytes_for(self, info: TensorInfo) -> memoryview:
        s = self.data_base + info.data_offsets[0]
        e = self.data_base + info.data_offsets[1]
        return memoryview(self._mmap)[s:e]

    def close(self) -> None:
        self._mmap.close()
        self._fh.close()


class SafeTensorDir:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        index = json.loads((model_dir / "model.safetensors.index.json").read_text())
        self.metadata = index.get("metadata", {})
        self.weight_map: dict[str, str] = index["weight_map"]
        self._shards: dict[str, Shard] = {}

    def _open_shard(self, shard_name: str) -> Shard:
        shard = self._shards.get(shard_name)
        if shard is None:
            shard = Shard(self.model_dir / shard_name)
            self._shards[shard_name] = shard
        return shard

    def info(self, name: str) -> TensorInfo:
        shard_name = self.weight_map[name]
        shard = self._open_shard(shard_name)
        return shard.infos[name]

    def raw(self, name: str) -> tuple[TensorInfo, memoryview]:
        info = self.info(name)
        return info, self._open_shard(info.shard).bytes_for(info)

    def close(self) -> None:
        for shard in self._shards.values():
            shard.close()
        self._shards.clear()

    def drop_cached_shards(self) -> None:
        self.close()


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def sha256_file(path: Path, chunk: int = 16 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def trim_process_memory() -> None:
    gc.collect()
    if sys.platform.startswith("linux"):
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass


def bf16_to_f32(raw: memoryview | bytes) -> np.ndarray:
    u16 = np.frombuffer(raw, dtype="<u2")
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32)


def tensor_to_f32(info: TensorInfo, raw: memoryview) -> np.ndarray:
    if info.dtype == "F32":
        return np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=False)
    if info.dtype == "F16":
        return np.frombuffer(raw, dtype="<f2").astype(np.float32)
    if info.dtype == "BF16":
        return bf16_to_f32(raw)
    if info.dtype == "U32":
        return np.frombuffer(raw, dtype="<u4").astype(np.float32)
    if info.dtype == "U8":
        return np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
    raise ValueError(f"cannot convert {info.name} dtype {info.dtype} to f32")


def read_numeric(info: TensorInfo, raw: memoryview) -> np.ndarray:
    if info.dtype == "F32":
        return np.frombuffer(raw, dtype="<f4").reshape(info.shape)
    if info.dtype == "F16":
        return np.frombuffer(raw, dtype="<f2").astype(np.float32).reshape(info.shape)
    if info.dtype == "BF16":
        return bf16_to_f32(raw).reshape(info.shape)
    if info.dtype == "U32":
        return np.frombuffer(raw, dtype="<u4").reshape(info.shape)
    if info.dtype == "U8":
        return np.frombuffer(raw, dtype=np.uint8).reshape(info.shape)
    raise ValueError(f"unsupported numeric dtype {info.dtype} for {info.name}")


def mlx_quant_params(config: dict[str, Any], stem: str) -> dict[str, Any]:
    q = config.get("quantization_config") or config.get("quantization") or {}
    default = {
        "group_size": int(q.get("group_size", 64)),
        "bits": int(q.get("bits", 4)),
        "mode": q.get("mode", "affine"),
    }
    override = q.get(stem)
    if isinstance(override, dict):
        merged = default | override
        merged["group_size"] = int(merged["group_size"])
        merged["bits"] = int(merged["bits"])
        return merged
    return default


def mlx_logical_shape(weight_shape: list[int], scale_shape: list[int], bits: int, group_size: int) -> list[int]:
    pack = 32 // bits
    by_weight = list(weight_shape[:-1]) + [weight_shape[-1] * pack]
    by_scale = list(scale_shape[:-1]) + [scale_shape[-1] * group_size]
    if by_weight != by_scale:
        raise ValueError(f"packed/scales shape mismatch: weight->{by_weight}, scale->{by_scale}")
    return by_weight


def dequant_mlx_affine_rows(
    q_words: np.ndarray,
    scales: np.ndarray,
    biases: np.ndarray,
    bits: int,
    group_size: int,
) -> np.ndarray:
    """Dequantize a row chunk from MLX affine packed uint32 words.

    q_words shape: [rows, packed_cols]
    scales/biases shape: [rows, groups]
    """
    if bits <= 0 or 32 % bits != 0:
        raise ValueError(f"bits must divide 32, got {bits}")
    pack = 32 // bits
    groups = scales.shape[1]
    words_per_group = group_size // pack
    q = q_words.reshape(q_words.shape[0], groups, words_per_group)
    shifts = (np.arange(pack, dtype=np.uint32) * bits).reshape(1, 1, 1, pack)
    mask = np.uint32((1 << bits) - 1)
    vals = ((q[..., None] >> shifts) & mask).astype(np.float32)
    vals = vals.reshape(q_words.shape[0], groups, group_size)
    vals *= scales.astype(np.float32)[:, :, None]
    vals += biases.astype(np.float32)[:, :, None]
    return vals.reshape(q_words.shape[0], groups * group_size)


def pack_q4k_scales(sc: np.ndarray, mn: np.ndarray) -> np.ndarray:
    out = np.zeros(sc.shape[:-1] + (12,), dtype=np.uint8)
    out[..., 0:4] = sc[..., 0:4] & 0x3F
    out[..., 4:8] = mn[..., 0:4] & 0x3F
    out[..., 0:4] |= (sc[..., 4:8] >> 4) << 6
    out[..., 4:8] |= (mn[..., 4:8] >> 4) << 6
    out[..., 8:12] = (sc[..., 4:8] & 0x0F) | ((mn[..., 4:8] & 0x0F) << 4)
    return out


def quantize_q4k_rows(rows: np.ndarray) -> bytes:
    """Quantize [rows, in_f] f32 to GGUF block_q4_K bytes.

    The quantizer is deliberately simple but byte-valid: each 32-value subblock
    gets an affine min/scale, then those min/scale values are quantized to the
    Q4_K block's 6-bit tables.
    """
    rows = np.asarray(rows, dtype=np.float32)
    if rows.ndim != 2:
        raise ValueError(f"expected 2D rows, got {rows.shape}")
    r, in_f = rows.shape
    if in_f % Q4K_BLOCK != 0:
        raise ValueError(f"Q4_K requires row length multiple of {Q4K_BLOCK}, got {in_f}")
    nb = in_f // Q4K_BLOCK
    x = rows.reshape(r, nb, 8, 32)
    mn0 = x.min(axis=3)
    mx0 = x.max(axis=3)
    offset = np.maximum(-mn0, 0.0)
    scale0 = (mx0 + offset) / 15.0
    scale0 = np.where(scale0 > 1e-30, scale0, 0.0)

    d = scale0.max(axis=2) / 63.0
    dmin = offset.max(axis=2) / 63.0
    sc_i = np.where(d[..., None] > 0, np.rint(scale0 / d[..., None]), 0).clip(0, 63).astype(np.uint8)
    mn_i = np.where(dmin[..., None] > 0, np.rint(offset / dmin[..., None]), 0).clip(0, 63).astype(np.uint8)

    # Use the actual f16 block scales that will be written, then round codes.
    d16 = d.astype("<f2")
    dmin16 = dmin.astype("<f2")
    d_eff = d16.astype(np.float32)[..., None]
    dm_eff = dmin16.astype(np.float32)[..., None]
    scale_eff = d_eff * sc_i.astype(np.float32)
    min_eff = dm_eff * mn_i.astype(np.float32)
    denom = scale_eff[..., None]
    q = np.zeros_like(x, dtype=np.float32)
    np.divide(x + min_eff[..., None], denom, out=q, where=denom > 0)
    q = np.rint(q).clip(0, 15).astype(np.uint8)

    qbytes = np.empty((r, nb, 128), dtype=np.uint8)
    for p in range(4):
        lo = q[:, :, 2 * p, :]
        hi = q[:, :, 2 * p + 1, :]
        qbytes[:, :, p * 32 : (p + 1) * 32] = lo | (hi << 4)

    packed = np.empty((r, nb, Q4K_BYTES), dtype=np.uint8)
    packed[:, :, 0:2] = d16.reshape(r, nb, 1).view(np.uint8).reshape(r, nb, 2)
    packed[:, :, 2:4] = dmin16.reshape(r, nb, 1).view(np.uint8).reshape(r, nb, 2)
    packed[:, :, 4:16] = pack_q4k_scales(sc_i, mn_i)
    packed[:, :, 16:144] = qbytes
    return packed.reshape(-1).tobytes()


def _f16_from_bytes(raw: np.ndarray) -> np.ndarray:
    return raw.view("<f2").astype(np.float32)


def q4k_dequant_rows(raw: bytes | memoryview, in_f: int) -> np.ndarray:
    if in_f % Q4K_BLOCK != 0:
        raise ValueError(in_f)
    row_bytes = (in_f // Q4K_BLOCK) * Q4K_BYTES
    if len(raw) % row_bytes != 0:
        raise ValueError("raw length is not a whole number of rows")
    rows = len(raw) // row_bytes
    nb = in_f // Q4K_BLOCK
    b = np.frombuffer(raw, dtype=np.uint8).reshape(rows, nb, Q4K_BYTES)
    d = _f16_from_bytes(b[:, :, 0:2].copy()).reshape(rows, nb)
    dmin = _f16_from_bytes(b[:, :, 2:4].copy()).reshape(rows, nb)
    scbuf = b[:, :, 4:16]
    sc = np.empty((rows, nb, 8), dtype=np.uint8)
    mn = np.empty((rows, nb, 8), dtype=np.uint8)
    sc[:, :, 0:4] = scbuf[:, :, 0:4] & 0x3F
    mn[:, :, 0:4] = scbuf[:, :, 4:8] & 0x3F
    sc[:, :, 4:8] = (scbuf[:, :, 8:12] & 0x0F) | ((scbuf[:, :, 0:4] >> 6) << 4)
    mn[:, :, 4:8] = (scbuf[:, :, 8:12] >> 4) | ((scbuf[:, :, 4:8] >> 6) << 4)
    qbytes = b[:, :, 16:144].reshape(rows, nb, 4, 32)
    q = np.empty((rows, nb, 8, 32), dtype=np.uint8)
    for p in range(4):
        q[:, :, 2 * p, :] = qbytes[:, :, p, :] & 0x0F
        q[:, :, 2 * p + 1, :] = qbytes[:, :, p, :] >> 4
    vals = d[:, :, None, None] * sc.astype(np.float32)[:, :, :, None] * q.astype(np.float32)
    vals -= dmin[:, :, None, None] * mn.astype(np.float32)[:, :, :, None]
    return vals.reshape(rows, in_f)


def q4k_dequant_rows_reference(raw: bytes | memoryview, in_f: int) -> np.ndarray:
    if in_f % Q4K_BLOCK != 0:
        raise ValueError(in_f)
    row_bytes = (in_f // Q4K_BLOCK) * Q4K_BYTES
    if len(raw) % row_bytes != 0:
        raise ValueError("raw length is not a whole number of rows")
    rows = len(raw) // row_bytes
    out = np.empty((rows, in_f), dtype=np.float32)
    b = memoryview(raw)
    for r in range(rows):
        row_base = r * row_bytes
        y = 0
        for ib in range(in_f // Q4K_BLOCK):
            base = row_base + ib * Q4K_BYTES
            d = np.frombuffer(b[base : base + 2], dtype="<f2", count=1)[0].astype(np.float32)
            dmin = np.frombuffer(b[base + 2 : base + 4], dtype="<f2", count=1)[0].astype(np.float32)
            scales = b[base + 4 : base + 16]
            qs = b[base + 16 : base + 144]
            qoff = 0
            for pair in range(4):
                g0 = pair * 2
                g1 = g0 + 1
                sc0, mn0 = q4k_scale_min_scalar(scales, g0)
                sc1, mn1 = q4k_scale_min_scalar(scales, g1)
                for l in range(32):
                    out[r, y] = d * sc0 * (qs[qoff + l] & 0x0F) - dmin * mn0
                    y += 1
                for l in range(32):
                    out[r, y] = d * sc1 * (qs[qoff + l] >> 4) - dmin * mn1
                    y += 1
                qoff += 32
    return out


def q4k_scale_min_scalar(scales: memoryview, j: int) -> tuple[int, int]:
    if j < 4:
        return scales[j] & 0x3F, scales[j + 4] & 0x3F
    return (
        (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4),
        (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
    )


def q4k_expected_size(shape: list[int]) -> int:
    if shape[-1] % Q4K_BLOCK != 0:
        raise ValueError(f"shape last dim {shape[-1]} not Q4_K-compatible")
    rows = int(np.prod(shape[:-1], dtype=np.int64))
    return rows * (shape[-1] // Q4K_BLOCK) * Q4K_BYTES


def row_chunk_for(in_f: int, target_bytes: int) -> int:
    # Working set is roughly f32 rows + q4 intermediates; keep comfortably below target_bytes.
    per_row = max(in_f * 16, 1)
    return max(1, min(4096, target_bytes // per_row))


def map_hy3_name(name: str, logical_shape: list[int] | None = None) -> str:
    if name == "model.embed_tokens.weight":
        return "token_embd.weight"
    if name == "model.norm.weight":
        return "output_norm.weight"
    if name == "lm_head.weight":
        return "output.weight"
    m = re.match(r"model\.layers\.(\d+)\.(.+)", name)
    if not m:
        return name
    il = int(m.group(1))
    s = m.group(2)
    attn = {
        "input_layernorm.weight": "attn_norm.weight",
        "post_attention_layernorm.weight": "ffn_norm.weight",
        "self_attn.q_norm.weight": "attn_q_norm.weight",
        "self_attn.k_norm.weight": "attn_k_norm.weight",
        "self_attn.q_proj.weight": "attn_q.weight",
        "self_attn.k_proj.weight": "attn_k.weight",
        "self_attn.v_proj.weight": "attn_v.weight",
        "self_attn.o_proj.weight": "attn_output.weight",
    }
    if s in attn:
        return f"blk.{il}.{attn[s]}"
    dense = {
        "mlp.gate_proj.weight": "ffn_gate.weight",
        "mlp.up_proj.weight": "ffn_up.weight",
        "mlp.down_proj.weight": "ffn_down.weight",
        "mlp.router.gate.weight": "ffn_gate_inp.weight",
        "mlp.router.expert_bias": "exp_probs_b.bias",
        "mlp.shared_mlp.gate_proj.weight": "ffn_gate_shexp.weight",
        "mlp.shared_mlp.up_proj.weight": "ffn_up_shexp.weight",
        "mlp.shared_mlp.down_proj.weight": "ffn_down_shexp.weight",
        "mlp.switch_mlp.gate_proj.weight": "ffn_gate_exps.weight",
        "mlp.switch_mlp.up_proj.weight": "ffn_up_exps.weight",
        "mlp.switch_mlp.down_proj.weight": "ffn_down_exps.weight",
    }
    if s in dense:
        return f"blk.{il}.{dense[s]}"
    return name


def proj_from_exps(mapped: str) -> str | None:
    if mapped.endswith(".ffn_gate_exps.weight"):
        return "gate"
    if mapped.endswith(".ffn_up_exps.weight"):
        return "up"
    if mapped.endswith(".ffn_down_exps.weight"):
        return "down"
    return None


def output_path_for(out_dir: Path, mapped: str, shape: list[int], qtype: str) -> Path:
    proj = proj_from_exps(mapped)
    if proj is not None and len(shape) == 3:
        il = int(mapped.split(".")[1])
        n_expert, out_f, in_f = shape
        return out_dir / "experts" / f"blk{il}-{proj}-{n_expert}x{out_f}x{in_f}.q4k"
    ext = {"Q4_K": ".q4k", "F32": ".f32"}.get(qtype, ".bin")
    return out_dir / "tensors" / f"{safe_name(mapped)}{ext}"


def write_q4k_from_mlx(
    store: SafeTensorDir,
    stem: str,
    out_path: Path,
    bits: int,
    group_size: int,
    logical_shape: list[int],
    max_work_bytes: int,
) -> None:
    winfo, wraw = store.raw(stem + ".weight")
    sinfo, sraw = store.raw(stem + ".scales")
    binfo, braw = store.raw(stem + ".biases")
    q_words = read_numeric(winfo, wraw).reshape(-1, winfo.shape[-1])
    scales = read_numeric(sinfo, sraw).reshape(-1, sinfo.shape[-1])
    biases = read_numeric(binfo, braw).reshape(-1, binfo.shape[-1])
    in_f = logical_shape[-1]
    chunk = row_chunk_for(in_f, max_work_bytes)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        for start in range(0, q_words.shape[0], chunk):
            end = min(start + chunk, q_words.shape[0])
            rows = dequant_mlx_affine_rows(
                q_words[start:end], scales[start:end], biases[start:end], bits, group_size
            )
            f.write(quantize_q4k_rows(rows))


def write_f32_from_mlx(
    store: SafeTensorDir,
    stem: str,
    out_path: Path,
    bits: int,
    group_size: int,
    max_work_bytes: int,
) -> list[int]:
    winfo, wraw = store.raw(stem + ".weight")
    sinfo, sraw = store.raw(stem + ".scales")
    binfo, braw = store.raw(stem + ".biases")
    logical = mlx_logical_shape(winfo.shape, sinfo.shape, bits, group_size)
    q_words = read_numeric(winfo, wraw).reshape(-1, winfo.shape[-1])
    scales = read_numeric(sinfo, sraw).reshape(-1, sinfo.shape[-1])
    biases = read_numeric(binfo, braw).reshape(-1, binfo.shape[-1])
    chunk = row_chunk_for(logical[-1], max_work_bytes)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        for start in range(0, q_words.shape[0], chunk):
            end = min(start + chunk, q_words.shape[0])
            rows = dequant_mlx_affine_rows(
                q_words[start:end], scales[start:end], biases[start:end], bits, group_size
            )
            f.write(np.asarray(rows, dtype="<f4").tobytes())
    return logical


def copy_tensor(store: SafeTensorDir, name: str, out_path: Path) -> TensorInfo:
    info, raw = store.raw(name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        f.write(raw)
    return info


def inventory(model_dir: Path, out_tsv: Path) -> dict[str, Any]:
    store = SafeTensorDir(model_dir)
    rows = []
    try:
        for name in sorted(store.weight_map):
            info = store.info(name)
            rows.append((name, info.dtype, "x".join(map(str, info.shape)), str(info.nbytes), info.shard))
    finally:
        store.close()
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with out_tsv.open("w") as f:
        f.write("name\tdtype\tshape\tnbytes\tshard\n")
        for row in rows:
            f.write("\t".join(row) + "\n")
    counts: dict[str, int] = {}
    bytes_by_dtype: dict[str, int] = {}
    for _, dtype, _, nbytes, _ in rows:
        counts[dtype] = counts.get(dtype, 0) + 1
        bytes_by_dtype[dtype] = bytes_by_dtype.get(dtype, 0) + int(nbytes)
    return {"tensor_count": len(rows), "dtype_counts": counts, "bytes_by_dtype": bytes_by_dtype}


def transcode(args: argparse.Namespace) -> None:
    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tensors").mkdir(exist_ok=True)
    (out_dir / "experts").mkdir(exist_ok=True)
    config = json.loads((model_dir / "config.json").read_text())
    store = SafeTensorDir(model_dir)
    max_work = int(args.max_work_mb) << 20
    manifest: dict[str, Any] = {
        "format": "bw24-hy3-q4k-repack-v1",
        "source_repo": "pipenetwork/Hy3-REAP50-MLX-4bit",
        "source_dir": str(model_dir),
        "created_utc": _dt.datetime.now(_dt.UTC).isoformat(),
        "quality": "unverified — pending target-rig gates",
        "source_metadata": store.metadata,
        "config_sha256": sha256_file(model_dir / "config.json"),
        "files": {},
        "tensors": {},
        "experts": [],
    }
    for side in ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json"]:
        p = model_dir / side
        if p.exists():
            manifest["files"][side] = {"bytes": p.stat().st_size, "sha256": sha256_file(p)}

    consumed: set[str] = set()
    try:
        for name in sorted(store.weight_map):
            if args.limit_tensors and len(manifest["tensors"]) >= args.limit_tensors:
                break
            if name in consumed or name.endswith(".scales") or name.endswith(".biases"):
                continue
            if name.endswith(".weight") and name[:-7] + ".scales" in store.weight_map and name[:-7] + ".biases" in store.weight_map:
                stem = name[:-7]
                params = mlx_quant_params(config, stem)
                mode = params.get("mode", "affine")
                bits = int(params["bits"])
                group_size = int(params["group_size"])
                if mode != "affine":
                    raise ValueError(f"{stem}: expected MLX affine, got {mode}")
                winfo = store.info(name)
                sinfo = store.info(stem + ".scales")
                logical = mlx_logical_shape(winfo.shape, sinfo.shape, bits, group_size)
                mapped = map_hy3_name(name, logical)
                if bits == 4 and logical[-1] % Q4K_BLOCK == 0:
                    qtype = "Q4_K"
                    out_path = output_path_for(out_dir, mapped, logical, qtype)
                    expected = q4k_expected_size(logical)
                    if not (args.resume and out_path.exists() and out_path.stat().st_size == expected):
                        write_q4k_from_mlx(store, stem, out_path, bits, group_size, logical, max_work)
                    if out_path.stat().st_size != expected:
                        raise RuntimeError(f"{out_path} size {out_path.stat().st_size} != {expected}")
                    row_bytes = (logical[-1] // Q4K_BLOCK) * Q4K_BYTES
                    rec = {
                        "source": name,
                        "mapped": mapped,
                        "file": str(out_path.relative_to(out_dir)),
                        "qtype": qtype,
                        "source_quant": {"mode": mode, "bits": bits, "group_size": group_size},
                        "shape": logical,
                        "ne": list(reversed(logical)),
                        "row_bytes": row_bytes,
                        "bytes": expected,
                    }
                    proj = proj_from_exps(mapped)
                    if proj is not None and len(logical) == 3:
                        n_expert, out_f, in_f = logical
                        expert_stride = out_f * row_bytes
                        rec["expert_stride"] = expert_stride
                        exp = {
                            "mapped": mapped,
                            "layer": int(mapped.split(".")[1]),
                            "proj": proj,
                            "file": rec["file"],
                            "qtype": qtype,
                            "n_expert": n_expert,
                            "out_f": out_f,
                            "in_f": in_f,
                            "row_bytes": row_bytes,
                            "expert_stride": expert_stride,
                            "expert_offsets": [
                                {"expert": i, "offset": i * expert_stride, "size": expert_stride}
                                for i in range(n_expert)
                            ],
                        }
                        manifest["experts"].append(exp)
                else:
                    qtype = "F32"
                    mapped = map_hy3_name(name, logical)
                    out_path = output_path_for(out_dir, mapped, logical, qtype)
                    expected = int(np.prod(logical, dtype=np.int64)) * 4
                    if not (args.resume and out_path.exists() and out_path.stat().st_size == expected):
                        write_f32_from_mlx(store, stem, out_path, bits, group_size, max_work)
                    rec = {
                        "source": name,
                        "mapped": mapped,
                        "file": str(out_path.relative_to(out_dir)),
                        "qtype": qtype,
                        "source_quant": {"mode": mode, "bits": bits, "group_size": group_size},
                        "shape": logical,
                        "ne": list(reversed(logical)),
                        "bytes": expected,
                        "note": "affine non-4bit dequantized to F32 for routing/selection safety",
                    }
                manifest["tensors"][mapped] = rec
                consumed.update({name, stem + ".scales", stem + ".biases"})
                print(f"{rec['qtype']:5s} {name} -> {rec['file']} {rec['bytes'] / 1e6:.1f} MB", flush=True)
                store.drop_cached_shards()
                trim_process_memory()
                continue

            info = store.info(name)
            mapped = map_hy3_name(name)
            out_path = output_path_for(out_dir, mapped, info.shape, info.dtype)
            if not (args.resume and out_path.exists() and out_path.stat().st_size == info.nbytes):
                copy_tensor(store, name, out_path)
            manifest["tensors"][mapped] = {
                "source": name,
                "mapped": mapped,
                "file": str(out_path.relative_to(out_dir)),
                "qtype": info.dtype,
                "shape": info.shape,
                "ne": list(reversed(info.shape)),
                "bytes": info.nbytes,
                "note": "copied source tensor bytes",
            }
            print(f"{info.dtype:5s} {name} -> {out_path.relative_to(out_dir)} {info.nbytes / 1e6:.3f} MB", flush=True)
            store.drop_cached_shards()
            trim_process_memory()
    finally:
        store.close()

    manifest_path = out_dir / "manifest.json"
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    tmp.replace(manifest_path)
    print(f"wrote {manifest_path}")


def encode_mlx_affine(arr: np.ndarray, bits: int = 4, group_size: int = 64) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.shape[-1] % group_size != 0:
        raise ValueError(arr.shape)
    rows = arr.reshape(-1, arr.shape[-1])
    groups = arr.shape[-1] // group_size
    x = rows.reshape(rows.shape[0], groups, group_size)
    beta = x.min(axis=2)
    alpha = x.max(axis=2)
    scale = (alpha - beta) / float((1 << bits) - 1)
    scale = np.where(scale > 1e-30, scale, 1.0).astype(np.float32)
    q = np.rint((x - beta[:, :, None]) / scale[:, :, None]).clip(0, (1 << bits) - 1).astype(np.uint32)
    pack = 32 // bits
    words_per_group = group_size // pack
    q = q.reshape(rows.shape[0], groups, words_per_group, pack)
    shifts = (np.arange(pack, dtype=np.uint32) * bits).reshape(1, 1, 1, pack)
    words = np.bitwise_or.reduce(q << shifts, axis=3)
    packed = words.reshape(*arr.shape[:-1], groups * words_per_group).astype("<u4")
    return packed, scale.reshape(*arr.shape[:-1], groups), beta.reshape(*arr.shape[:-1], groups)


def write_safetensors(path: Path, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    header: dict[str, Any] = {}
    off = 0
    payload = bytearray()
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + len(data)]}
        payload.extend(data)
        off += len(data)
    h = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(h)) + h + payload)


def run_synthetic_tests() -> None:
    rng = np.random.default_rng(123)
    arr = rng.normal(0, 0.3, size=(7, 512)).astype(np.float32)
    arr[0, :64] += np.linspace(-1.0, 1.0, 64, dtype=np.float32)
    w, s, b = encode_mlx_affine(arr, bits=4, group_size=64)
    got = dequant_mlx_affine_rows(w.reshape(7, -1), s.reshape(7, -1), b.reshape(7, -1), 4, 64)
    mlx_err = float(np.max(np.abs(got - arr)))
    mlx_bound = float(np.max(s) * 0.5) + 1e-6
    if mlx_err > mlx_bound:
        raise AssertionError(f"MLX affine dequant error {mlx_err} exceeds half-step bound {mlx_bound}")
    raw = quantize_q4k_rows(got)
    back = q4k_dequant_rows(raw, 512)
    ref = q4k_dequant_rows_reference(raw, 512)
    if not np.array_equal(back, ref):
        raise AssertionError("vectorized Q4_K dequant differs from scalar ggml-layout reference")
    max_abs = float(np.max(np.abs(back - got)))
    # The simple Q4_K quantizer is allowed its own low-bit rounding, but it should remain bounded.
    if max_abs > 0.20:
        raise AssertionError(f"Q4_K synthetic roundtrip too loose: {max_abs}")
    raw2 = quantize_q4k_rows(got)
    if raw != raw2:
        raise AssertionError("Q4_K quantization is not byte-deterministic")

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        shard = root / "model-00001-of-00001.safetensors"
        write_safetensors(
            shard,
            {
                "x.weight": ("U32", list(w.shape), w.tobytes()),
                "x.scales": ("F32", list(s.shape), s.astype("<f4").tobytes()),
                "x.biases": ("F32", list(b.shape), b.astype("<f4").tobytes()),
            },
        )
        idx = {"metadata": {"total_size": shard.stat().st_size}, "weight_map": {
            "x.weight": shard.name, "x.scales": shard.name, "x.biases": shard.name
        }}
        (root / "model.safetensors.index.json").write_text(json.dumps(idx))
        store = SafeTensorDir(root)
        try:
            with tempfile.NamedTemporaryFile() as out:
                write_q4k_from_mlx(store, "x", Path(out.name), 4, 64, [7, 512], 32 << 20)
                disk = Path(out.name).read_bytes()
                if disk != raw:
                    raise AssertionError("safetensors streamed transcode differs from in-memory transcode")
        finally:
            store.close()

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        out_dir = root / "out"
        logical = rng.normal(0, 0.2, size=(2, 32, 256)).astype(np.float32)
        w2, s2, b2 = encode_mlx_affine(logical, bits=4, group_size=64)
        name = "model.layers.1.mlp.switch_mlp.gate_proj"
        shard = root / "model-00001-of-00001.safetensors"
        write_safetensors(
            shard,
            {
                f"{name}.weight": ("U32", list(w2.shape), w2.tobytes()),
                f"{name}.scales": ("F32", list(s2.shape), s2.astype("<f4").tobytes()),
                f"{name}.biases": ("F32", list(b2.shape), b2.astype("<f4").tobytes()),
            },
        )
        idx = {"metadata": {"total_size": shard.stat().st_size}, "weight_map": {
            f"{name}.weight": shard.name, f"{name}.scales": shard.name, f"{name}.biases": shard.name
        }}
        (root / "model.safetensors.index.json").write_text(json.dumps(idx))
        (root / "config.json").write_text(json.dumps({
            "quantization_config": {"group_size": 64, "bits": 4, "mode": "affine"}
        }))
        transcode(argparse.Namespace(
            model_dir=str(root),
            out_dir=str(out_dir),
            max_work_mb=32,
            resume=True,
            limit_tensors=0,
        ))
        manifest = json.loads((out_dir / "manifest.json").read_text())
        rec = manifest["tensors"]["blk.1.ffn_gate_exps.weight"]
        exp = manifest["experts"][0]
        if rec["qtype"] != "Q4_K" or rec["shape"] != [2, 32, 256]:
            raise AssertionError("synthetic transcode manifest tensor metadata mismatch")
        if exp["expert_stride"] != 32 * 144 or exp["expert_offsets"][1]["offset"] != 32 * 144:
            raise AssertionError("synthetic expert offset manifest mismatch")
        if (out_dir / rec["file"]).stat().st_size != q4k_expected_size([2, 32, 256]):
            raise AssertionError("synthetic transcode Q4_K file size mismatch")
    print("synthetic tests: PASS")


def run_real_sample(model_dir: Path, limit: int) -> None:
    config = json.loads((model_dir / "config.json").read_text())
    store = SafeTensorDir(model_dir)
    checked = 0
    try:
        for name in sorted(store.weight_map):
            if checked >= limit:
                break
            if not name.endswith(".weight"):
                continue
            stem = name[:-7]
            if stem + ".scales" not in store.weight_map or stem + ".biases" not in store.weight_map:
                continue
            params = mlx_quant_params(config, stem)
            if params.get("mode") != "affine" or int(params["bits"]) != 4:
                continue
            try:
                winfo, wraw = store.raw(name)
                sinfo, sraw = store.raw(stem + ".scales")
                binfo, braw = store.raw(stem + ".biases")
            except FileNotFoundError:
                continue
            logical = mlx_logical_shape(winfo.shape, sinfo.shape, 4, int(params["group_size"]))
            if logical[-1] % Q4K_BLOCK != 0:
                continue
            rows = min(3, int(np.prod(logical[:-1], dtype=np.int64)))
            q_words = read_numeric(winfo, wraw).reshape(-1, winfo.shape[-1])[:rows].copy()
            scales = read_numeric(sinfo, sraw).reshape(-1, sinfo.shape[-1])[:rows].copy()
            biases = read_numeric(binfo, braw).reshape(-1, binfo.shape[-1])[:rows].copy()
            del wraw, sraw, braw
            f = dequant_mlx_affine_rows(q_words, scales, biases, 4, int(params["group_size"]))
            raw = quantize_q4k_rows(f)
            back = q4k_dequant_rows(raw, logical[-1])
            if not np.isfinite(back).all():
                raise AssertionError(f"non-finite Q4K sample for {name}")
            print(f"real sample {checked + 1}: {name} rows={rows} max_abs={np.max(np.abs(back - f)):.6f}")
            checked += 1
    finally:
        store.close()
    if checked == 0:
        raise SystemExit("no complete real 4-bit tensors available to sample yet")
    print("real sampled tests: PASS")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    inv = sub.add_parser("inventory")
    inv.add_argument("model_dir")
    inv.add_argument("--out", default="research/hy3-reap50-tensor-inventory.tsv")
    tr = sub.add_parser("transcode")
    tr.add_argument("model_dir")
    tr.add_argument("out_dir")
    tr.add_argument("--max-work-mb", type=int, default=512)
    tr.add_argument("--resume", action="store_true", default=True)
    tr.add_argument("--limit-tensors", type=int, default=0, help="test/debug: stop after N output tensors")
    tst = sub.add_parser("test")
    tst.add_argument("--real-model-dir")
    tst.add_argument("--real-limit", type=int, default=3)
    args = p.parse_args(argv)
    if args.cmd == "inventory":
        summary = inventory(Path(args.model_dir), Path(args.out))
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.cmd == "transcode":
        transcode(args)
        return 0
    if args.cmd == "test":
        run_synthetic_tests()
        if args.real_model_dir:
            run_real_sample(Path(args.real_model_dir), args.real_limit)
        return 0
    raise AssertionError(args.cmd)


if __name__ == "__main__":
    raise SystemExit(main())
