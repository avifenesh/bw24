#!/usr/bin/env python3
"""Build a bw24 expert overlay with exact per-expert mixed GGUF encodings.

The quantization source may be an indexed BF16/F16/F32 Hugging Face checkpoint or the stacked
MLX-affine Hy3 checkpoint. Dense/router tensors resolve from --fallback-dir, which may itself be
a complete bw24 manifest repack. Experts declared pruned in the plan are omitted and masked by the
runtime before top-k routing. No model or GPU is loaded by this CPU-only preparation tool.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import struct
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import numpy as np

from hy3_mlx_to_q4k import (
    SafeTensorDir,
    bf16_to_f32,
    dequant_mlx_affine_rows,
    mlx_logical_shape,
    mlx_quant_params,
    read_numeric,
    row_chunk_for,
    sha256_file,
    trim_process_memory,
)
from ggml_quant_bridge import (
    EXTERNAL_QTYPES,
    GgmlQuantBridge,
    load_importance_sidecar,
)


PLAN_FORMAT = "bw24-expert-tier-plan-v2"
OVERLAY_FORMAT = "bw24-expert-overlay-v2"
COMPLETION_RECEIPT_FORMAT = "bw24-expert-overlay-file-completion-v1"
PROJECTIONS = ("gate", "up", "down")
QTYPES = {
    "Q8_0": (32, 34, ".q8"),
    "Q2_K": (256, 84, ".q2k"),
    "Q3_K": (256, 110, ".q3k"),
    "NVFP4": (64, 36, ".nvfp4"),
    "IQ4_XS": (256, 136, ".iq4xs"),
    "Q4_K": (256, 144, ".q4k"),
}


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _layers_from_model(model: dict[str, Any]) -> list[int]:
    layers = model.get("moe_layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError("plan.model.moe_layers must be a non-empty list")
    if len(layers) == 2 and model.get("moe_layers_are_range", False):
        return list(range(int(layers[0]), int(layers[1]) + 1))
    return [int(x) for x in layers]


def _parse_layer_subset(raw: str | None, available: list[int]) -> list[int]:
    if not raw:
        return available
    if "-" in raw:
        lo, hi = (int(value) for value in raw.split("-", 1))
        requested = list(range(lo, hi + 1))
    else:
        requested = [int(value) for value in raw.split(",") if value]
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("--layers must select distinct layers")
    if not set(requested).issubset(available):
        raise ValueError(f"--layers contains values outside the plan: {requested}")
    return requested


def load_assignments(
    path: Path,
) -> tuple[dict[str, Any], dict[tuple[int, int, str], str], dict[int, set[int]]]:
    plan = json.loads(path.read_text())
    if plan.get("format") != PLAN_FORMAT:
        raise ValueError(f"{path}: format must be {PLAN_FORMAT!r}")
    model = plan.get("model")
    if not isinstance(model, dict):
        raise ValueError("plan.model is required")
    n_expert = int(model["expert_count"])
    layers = _layers_from_model(model)
    if n_expert <= 0 or any(layer < 0 for layer in layers):
        raise ValueError("expert_count must be positive and layer ids non-negative")

    pruned: dict[int, set[int]] = {layer: set() for layer in layers}
    for layer_s, ids in plan.get("pruned_experts", {}).items():
        layer = int(layer_s)
        if layer not in pruned:
            raise ValueError(f"pruned_experts contains non-MoE layer {layer}")
        pruned[layer] = {int(x) for x in ids}
        if any(ex < 0 or ex >= n_expert for ex in pruned[layer]):
            raise ValueError(f"pruned_experts.{layer} contains an out-of-range id")

    expanded: dict[tuple[int, int, str], str] = {}
    for i, group in enumerate(plan.get("assignments", [])):
        layer = int(group["layer"])
        experts = [int(x) for x in group.get("experts", [])]
        projections = group.get("projections", list(PROJECTIONS))
        qtype = group.get("qtype")
        if layer not in pruned or not experts:
            raise ValueError(f"assignment {i}: invalid layer or empty experts")
        if qtype not in QTYPES:
            raise ValueError(f"assignment {i}: qtype must be one of {sorted(QTYPES)}, got {qtype}")
        if any(ex < 0 or ex >= n_expert or ex in pruned[layer] for ex in experts):
            raise ValueError(f"assignment {i}: expert is out of range or declared pruned")
        if not projections or any(proj not in PROJECTIONS for proj in projections):
            raise ValueError(f"assignment {i}: projections must be drawn from {PROJECTIONS}")
        for expert in experts:
            for proj in projections:
                key = (layer, expert, proj)
                if key in expanded:
                    raise ValueError(f"assignment {i}: duplicate selection {key}")
                expanded[key] = qtype

    expected = {
        (layer, expert, proj)
        for layer in layers
        for expert in range(n_expert)
        if expert not in pruned[layer]
        for proj in PROJECTIONS
    }
    missing = expected - expanded.keys()
    extra = expanded.keys() - expected
    if missing or extra:
        sample = sorted(missing)[:5]
        raise ValueError(
            f"v2 plans must encode every retained expert projection: missing={len(missing)} "
            f"extra={len(extra)} sample_missing={sample}"
        )
    return plan, expanded, pruned


def _round(x: np.ndarray) -> np.ndarray:
    return np.rint(x)


def quantize_q8_0_rows(rows: np.ndarray) -> bytes:
    rows = np.asarray(rows, dtype=np.float32)
    r, in_f = rows.shape
    if in_f % 32:
        raise ValueError(f"Q8_0 requires in_features divisible by 32, got {in_f}")
    blocks = rows.reshape(r, in_f // 32, 32)
    amax = np.max(np.abs(blocks), axis=2)
    d = amax / 127.0
    d16 = d.astype("<f2")
    q = np.zeros_like(blocks, dtype=np.float32)
    np.divide(blocks, d[..., None], out=q, where=d[..., None] > 0)
    q = _round(q).clip(-127, 127).astype(np.int8)
    packed = np.empty((r, in_f // 32, 34), dtype=np.uint8)
    packed[:, :, :2] = d16.reshape(r, in_f // 32, 1).view(np.uint8).reshape(r, in_f // 32, 2)
    packed[:, :, 2:34] = q.view(np.uint8)
    return packed.reshape(-1).tobytes()


def quantize_q2k_rows(rows: np.ndarray) -> bytes:
    rows = np.asarray(rows, dtype=np.float32)
    r, in_f = rows.shape
    if in_f % 256:
        raise ValueError(f"Q2_K requires in_features divisible by 256, got {in_f}")
    nb = in_f // 256
    x = rows.reshape(r, nb, 16, 16)
    mn = x.min(axis=3)
    mx = x.max(axis=3)
    offset = np.maximum(-mn, 0.0)
    scale = (mx + offset) / 3.0
    scale = np.where(scale > 1e-30, scale, 0.0)
    d = scale.max(axis=2) / 15.0
    dmin = offset.max(axis=2) / 15.0
    scale_units = np.zeros_like(scale, dtype=np.float32)
    min_units = np.zeros_like(offset, dtype=np.float32)
    np.divide(scale, d[..., None], out=scale_units, where=d[..., None] > 0)
    np.divide(offset, dmin[..., None], out=min_units, where=dmin[..., None] > 0)
    sc = _round(scale_units).clip(0, 15).astype(np.uint8)
    mi = _round(min_units).clip(0, 15).astype(np.uint8)
    d16 = d.astype("<f2")
    dm16 = dmin.astype("<f2")
    se = d16.astype(np.float32)[..., None] * sc
    me = dm16.astype(np.float32)[..., None] * mi
    q = np.zeros_like(x, dtype=np.float32)
    np.divide(x + me[..., None], se[..., None], out=q, where=se[..., None] > 0)
    q = _round(q).clip(0, 3).astype(np.uint8).reshape(r, nb, 256)
    qs = np.empty((r, nb, 64), dtype=np.uint8)
    for half in range(2):
        base = half * 128
        for lane in range(4):
            qs[:, :, half * 32 : (half + 1) * 32] = (
                qs[:, :, half * 32 : (half + 1) * 32]
                if lane
                else np.zeros((r, nb, 32), dtype=np.uint8)
            )
            qs[:, :, half * 32 : (half + 1) * 32] |= q[:, :, base + lane * 32 : base + (lane + 1) * 32] << (2 * lane)
    packed = np.empty((r, nb, 84), dtype=np.uint8)
    packed[:, :, :16] = sc | (mi << 4)
    packed[:, :, 16:80] = qs
    packed[:, :, 80:82] = d16.reshape(r, nb, 1).view(np.uint8).reshape(r, nb, 2)
    packed[:, :, 82:84] = dm16.reshape(r, nb, 1).view(np.uint8).reshape(r, nb, 2)
    return packed.reshape(-1).tobytes()


def quantize_q3k_rows(rows: np.ndarray) -> bytes:
    rows = np.asarray(rows, dtype=np.float32)
    r, in_f = rows.shape
    if in_f % 256:
        raise ValueError(f"Q3_K requires in_features divisible by 256, got {in_f}")
    nb = in_f // 256
    x = rows.reshape(r, nb, 16, 16)
    max_idx = np.abs(x).argmax(axis=3)
    max_val = np.take_along_axis(x, max_idx[..., None], axis=3)[..., 0]
    group_scale = -max_val / 4.0
    winner = np.abs(group_scale).argmax(axis=2)
    max_scale = np.take_along_axis(group_scale, winner[..., None], axis=2)[..., 0]
    iscale = np.where(np.abs(max_scale) > 1e-30, -32.0 / max_scale, 0.0)
    enc = _round(iscale[..., None] * group_scale).clip(-32, 31).astype(np.int16) + 32
    d = np.where(iscale != 0, 1.0 / iscale, 0.0)
    d16 = d.astype("<f2")
    se = d16.astype(np.float32)[..., None] * (enc - 32)
    q = np.zeros_like(x, dtype=np.float32)
    np.divide(x, se[..., None], out=q, where=np.abs(se[..., None]) > 0)
    q = _round(q).clip(-4, 3).astype(np.int8)
    codes = (q + 4).astype(np.uint8).reshape(r, nb, 256)

    hmask = np.zeros((r, nb, 32), dtype=np.uint8)
    low = codes.copy()
    for j in range(256):
        high = low[:, :, j] > 3
        hmask[:, :, j % 32] |= high.astype(np.uint8) << (j // 32)
        low[:, :, j] -= high.astype(np.uint8) * 4
    qs = np.zeros((r, nb, 64), dtype=np.uint8)
    for half in range(2):
        base = half * 128
        for lane in range(4):
            qs[:, :, half * 32 : (half + 1) * 32] |= (
                low[:, :, base + lane * 32 : base + (lane + 1) * 32] << (2 * lane)
            )
    scales = np.zeros((r, nb, 12), dtype=np.uint8)
    enc8 = enc.astype(np.uint8)
    scales[:, :, :8] = enc8[:, :, :8] & 0x0f
    scales[:, :, :8] |= (enc8[:, :, 8:16] & 0x0f) << 4
    for j in range(16):
        scales[:, :, 8 + j % 4] |= ((enc8[:, :, j] >> 4) & 3) << (2 * (j // 4))
    packed = np.empty((r, nb, 110), dtype=np.uint8)
    packed[:, :, :32] = hmask
    packed[:, :, 32:96] = qs
    packed[:, :, 96:108] = scales
    packed[:, :, 108:110] = d16.reshape(r, nb, 1).view(np.uint8).reshape(r, nb, 2)
    return packed.reshape(-1).tobytes()


def _ue4m3_table() -> np.ndarray:
    values = np.zeros(127, dtype=np.float32)
    for code in range(1, 127):
        exp = (code >> 3) & 0x0f
        man = code & 7
        values[code] = man * 2.0**-9 if exp == 0 else (1.0 + man / 8.0) * 2.0 ** (exp - 7)
    return values


UE4M3 = _ue4m3_table()
E2M1 = np.asarray([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def quantize_nvfp4_rows(rows: np.ndarray) -> bytes:
    rows = np.asarray(rows, dtype=np.float32)
    r, in_f = rows.shape
    if in_f % 64:
        raise ValueError(f"NVFP4 requires in_features divisible by 64, got {in_f}")
    nb = in_f // 64
    x = rows.reshape(r, nb, 4, 16)
    target = np.max(np.abs(x), axis=3) / 6.0
    hi = np.searchsorted(UE4M3, target, side="left").clip(0, 126)
    lo = np.maximum(hi - 1, 0)
    choose_hi = np.abs(UE4M3[hi] - target) < np.abs(target - UE4M3[lo])
    scode = np.where(choose_hi, hi, lo).astype(np.uint8)
    scale = UE4M3[scode]
    norm = np.zeros_like(x)
    np.divide(x, scale[..., None], out=norm, where=scale[..., None] > 0)
    mag = np.abs(norm)
    code = np.abs(mag[..., None] - E2M1).argmin(axis=4).astype(np.uint8)
    code = np.where((code != 0) & (norm < 0), code | 8, code).astype(np.uint8)
    qs = code[..., :8] | (code[..., 8:] << 4)
    packed = np.empty((r, nb, 36), dtype=np.uint8)
    packed[:, :, :4] = scode
    packed[:, :, 4:36] = qs.reshape(r, nb, 32)
    return packed.reshape(-1).tobytes()


QUANTIZERS = {
    "Q8_0": quantize_q8_0_rows,
    "Q2_K": quantize_q2k_rows,
    "Q3_K": quantize_q3k_rows,
    "NVFP4": quantize_nvfp4_rows,
}


def _external_quantization_context(
    plan: dict[str, Any], assignments: dict[tuple[int, int, str], str], args: argparse.Namespace,
) -> tuple[GgmlQuantBridge | None, dict[str, dict[str, Any]]]:
    external = sorted(set(assignments.values()) & set(EXTERNAL_QTYPES))
    if not external:
        return None, {}
    lib = getattr(args, "ggml_lib", None)
    lib_sha = getattr(args, "ggml_lib_sha256", None)
    commit = getattr(args, "ggml_source_commit", None)
    if not all((lib, lib_sha, commit)):
        raise ValueError(f"{external} require --ggml-lib, its SHA-256, and source commit")
    sensitivity_receipt = plan.get("calibration", {}).get("quant_sensitivity")
    if not isinstance(sensitivity_receipt, dict):
        raise ValueError(f"{external} require a plan-bound quant sensitivity map")
    sensitivity_path = Path(sensitivity_receipt["path"])
    if sha256_file(sensitivity_path) != sensitivity_receipt["sha256"]:
        raise ValueError("plan-bound quant sensitivity map hash changed")
    sensitivity = json.loads(sensitivity_path.read_text())
    sidecars = sensitivity.get("importance_sidecars", {})
    layers = {str(layer) for layer in _layers_from_model(plan["model"])}
    if set(sidecars) != layers:
        raise ValueError(f"{external} require one private importance sidecar per layer")
    provenance = sensitivity.get("measurement", {}).get("exact_quantizer_implementation", {})
    for qtype in external:
        record = provenance.get(qtype, {}) if isinstance(provenance, dict) else {}
        if (
            record.get("library_sha256") != lib_sha
            or record.get("llama_cpp_commit") != commit
        ):
            raise ValueError(f"{qtype} bridge differs from sensitivity provenance")
    return GgmlQuantBridge(Path(lib), lib_sha, commit), sidecars


def _dequant_q8_0(raw: bytes, in_f: int) -> np.ndarray:
    b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, in_f // 32, 34)
    d = b[:, :, :2].copy().reshape(-1, in_f // 32, 2).view("<f2").reshape(-1, in_f // 32)
    q = b[:, :, 2:34].view(np.int8).astype(np.float32)
    return (q * d.astype(np.float32)[..., None]).reshape(-1, in_f)


def _dequant_q2k(raw: bytes, in_f: int) -> np.ndarray:
    row_bytes = in_f // 256 * 84
    b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, in_f // 256, 84)
    out = np.empty((len(b), in_f), dtype=np.float32)
    for row in range(len(b)):
        for ib in range(in_f // 256):
            block = b[row, ib]
            d = block[80:82].copy().view("<f2")[0].astype(np.float32)
            dm = block[82:84].copy().view("<f2")[0].astype(np.float32)
            for j in range(256):
                sc = block[j // 16]
                within = j % 128
                q = (block[16 + (j // 128) * 32 + within % 32] >> (2 * (within // 32))) & 3
                out[row, ib * 256 + j] = d * (sc & 15) * q - dm * (sc >> 4)
    assert len(raw) == len(out) * row_bytes
    return out


def _dequant_q3k(raw: bytes, in_f: int) -> np.ndarray:
    b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, in_f // 256, 110)
    out = np.empty((len(b), in_f), dtype=np.float32)
    for row in range(len(b)):
        for ib in range(in_f // 256):
            block = b[row, ib]
            enc = np.zeros(16, dtype=np.int16)
            for j in range(16):
                lo = (block[96 + j] & 15) if j < 8 else (block[96 + j - 8] >> 4)
                hi = (block[104 + j % 4] >> (2 * (j // 4))) & 3
                enc[j] = lo | (hi << 4)
            d = block[108:110].copy().view("<f2")[0].astype(np.float32)
            for j in range(256):
                within = j % 128
                low = (block[32 + (j // 128) * 32 + within % 32] >> (2 * (within // 32))) & 3
                high = (block[j % 32] >> (j // 32)) & 1
                q = int(low) - (0 if high else 4)
                out[row, ib * 256 + j] = d * (enc[j // 16] - 32) * q
    return out


def _dequant_nvfp4(raw: bytes, in_f: int) -> np.ndarray:
    b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, in_f // 64, 36)
    out = np.empty((len(b), in_f), dtype=np.float32)
    for row in range(len(b)):
        for ib in range(in_f // 64):
            block = b[row, ib]
            for sub in range(4):
                scale = UE4M3[block[sub]]
                for j in range(16):
                    byte = block[4 + sub * 8 + j % 8]
                    code = (byte & 15) if j < 8 else (byte >> 4)
                    sign = -1.0 if code & 8 else 1.0
                    out[row, ib * 64 + sub * 16 + j] = sign * E2M1[code & 7] * scale
    return out


def ordinary_expert_name(store: SafeTensorDir, layer: int, expert: int, proj: str) -> str | None:
    w = {"gate": "w1", "down": "w2", "up": "w3"}[proj]
    candidates = [
        f"model.layers.{layer}.mlp.experts.{expert}.{proj}_proj.weight",
        f"model.layers.{layer}.block_sparse_moe.experts.{expert}.{w}.weight",
    ]
    candidates += [f"model.language_model.{name[len('model.') :]}" for name in candidates]
    candidates += [f"language_model.{name}" for name in candidates]
    return next((name for name in candidates if name in store.weight_map), None)


def stacked_mlx_stem(store: SafeTensorDir, layer: int, proj: str) -> str | None:
    candidates = [
        f"model.layers.{layer}.mlp.switch_mlp.{proj}_proj",
        f"model.language_model.layers.{layer}.mlp.switch_mlp.{proj}_proj",
    ]
    return next(
        (stem for stem in candidates if all(stem + suffix in store.weight_map for suffix in (".weight", ".scales", ".biases"))),
        None,
    )


class ProjectionSource:
    def __init__(
        self, store: SafeTensorDir, config: dict[str, Any], layer: int, proj: str,
        active: list[int], max_work: int,
    ):
        self.store = store
        self.layer = layer
        self.proj = proj
        self.active = active
        self.max_work = max_work
        self.stem = stacked_mlx_stem(store, layer, proj)
        self.stacked: tuple[np.ndarray, np.ndarray, np.ndarray, int, int] | None = None
        if self.stem is not None:
            params = mlx_quant_params(config, self.stem)
            if params.get("mode", "affine") != "affine":
                raise ValueError(f"{self.stem}: only MLX affine is supported")
            winfo, wraw = store.raw(self.stem + ".weight")
            sinfo, sraw = store.raw(self.stem + ".scales")
            binfo, braw = store.raw(self.stem + ".biases")
            logical = mlx_logical_shape(winfo.shape, sinfo.shape, int(params["bits"]), int(params["group_size"]))
            if len(logical) != 3:
                raise ValueError(f"{self.stem}: expected [experts,out,in], got {logical}")
            self.n_expert, self.out_f, self.in_f = logical
            self.stacked = (
                read_numeric(winfo, wraw).reshape(winfo.shape),
                read_numeric(sinfo, sraw).reshape(sinfo.shape),
                read_numeric(binfo, braw).reshape(binfo.shape),
                int(params["bits"]),
                int(params["group_size"]),
            )
        else:
            first = ordinary_expert_name(store, layer, active[0], proj)
            if first is None:
                raise KeyError(f"no expert source for layer={layer} projection={proj}")
            info = store.info(first)
            if len(info.shape) != 2 or info.dtype not in {"BF16", "F16", "F32"}:
                raise ValueError(f"{first}: expected 2D BF16/F16/F32, got {info.dtype} {info.shape}")
            self.n_expert = max(active) + 1
            self.out_f, self.in_f = info.shape

    def rows(self, expert: int) -> Iterator[np.ndarray]:
        chunk = row_chunk_for(self.in_f, self.max_work)
        if self.stacked is not None:
            q, scales, biases, bits, group_size = self.stacked
            for start in range(0, self.out_f, chunk):
                end = min(start + chunk, self.out_f)
                yield dequant_mlx_affine_rows(
                    q[expert, start:end], scales[expert, start:end], biases[expert, start:end],
                    bits, group_size,
                )
            return
        name = ordinary_expert_name(self.store, self.layer, expert, self.proj)
        if name is None:
            raise KeyError(f"missing retained expert {self.layer}/{expert}/{self.proj}")
        info, raw = self.store.raw(name)
        if info.shape != [self.out_f, self.in_f]:
            raise ValueError(f"{name}: shape changed to {info.shape}")
        if info.dtype == "BF16":
            arr = bf16_to_f32(raw).reshape(info.shape)
        elif info.dtype == "F16":
            arr = np.frombuffer(raw, dtype="<f2").astype(np.float32).reshape(info.shape)
        else:
            arr = np.frombuffer(raw, dtype="<f4").reshape(info.shape)
        for start in range(0, self.out_f, chunk):
            yield np.asarray(arr[start : start + chunk], dtype=np.float32)

    def description(self, expert: int) -> str:
        if self.stem is not None:
            return f"{self.stem}.weight[{expert}]"
        return ordinary_expert_name(self.store, self.layer, expert, self.proj) or "missing"


def _fingerprint(path: Path, names: tuple[str, ...]) -> dict[str, Any]:
    out = {}
    for name in names:
        p = path / name
        if p.exists():
            out[name] = {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
    return out


def _completion_receipt_path(out_path: Path) -> Path:
    return out_path.with_name(out_path.name + ".complete.json")


def _completion_receipt_matches(path: Path, expected: dict[str, Any]) -> bool:
    try:
        return json.loads(path.read_text()) == expected
    except (OSError, json.JSONDecodeError):
        return False


def _write_completion_receipt(path: Path, receipt: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def _install_tensor_overrides(
    out_dir: Path, override_path: Path | None, manifest: dict[str, Any]
) -> None:
    override_dir = out_dir / "overrides"
    if override_path is None:
        shutil.rmtree(override_dir, ignore_errors=True)
        return
    receipt = json.loads(override_path.read_text())
    if receipt.get("format") != "bw24-tensor-overrides-v1":
        raise ValueError(f"{override_path}: unsupported tensor override format")
    blob = receipt.get("blob")
    tensors = receipt.get("tensors")
    if not isinstance(blob, dict) or not isinstance(tensors, dict) or not tensors:
        raise ValueError(f"{override_path}: missing blob or tensors")
    blob_path = Path(blob["path"])
    if blob_path.stat().st_size != int(blob["bytes"]):
        raise ValueError(f"{override_path}: override blob size changed")
    if sha256_file(blob_path) != blob["sha256"]:
        raise ValueError(f"{override_path}: override blob hash changed")
    override_dir.mkdir(parents=True, exist_ok=True)
    rel = Path("overrides") / f"{blob['sha256']}.bin"
    installed = out_dir / rel
    shutil.copyfile(blob_path, installed)
    allowed_suffixes = (".ffn_gate_inp.weight", ".exp_probs_b.bias")
    for name, record in sorted(tensors.items()):
        if not name.startswith("blk.") or not name.endswith(allowed_suffixes):
            raise ValueError(f"{override_path}: disallowed tensor override {name}")
        if name in manifest["tensors"]:
            raise ValueError(f"{override_path}: tensor override collision {name}")
        if record.get("qtype") != "F32":
            raise ValueError(f"{override_path}: {name} must remain F32")
        ne = record.get("ne")
        if not isinstance(ne, list) or not ne or any(int(value) <= 0 for value in ne):
            raise ValueError(f"{override_path}: {name} has invalid ne")
        offset = int(record["offset"])
        size = int(record["bytes"])
        if offset < 0 or size != int(np.prod(ne, dtype=np.int64)) * 4:
            raise ValueError(f"{override_path}: {name} has invalid F32 extent")
        if offset + size > installed.stat().st_size:
            raise ValueError(f"{override_path}: {name} exceeds override blob")
        manifest["tensors"][name] = {
            "source": record.get("source", "healed-router"),
            "file": str(rel),
            "offset": offset,
            "qtype": "F32",
            "ne": [int(value) for value in ne],
            "bytes": size,
        }
    manifest["tensor_overrides"] = {
        "receipt_path": str(override_path.resolve()),
        "receipt_sha256": sha256_file(override_path),
        "blob_sha256": blob["sha256"],
        "bytes": int(blob["bytes"]),
        "tensor_count": len(tensors),
    }


def prepare(args: argparse.Namespace) -> None:
    source_dir = Path(args.source_dir).resolve()
    fallback_dir = Path(args.fallback_dir).resolve() if args.fallback_dir else source_dir
    out_dir = Path(args.out_dir).resolve()
    plan_path = Path(args.plan).resolve()
    plan, assignments, pruned = load_assignments(plan_path)
    bridge, importance_receipts = _external_quantization_context(plan, assignments, args)
    if not (source_dir / "model.safetensors.index.json").exists():
        raise FileNotFoundError("quantization source requires model.safetensors.index.json")
    config = json.loads((source_dir / "config.json").read_text())
    is_mlx = bool(config.get("quantization") or config.get("quantization_config"))
    if is_mlx and fallback_dir == source_dir:
        raise ValueError("MLX quant sources require --fallback-dir pointing to a complete bw24 repack")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "experts").mkdir(exist_ok=True)
    store = SafeTensorDir(source_dir)
    max_work = int(args.max_work_mb) << 20
    if args.workers < 1:
        raise ValueError("workers must be at least 1")
    plan_sha256 = sha256_file(plan_path)
    source_fingerprints = _fingerprint(source_dir, ("config.json", "model.safetensors.index.json"))
    fallback_fingerprints = _fingerprint(
        fallback_dir, ("manifest.json", "config.json", "model.safetensors.index.json")
    )
    available_layers = _layers_from_model(plan["model"])
    layers = _parse_layer_subset(getattr(args, "layers", None), available_layers)
    fragment_path = getattr(args, "manifest_fragment", None)
    if fragment_path and getattr(args, "tensor_overrides", None):
        raise ValueError("tensor overrides are installed only while merging complete fragments")
    manifest: dict[str, Any] = {
        "format": OVERLAY_FORMAT,
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "source_dir": str(fallback_dir),
        "quant_source_dir": str(source_dir),
        "quality": "unverified - pending target-machine correctness and public eval gates",
        "plan": plan,
        "plan_sha256": plan_sha256,
        "plan_canonical_sha256": canonical_json_sha256(plan),
        "pruned_experts": {str(layer): sorted(ids) for layer, ids in pruned.items() if ids},
        "source_fingerprints": source_fingerprints,
        "fallback_fingerprints": fallback_fingerprints,
        "tensors": {},
        "tier_summary": {},
    }
    if bridge is not None:
        manifest["external_quantizer"] = bridge.provenance
        manifest["importance_sidecars"] = importance_receipts
    if fragment_path:
        manifest["fragment_layers"] = layers
    n_expert = int(plan["model"]["expert_count"])
    try:
        for layer in layers:
            active = [ex for ex in range(n_expert) if ex not in pruned[layer]]
            layer_importance: dict[str, np.ndarray] | None = None
            for proj in PROJECTIONS:
                source = ProjectionSource(store, config, layer, proj, active, max_work)
                if source.stem is not None and source.n_expert < n_expert:
                    raise ValueError(
                        f"source layer {layer}/{proj} has {source.n_expert} experts, plan expects {n_expert}"
                    )
                if bridge is not None and layer_importance is None:
                    if proj != "gate":
                        raise AssertionError("gate must initialize the layer importance sidecar")
                    layer_importance = load_importance_sidecar(
                        importance_receipts[str(layer)], n_expert, source.in_f, source.out_f
                    )
                rel = Path("experts") / f"blk{layer}-{proj}-mixed.bin"
                out_path = out_dir / rel
                expert_layout = []
                expected = 0
                for expert in active:
                    qtype = assignments[layer, expert, proj]
                    block, type_size, _ = QTYPES[qtype]
                    if source.in_f % block:
                        raise ValueError(
                            f"{layer}/{expert}/{proj}: in_features not aligned for {qtype}"
                        )
                    row_bytes = source.in_f // block * type_size
                    size = source.out_f * row_bytes
                    expert_layout.append({
                        "expert": expert,
                        "qtype": qtype,
                        "offset": expected,
                        "row_bytes": row_bytes,
                        "bytes": size,
                    })
                    expected += size
                layout_by_expert = {item["expert"]: item for item in expert_layout}
                receipt = {
                    "format": COMPLETION_RECEIPT_FORMAT,
                    "plan_sha256": plan_sha256,
                    "file": str(rel),
                    "layer": layer,
                    "projection": proj,
                    "expert_layout": expert_layout,
                    "source": {
                        "path": str(source_dir),
                        "fingerprints": source_fingerprints,
                    },
                    "fallback": {
                        "path": str(fallback_dir),
                        "fingerprints": fallback_fingerprints,
                    },
                    "shape": {
                        "experts": len(active),
                        "out_features": source.out_f,
                        "in_features": source.in_f,
                    },
                    "expected_bytes": expected,
                }
                if any(item["qtype"] in EXTERNAL_QTYPES for item in expert_layout):
                    receipt["external_quantizer"] = bridge.provenance
                    receipt["importance_sidecar"] = importance_receipts[str(layer)]
                receipt_path = _completion_receipt_path(out_path)
                reuse = (
                    args.resume
                    and out_path.exists()
                    and out_path.stat().st_size == expected
                    and _completion_receipt_matches(receipt_path, receipt)
                )
                if not reuse:
                    receipt_path.unlink(missing_ok=True)
                handle = None if reuse else out_path.open("wb")
                try:
                    offset = 0

                    def encode_expert(expert: int) -> tuple[int, bytes]:
                        qtype = assignments[layer, expert, proj]
                        if qtype in EXTERNAL_QTYPES:
                            assert bridge is not None and layer_importance is not None
                            key = "input" if proj in ("gate", "up") else "down"
                            importance = layer_importance[key][expert]
                            parts = [
                                bridge.quantize(rows, qtype, importance)
                                for rows in source.rows(expert)
                            ]
                        else:
                            parts = [QUANTIZERS[qtype](rows) for rows in source.rows(expert)]
                        return expert, b"".join(parts)

                    if handle is not None and args.workers > 1 and source.stem is None:
                        # Populate the shard cache before worker threads call store.raw(). The
                        # mmaps are then immutable shared reads; no thread races through _open_shard.
                        for expert in active:
                            name = ordinary_expert_name(store, layer, expert, proj)
                            if name is None:
                                raise KeyError(f"missing retained expert {layer}/{expert}/{proj}")
                            store.info(name)

                    if handle is None:
                        batches = ([(expert, b"")] for expert in active)
                    elif args.workers == 1:
                        batches = ([encode_expert(expert)] for expert in active)
                    else:
                        pool = ThreadPoolExecutor(max_workers=args.workers)
                        batches = (
                            list(pool.map(encode_expert, active[start : start + args.workers]))
                            for start in range(0, len(active), args.workers)
                        )

                    try:
                        encoded = (item for batch in batches for item in batch)
                        for expert, data in encoded:
                            qtype = assignments[layer, expert, proj]
                            layout = layout_by_expert[expert]
                            row_bytes = layout["row_bytes"]
                            size = layout["bytes"]
                            if handle is not None:
                                if len(data) != size:
                                    raise RuntimeError(
                                        f"{layer}/{expert}/{proj}: wrote {len(data)}, expected {size}"
                                    )
                                handle.write(data)
                            mapped = f"blk.{layer}.ffn_{proj}_exps.{expert}.weight"
                            manifest["tensors"][mapped] = {
                                "source": source.description(expert),
                                "file": str(rel),
                                "offset": offset,
                                "qtype": qtype,
                                "ne": [source.in_f, source.out_f],
                                "row_bytes": row_bytes,
                                "bytes": size,
                            }
                            summary = manifest["tier_summary"].setdefault(
                                qtype, {"experts": 0, "projections": 0, "bytes": 0}
                            )
                            summary["projections"] += 1
                            summary["bytes"] += size
                            if proj == "gate":
                                summary["experts"] += 1
                            offset += size
                    finally:
                        if handle is not None and args.workers > 1:
                            pool.shutdown()
                    if offset != expected:
                        raise RuntimeError(f"{rel}: layout {offset} != expected {expected}")
                finally:
                    if handle is not None:
                        handle.close()
                if out_path.stat().st_size != expected:
                    raise RuntimeError(f"{out_path}: size mismatch")
                if not reuse:
                    _write_completion_receipt(receipt_path, receipt)
                print(f"layer={layer:02d} proj={proj:4s} experts={len(active)} bytes={expected / 1e9:.3f}G", flush=True)
                source.stacked = None
                store.drop_cached_shards()
                trim_process_memory()
    finally:
        store.close()

    raw_override = getattr(args, "tensor_overrides", None)
    _install_tensor_overrides(
        out_dir, Path(raw_override).resolve() if raw_override else None, manifest
    )
    manifest["artifact_bytes"] = sum(v["bytes"] for v in manifest["tier_summary"].values())
    manifest["payload_bytes"] = manifest["artifact_bytes"] + int(
        manifest.get("tensor_overrides", {}).get("bytes", 0)
    )
    manifest_path = Path(fragment_path).resolve() if fragment_path else out_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    tmp.replace(manifest_path)
    print(f"wrote {manifest_path} ({len(manifest['tensors'])} expert projections)")


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
    root = Path(tempfile.mkdtemp(prefix="bw24-tiered-expert-test-"))
    try:
        source, out = root / "source", root / "overlay"
        source.mkdir()
        tensors = {}
        weight_map = {}
        shard = "model-00001-of-00001.safetensors"
        for proj in PROJECTIONS:
            for expert in range(4):
                name = f"model.layers.0.mlp.experts.{expert}.{proj}_proj.weight"
                vals = np.sin(np.arange(512, dtype=np.float32) * (expert + 1) / 31).reshape(2, 256)
                raw = (vals.view(np.uint32) >> 16).astype("<u2").tobytes()
                tensors[name] = ([2, 256], raw)
                weight_map[name] = shard
        _write_safetensors(source / shard, tensors)
        (source / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
        (source / "config.json").write_text("{}\n")
        plan = root / "plan.json"
        plan.write_text(json.dumps({
            "format": PLAN_FORMAT,
            "model": {"expert_count": 4, "moe_layers": [0]},
            "pruned_experts": {},
            "assignments": [
                {"layer": 0, "experts": [0], "qtype": "Q8_0"},
                {"layer": 0, "experts": [1], "qtype": "NVFP4"},
                {"layer": 0, "experts": [2], "qtype": "Q3_K"},
                {"layer": 0, "experts": [3], "qtype": "Q2_K"},
            ],
        }))
        prepare(SimpleNamespace(
            source_dir=str(source), fallback_dir=None, out_dir=str(out), plan=str(plan),
            max_work_mb=8, resume=False, workers=1, tensor_overrides=None,
        ))
        manifest = json.loads((out / "manifest.json").read_text())
        assert manifest["format"] == OVERLAY_FORMAT
        assert manifest["plan_sha256"] == sha256_file(plan)
        assert manifest["plan_canonical_sha256"] == canonical_json_sha256(manifest["plan"])
        assert manifest["plan_sha256"] != manifest["plan_canonical_sha256"]
        assert len(manifest["tensors"]) == 12
        override_blob = root / "router-overrides.bin"
        override_values = np.arange(12, dtype="<f4")
        override_blob.write_bytes(override_values.tobytes())
        override_receipt = root / "router-overrides.json"
        override_receipt.write_text(json.dumps({
            "format": "bw24-tensor-overrides-v1",
            "blob": {
                "path": str(override_blob), "bytes": override_blob.stat().st_size,
                "sha256": sha256_file(override_blob),
            },
            "tensors": {
                "blk.0.ffn_gate_inp.weight": {
                    "source": "model.layers.0.mlp.router.gate.weight",
                    "offset": 0, "qtype": "F32", "ne": [4, 3], "bytes": 48,
                }
            },
        }))
        override_manifest = {"tensors": {}}
        _install_tensor_overrides(out, override_receipt, override_manifest)
        record = override_manifest["tensors"]["blk.0.ffn_gate_inp.weight"]
        installed = out / record["file"]
        assert installed.read_bytes() == override_blob.read_bytes()
        assert override_manifest["tensor_overrides"]["tensor_count"] == 1
        _install_tensor_overrides(out, None, override_manifest)
        assert not (out / "overrides").exists()
        for proj in PROJECTIONS:
            records = [manifest["tensors"][f"blk.0.ffn_{proj}_exps.{ex}.weight"] for ex in range(4)]
            assert [r["qtype"] for r in records] == ["Q8_0", "NVFP4", "Q3_K", "Q2_K"]
            assert records[0]["offset"] == 0
            assert records[1]["offset"] == records[0]["bytes"]
            assert (out / records[0]["file"]).stat().st_size == sum(r["bytes"] for r in records)
        probe = np.sin(np.arange(512, dtype=np.float32) / 17).reshape(2, 256)
        for qtype, dequant, limit in (
            ("Q8_0", _dequant_q8_0, 0.0001),
            ("Q2_K", _dequant_q2k, 0.08),
            ("Q3_K", _dequant_q3k, 0.03),
            ("NVFP4", _dequant_nvfp4, 0.03),
        ):
            restored = dequant(QUANTIZERS[qtype](probe), 256)
            mse = float(np.mean((restored - probe) ** 2))
            assert np.isfinite(restored).all() and mse < limit, (qtype, mse)
        parallel_out = root / "overlay-parallel"
        prepare(SimpleNamespace(
            source_dir=str(source), fallback_dir=None, out_dir=str(parallel_out), plan=str(plan),
            max_work_mb=8, resume=False, workers=3,
        ))
        parallel_manifest = json.loads((parallel_out / "manifest.json").read_text())
        assert parallel_manifest["tensors"] == manifest["tensors"]
        assert parallel_manifest["tier_summary"] == manifest["tier_summary"]
        for path in sorted((out / "experts").iterdir()):
            assert path.read_bytes() == (parallel_out / "experts" / path.name).read_bytes()

        plan_a_bytes = {
            path.name: path.read_bytes() for path in (out / "experts").glob("*.bin")
        }
        plan.write_text(json.dumps({
            "format": PLAN_FORMAT,
            "model": {"expert_count": 4, "moe_layers": [0]},
            "pruned_experts": {},
            "assignments": [
                {"layer": 0, "experts": [0], "qtype": "Q2_K"},
                {"layer": 0, "experts": [1], "qtype": "Q8_0"},
                {"layer": 0, "experts": [2], "qtype": "NVFP4"},
                {"layer": 0, "experts": [3], "qtype": "Q3_K"},
            ],
        }))
        prepare(SimpleNamespace(
            source_dir=str(source), fallback_dir=None, out_dir=str(out), plan=str(plan),
            max_work_mb=8, resume=True, workers=1,
        ))
        fresh_b = root / "overlay-plan-b-fresh"
        prepare(SimpleNamespace(
            source_dir=str(source), fallback_dir=None, out_dir=str(fresh_b), plan=str(plan),
            max_work_mb=8, resume=False, workers=1,
        ))

        def stable_manifest(path: Path) -> dict[str, Any]:
            value = json.loads((path / "manifest.json").read_text())
            value.pop("created_utc")
            return value

        assert stable_manifest(out) == stable_manifest(fresh_b)
        plan_b_paths = sorted((fresh_b / "experts").glob("*.bin"))
        assert all(path.stat().st_size == len(plan_a_bytes[path.name]) for path in plan_b_paths)
        assert any(path.read_bytes() != plan_a_bytes[path.name] for path in plan_b_paths)
        for path in sorted((fresh_b / "experts").iterdir()):
            assert path.read_bytes() == (out / "experts" / path.name).read_bytes()

        # A missing receipt must rebuild only that projection, even when stale bytes have
        # exactly the expected size. Matching receipts keep the other projections resumable.
        gate_path = out / "experts" / "blk0-gate-mixed.bin"
        gate_path.write_bytes(plan_a_bytes[gate_path.name])
        _completion_receipt_path(gate_path).unlink()
        assert gate_path.read_bytes() != (fresh_b / "experts" / gate_path.name).read_bytes()
        calls = {qtype: 0 for qtype in QUANTIZERS}
        original_quantizers = QUANTIZERS.copy()
        for qtype, quantizer in original_quantizers.items():
            def counted(rows: np.ndarray, *, _qtype: str = qtype, _quantizer=quantizer) -> bytes:
                calls[_qtype] += 1
                return _quantizer(rows)

            QUANTIZERS[qtype] = counted
        try:
            prepare(SimpleNamespace(
                source_dir=str(source), fallback_dir=None, out_dir=str(out), plan=str(plan),
                max_work_mb=8, resume=True, workers=1,
            ))
        finally:
            QUANTIZERS.clear()
            QUANTIZERS.update(original_quantizers)
        assert calls == {"Q8_0": 1, "Q2_K": 1, "Q3_K": 1, "NVFP4": 1}, calls
        assert stable_manifest(out) == stable_manifest(fresh_b)
        for path in sorted((fresh_b / "experts").iterdir()):
            assert path.read_bytes() == (out / "experts" / path.name).read_bytes()
        print("tiered expert overlay self-test: PASS")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def probe(args: argparse.Namespace) -> None:
    source_dir = Path(args.source_dir).resolve()
    config = json.loads((source_dir / "config.json").read_text())
    store = SafeTensorDir(source_dir)
    source = None
    try:
        source = ProjectionSource(
            store, config, args.layer, args.projection, [args.expert], args.max_work_mb << 20,
        )
        if args.expert >= source.n_expert:
            raise ValueError(f"expert {args.expert} >= source expert count {source.n_expert}")
        row = next(source.rows(args.expert))[:1]
        sizes = {qtype: len(quantizer(row)) for qtype, quantizer in QUANTIZERS.items()}
        print(json.dumps({
            "source": source.description(args.expert),
            "n_expert": source.n_expert,
            "out_f": source.out_f,
            "in_f": source.in_f,
            "one_row_bytes": sizes,
        }, indent=2, sort_keys=True))
    finally:
        if source is not None:
            source.stacked = None
            del source
        store.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("source_dir")
    prep.add_argument("out_dir")
    prep.add_argument("--fallback-dir", help="complete HF or bw24 repack used for non-overlay tensors")
    prep.add_argument("--plan", required=True)
    prep.add_argument("--max-work-mb", type=int, default=512)
    prep.add_argument("--workers", type=int, default=1)
    prep.add_argument("--resume", action="store_true")
    prep.add_argument("--ggml-lib", type=Path)
    prep.add_argument("--ggml-lib-sha256")
    prep.add_argument("--ggml-source-commit")
    prep.add_argument(
        "--layers",
        help="optional comma-separated or inclusive range subset for a disjoint fragment build",
    )
    prep.add_argument(
        "--manifest-fragment",
        help="write an incomplete layer-fragment manifest here instead of OUT_DIR/manifest.json",
    )
    prep.add_argument(
        "--tensor-overrides",
        help="bw24-tensor-overrides-v1 receipt whose F32 router tensors override fallback",
    )
    inspect = sub.add_parser("probe")
    inspect.add_argument("source_dir")
    inspect.add_argument("--layer", type=int, required=True)
    inspect.add_argument("--expert", type=int, required=True)
    inspect.add_argument("--projection", choices=PROJECTIONS, required=True)
    inspect.add_argument("--max-work-mb", type=int, default=64)
    sub.add_parser("test")
    args = parser.parse_args()
    if args.cmd == "prepare":
        prepare(args)
    elif args.cmd == "probe":
        probe(args)
    else:
        self_test()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
