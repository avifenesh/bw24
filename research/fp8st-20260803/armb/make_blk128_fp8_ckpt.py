#!/usr/bin/env python3
"""Build a REAL Qwen-official-style block-128 FP8 safetensors checkpoint from a BF16 ST dir.

No block-128 FP8 checkpoint exists on the 5090 rig (scan 2026-08-03: the only FP8-bearing
local checkpoint is nvidia-qwen36-27b-nvfp4, which carries per-tensor `weight_scale`, not the
`weight_scale_inv` grid). ARM B's real-checkpoint argmax gate needs a genuine loader input, so
this script produces one: every 2D Linear weight becomes F8_E4M3 codes + a BF16
`weight_scale_inv` grid of shape [ceil(out/128), ceil(in/128)] — the exact encoding
Qwen3.6-27B-FP8 / DeepSeek-V3 ship. Non-Linear tensors (norms, embeddings) pass through
unchanged.

Per-block quantization: s = amax(block)/448 (e4m3 max normal), codes = nearest-e4m3(w/s).
That is a real FP8 checkpoint, not a fixture: dynamic range varies block to block, which is
precisely the property ARM A's global fold destroys and ARM B' preserves.

Usage: make_blk128_fp8_ckpt.py <src_bf16_dir> <dst_dir> [max_layers]
Pure numpy (no torch / no safetensors package needed).
"""
import json
import os
import struct
import sys

import numpy as np

# --- e4m3 (signed, bias 7, NaN = magnitude 0x7F) decode table, memra convention ---------------
def e4m3_table():
    t = np.zeros(256, dtype=np.float32)
    for c in range(256):
        mag = c & 0x7F
        if mag == 0x7F:
            t[c] = 0.0  # NaN code -> 0.0 (modelopt convention, matches fp8_e4m3_to_f32)
            continue
        exp = (mag >> 3) & 0xF
        man = float(mag & 0x7)
        raw = (man / 8.0) * 2.0 ** -6 if exp == 0 else (1.0 + man / 8.0) * 2.0 ** (exp - 7)
        t[c] = -raw if (c & 0x80) else raw
    return t


E4M3 = e4m3_table()
# magnitudes of codes 0..0x7E, strictly increasing (0x7F is NaN) — the encode grid
MAGS = E4M3[: 0x7F].copy()


def f32_to_e4m3(x):
    """Nearest-e4m3 encode, ties to the EVEN code, saturating at +-448. Mirrors
    nvfp4_repack::f32_to_fp8_e4m3."""
    x = np.asarray(x, dtype=np.float32)
    sign = np.signbit(x).astype(np.uint8) << 7
    ax = np.abs(x)
    ax = np.where(np.isnan(ax), 0.0, ax)
    # searchsorted over the increasing magnitude grid, then pick the nearer neighbour
    hi = np.searchsorted(MAGS, ax, side="left").clip(0, 0x7E)
    lo = (hi - 1).clip(0, 0x7E)
    dlo = ax - MAGS[lo]
    dhi = MAGS[hi] - ax
    pick = np.where(dlo < dhi, lo, np.where(dhi < dlo, hi, np.where(lo % 2 == 0, lo, hi)))
    pick = np.where(ax >= MAGS[0x7E], 0x7E, pick).astype(np.uint8)
    code = np.where(pick == 0, np.uint8(0), (sign | pick)).astype(np.uint8)
    return code


def bf16_to_f32(raw):
    u16 = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
    return (u16 << 16).view(np.float32) if False else np.frombuffer(
        (u16 << 16).astype("<u4").tobytes(), dtype="<f4")


def f32_to_bf16_bytes(a):
    u32 = np.asarray(a, dtype="<f4").view("<u4")
    return (u32 >> 16).astype("<u2").tobytes()


def read_st(path):
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        hdr = json.loads(f.read(hlen))
        base = 8 + hlen
        blob = np.memmap(path, dtype=np.uint8, mode="r")
    return hdr, base, blob


DT = {"F32": ("<f4", 4), "F16": ("<f2", 2), "BF16": (None, 2), "I64": ("<i8", 8),
      "U8": ("<u1", 1), "I32": ("<i4", 4), "F8_E4M3": ("<u1", 1), "BOOL": ("<u1", 1)}


def main():
    src, dst = sys.argv[1], sys.argv[2]
    max_layers = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    os.makedirs(dst, exist_ok=True)

    shards = sorted(p for p in os.listdir(src) if p.endswith(".safetensors"))
    out_tensors = {}  # name -> (dtype, shape, bytes)
    n_fp8 = 0
    for sh in shards:
        hdr, base, blob = read_st(os.path.join(src, sh))
        for name, meta in hdr.items():
            if name == "__metadata__":
                continue
            shape = meta["shape"]
            dt = meta["dtype"]
            o0, o1 = meta["data_offsets"]
            raw = bytes(blob[base + o0: base + o1])
            if max_layers and ".layers." in name:
                li = int(name.split(".layers.")[1].split(".")[0])
                if li >= max_layers:
                    continue
            quantizable = (
                len(shape) == 2
                and dt in ("BF16", "F16", "F32")
                and (".mlp." in name or ".self_attn." in name)
                and name.endswith(".weight")
                and shape[0] >= 128 and shape[1] >= 128
            )
            if not quantizable:
                out_tensors[name] = (dt, shape, raw)
                continue
            out_f, in_f = shape
            if dt == "BF16":
                w = bf16_to_f32(raw).reshape(out_f, in_f)
            elif dt == "F16":
                w = np.frombuffer(raw, dtype="<f2").astype(np.float32).reshape(out_f, in_f)
            else:
                w = np.frombuffer(raw, dtype="<f4").reshape(out_f, in_f)
            rows, cols = -(-out_f // 128), -(-in_f // 128)
            codes = np.zeros((out_f, in_f), dtype=np.uint8)
            grid = np.zeros((rows, cols), dtype=np.float32)
            for ob in range(rows):
                r0, r1 = ob * 128, min((ob + 1) * 128, out_f)
                for kb in range(cols):
                    c0, c1 = kb * 128, min((kb + 1) * 128, in_f)
                    tile = w[r0:r1, c0:c1]
                    amax = float(np.max(np.abs(tile))) if tile.size else 0.0
                    # the checkpoint's grid is stored BF16 -> round the scale to bf16 FIRST so
                    # the loader's decode reproduces exactly the s used for the codes here
                    s = amax / 448.0 if amax > 0 else 1.0
                    s = np.frombuffer(f32_to_bf16_bytes(np.float32(s)) + b"\0\0",
                                      dtype="<f4")[0] if False else np.frombuffer(
                        (np.uint32(np.float32(s).view(np.uint32)) >> 16 << 16
                         ).astype("<u4").tobytes(), dtype="<f4")[0]
                    if not (s > 0 and np.isfinite(s)):
                        s = np.float32(1.0)
                    grid[ob, kb] = s
                    codes[r0:r1, c0:c1] = f32_to_e4m3(tile / s)
            out_tensors[name] = ("F8_E4M3", [out_f, in_f], codes.tobytes())
            stem = name[: -len(".weight")]
            out_tensors[stem + ".weight_scale_inv"] = (
                "BF16", [rows, cols], f32_to_bf16_bytes(grid))
            n_fp8 += 1
        del blob

    # write ONE shard (these models are small enough)
    hdr = {}
    off = 0
    payload = []
    for name in sorted(out_tensors):
        dt, shape, raw = out_tensors[name]
        hdr[name] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + len(raw)]}
        payload.append(raw)
        off += len(raw)
    hj = json.dumps(hdr).encode()
    pad = (-len(hj)) % 8
    hj += b" " * pad
    with open(os.path.join(dst, "model.safetensors"), "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        for p in payload:
            f.write(p)
    for aux in ("config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
                "merges.txt", "generation_config.json", "chat_template.jinja"):
        s = os.path.join(src, aux)
        if os.path.exists(s):
            with open(s, "rb") as fi, open(os.path.join(dst, aux), "wb") as fo:
                fo.write(fi.read())
    if max_layers:
        cfgp = os.path.join(dst, "config.json")
        cfg = json.load(open(cfgp))
        tgt = cfg.get("text_config", cfg)
        tgt["num_hidden_layers"] = max_layers
        json.dump(cfg, open(cfgp, "w"))
    print(f"wrote {dst}: {len(out_tensors)} tensors, {n_fp8} block-128 FP8 weights, "
          f"{off / 1e9:.2f} GB payload")


if __name__ == "__main__":
    main()
