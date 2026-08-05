#!/usr/bin/env python3
"""Transcode the local 27B FP8-ST checkpoint from PER-TENSOR scale to BLOCK-128 scale.

WHY. lane/fp8-blk128-decode needs a 27B-CLASS block-128 FP8 checkpoint: the owner rule is that
verdicts anchor on 27B shapes only (the 1.7B synth checkpoint is a bring-up instrument, never a
verdict), and no official Qwen3.6-27B-FP8 is staged on this box (29 GB, lives on the remote 2x5090).
What IS staged is `nvidia-qwen36-27b-nvfp4` — the same 27B, MIXED_PRECISION, whose 208 attention /
linear-attn projections are F8_E4M3 with a SCALAR `weight_scale` (modelopt per-tensor class, the
class lane/fp8-decode-v1 already ships natively). Re-expressing exactly those 208 tensors in the
Qwen-official block-128 encoding produces a genuine 27B block-128 checkpoint on real 27B shapes.

WHAT IT IS AND IS NOT. This is a TRANSCODE, and every number measured on it must say so:
  * The container is genuine: F8_E4M3 codes + a BF16 `weight_scale_inv` of shape
    [ceil(out/128), ceil(in/128)], per-block s = amax/448 — byte-for-byte the encoding
    Qwen3.6-FP8 / DeepSeek-V3 ship and the one `f8_scales` classifies as Block128.
  * The VALUES are twice-quantized (per-tensor e4m3 -> f32 -> per-block e4m3). That makes it
    unusable as a MODEL-QUALITY artifact vs the original, and this script never claims otherwise.
    It is a PERFORMANCE + RESIDENCY + DISPATCH instrument, and for those it is exact: the decode
    A/B and the teacher-forced exactness arm both run on THIS checkpoint's bytes, native arm vs
    Q8_0-slab arm, so the comparison is single-variable and the transcode cancels.
  * Dequant of the source is EXACT (value = code * scalar, both f32-representable), so no
    information is lost before the re-block; per-block re-quantization then gives each 128x128 tile
    its own scale, which is strictly finer than the single scalar it replaces.

The NVFP4 MLP half of the checkpoint (193 U8 weight planes + their e4m3 micro-scale planes) and
every norm/embedding tensor pass through BYTE-IDENTICAL: the FP8 arm is the only variable.

Output keeps the source's 3-shard layout and rewrites model.safetensors.index.json, so the
`weight_scale` -> `weight_scale_inv` rename is visible to the loader exactly as a real checkpoint's
would be.

Pure numpy (no torch). Usage: make_27b_blk128.py <src_dir> <dst_dir>
"""
import json
import os
import struct
import sys

import numpy as np


# --- e4m3 (signed, bias 7, NaN = magnitude 0x7F) decode table, memra convention ---------------
# 0x7F/0xFF decode to 0.0 to match nvfp4_repack::fp8_e4m3_to_f32 (the loader's own dequant), so a
# source NaN code round-trips to code 0x00 and the output carries NO NaN codes at all — which is
# also the native block-128 residency arm's dispatch precondition (`fp8_blk_nan_count == 0`).
def e4m3_table():
    t = np.zeros(256, dtype=np.float32)
    for c in range(256):
        mag = c & 0x7F
        if mag == 0x7F:
            t[c] = 0.0
            continue
        exp = (mag >> 3) & 0xF
        man = float(mag & 0x7)
        raw = (man / 8.0) * 2.0 ** -6 if exp == 0 else (1.0 + man / 8.0) * 2.0 ** (exp - 7)
        t[c] = -raw if (c & 0x80) else raw
    return t


E4M3 = e4m3_table()
MAGS = E4M3[:0x7F].copy()  # magnitudes of codes 0..0x7E, strictly increasing


def f32_to_e4m3(x):
    """Nearest-e4m3 encode, ties to EVEN code, saturating at +-448. Mirrors
    nvfp4_repack::f32_to_fp8_e4m3. Never emits 0x7F (the NaN code)."""
    x = np.asarray(x, dtype=np.float32)
    sign = (np.signbit(x).astype(np.uint8) << 7)
    ax = np.abs(x)
    ax = np.where(np.isnan(ax), np.float32(0.0), ax)
    hi = np.searchsorted(MAGS, ax, side="left").clip(0, 0x7E)
    lo = (hi - 1).clip(0, 0x7E)
    dlo = ax - MAGS[lo]
    dhi = MAGS[hi] - ax
    pick = np.where(dlo < dhi, lo, np.where(dhi < dlo, hi, np.where(lo % 2 == 0, lo, hi)))
    pick = np.where(ax >= MAGS[0x7E], 0x7E, pick).astype(np.uint8)
    return np.where(pick == 0, np.uint8(0), (sign | pick)).astype(np.uint8)


def bf16_round(a):
    """Round f32 -> bf16 -> f32 (truncate mantissa). The grid ships BF16, so the codes must be
    encoded against the ROUNDED scale or the loader's decode would not reproduce this `s`."""
    u = np.asarray(a, dtype="<f4").view("<u4")
    return ((u >> 16) << 16).view("<f4")


def f32_to_bf16_bytes(a):
    u = np.asarray(a, dtype="<f4").view("<u4")
    return (u >> 16).astype("<u2").tobytes()


def read_header(path):
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        hdr = json.loads(f.read(hlen))
    return hdr, 8 + hlen


def blockify(codes_u8, out_f, in_f, scalar):
    """per-tensor e4m3 codes + scalar -> (block-128 codes, [rows,cols] f32 grid).

    Fully vectorized when both dims are multiples of 128 (true for every 27B projection shape:
    12288/5120/1024/6144/10240/17408 all divide by 128); falls back to a tiled loop otherwise so
    the script stays correct for ragged shapes (e.g. a future model with a 320-wide dim).
    """
    w = E4M3[codes_u8.reshape(out_f, in_f)] * np.float32(scalar)
    rows, cols = -(-out_f // 128), -(-in_f // 128)
    if out_f % 128 == 0 and in_f % 128 == 0:
        tiles = w.reshape(rows, 128, cols, 128)
        amax = np.abs(tiles).max(axis=(1, 3))
        s = np.where(amax > 0, amax / np.float32(448.0), np.float32(1.0)).astype(np.float32)
        s = bf16_round(s)
        s = np.where(np.isfinite(s) & (s > 0), s, np.float32(1.0)).astype(np.float32)
        codes = f32_to_e4m3(tiles / s[:, None, :, None]).reshape(out_f, in_f)
        return codes, s
    grid = np.ones((rows, cols), dtype=np.float32)
    codes = np.zeros((out_f, in_f), dtype=np.uint8)
    for ob in range(rows):
        r0, r1 = ob * 128, min((ob + 1) * 128, out_f)
        for kb in range(cols):
            c0, c1 = kb * 128, min((kb + 1) * 128, in_f)
            tile = w[r0:r1, c0:c1]
            amax = float(np.abs(tile).max()) if tile.size else 0.0
            s = bf16_round(np.float32(amax / 448.0 if amax > 0 else 1.0))
            if not (s > 0 and np.isfinite(s)):
                s = np.float32(1.0)
            grid[ob, kb] = s
            codes[r0:r1, c0:c1] = f32_to_e4m3(tile / s)
    return codes, grid


def main():
    src, dst = sys.argv[1], sys.argv[2]
    os.makedirs(dst, exist_ok=True)

    idx_path = os.path.join(src, "model.safetensors.index.json")
    index = json.load(open(idx_path))
    wmap = index["weight_map"]
    shards = sorted(set(wmap.values()))

    new_wmap = {}
    n_blk = 0
    total_out = 0
    for sh in shards:
        hdr, base = read_header(os.path.join(src, sh))
        blob = np.memmap(os.path.join(src, sh), dtype=np.uint8, mode="r")
        names = [k for k in hdr if k != "__metadata__"]
        # A weight is in the per-tensor FP8 class iff it is F8_E4M3 2D AND its `.weight_scale`
        # sibling is a SCALAR. That excludes the NVFP4 micro-scale planes, which are also stored as
        # F8_E4M3 but are `*.weight_scale` themselves (shape [rows, cols/16]) — they pass through.
        fp8_wt = set()
        for n in names:
            if hdr[n]["dtype"] != "F8_E4M3" or len(hdr[n]["shape"]) != 2:
                continue
            if not n.endswith(".weight"):
                continue
            sib = n[: -len(".weight")] + ".weight_scale"
            if sib in hdr and len(hdr[sib]["shape"]) == 0:
                fp8_wt.add(n)
        drop = {n[: -len(".weight")] + ".weight_scale" for n in fp8_wt}

        out = {}   # new name -> (dtype, shape, bytes)
        for n in names:
            if n in drop:
                continue
            meta = hdr[n]
            o0, o1 = meta["data_offsets"]
            if n not in fp8_wt:
                out[n] = (meta["dtype"], meta["shape"], bytes(blob[base + o0: base + o1]))
                continue
            out_f, in_f = meta["shape"]
            sinfo = hdr[n[: -len(".weight")] + ".weight_scale"]
            s0, s1 = sinfo["data_offsets"]
            sraw = bytes(blob[base + s0: base + s1])
            if sinfo["dtype"] == "F32":
                scalar = float(np.frombuffer(sraw, dtype="<f4")[0])
            else:  # BF16 scalar
                scalar = float(np.frombuffer(
                    (np.frombuffer(sraw, dtype="<u2").astype("<u4") << 16).tobytes(),
                    dtype="<f4")[0])
            codes = np.asarray(blob[base + o0: base + o1])
            new_codes, grid = blockify(codes, out_f, in_f, scalar)
            assert not (new_codes & 0x7F == 0x7F).any(), f"{n}: NaN code emitted"
            out[n] = ("F8_E4M3", [out_f, in_f], new_codes.tobytes())
            out[n[: -len(".weight")] + ".weight_scale_inv"] = (
                "BF16", list(grid.shape), f32_to_bf16_bytes(grid))
            n_blk += 1
            if n_blk % 16 == 0:
                print(f"  [{n_blk}] {n} {out_f}x{in_f} grid {grid.shape}", flush=True)

        # write the shard
        nh, off, payload = {}, 0, []
        for name in sorted(out):
            dt, shape, raw = out[name]
            nh[name] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + len(raw)]}
            payload.append(raw)
            off += len(raw)
            new_wmap[name] = sh
        hj = json.dumps(nh).encode()
        hj += b" " * ((-len(hj)) % 8)
        with open(os.path.join(dst, sh), "wb") as f:
            f.write(struct.pack("<Q", len(hj)))
            f.write(hj)
            for p in payload:
                f.write(p)
        total_out += off
        print(f"shard {sh}: {len(out)} tensors, {off / 1e9:.2f} GB", flush=True)
        del blob, out, payload

    json.dump({"metadata": index.get("metadata", {}), "weight_map": new_wmap},
              open(os.path.join(dst, "model.safetensors.index.json"), "w"))
    for aux in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                "vocab.json", "merges.txt", "chat_template.jinja", "hf_quant_config.json",
                "preprocessor_config.json", "processor_config.json", "configuration.json",
                "video_preprocessor_config.json", "frspec-corpus-32768.gguf"):
        s = os.path.join(src, aux)
        if os.path.exists(s):
            with open(s, "rb") as fi, open(os.path.join(dst, aux), "wb") as fo:
                while True:
                    b = fi.read(1 << 22)
                    if not b:
                        break
                    fo.write(b)
    print(f"DONE {dst}: {n_blk} block-128 FP8 weights, {total_out / 1e9:.2f} GB payload")


if __name__ == "__main__":
    main()
