#!/usr/bin/env python3
"""Scope the f8f4 seam: which fast-gate models carry NVFP4 (ggml type 40) 2-D weights?

The f8f4 seam (MEMRA_MMQ_F8F4=1) lives in mmq_ffi.rs qmatvec_mmq_nvfp4_w4a8_scaled, which is
reached ONLY from the `q if q == QT_NVFP4 && use_w4a8` arm of qmatvec_mmq (mmq_ffi.rs:568).
A model with zero NVFP4 tensors therefore CANNOT route prefill through the f8f4 tile, and its
argmax/acceptance is invariant to the flag BY CONSTRUCTION (no dispatch site).

Reads only the GGUF header (no tensor data), prints a per-model type histogram.
"""
import struct
import sys

NVFP4 = 40
NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS",
    17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S",
    23: "IQ4_XS", 24: "I8", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4", 40: "NVFP4",
    41: "Q1_0",
}


def rd(f, fmt):
    n = struct.calcsize(fmt)
    b = f.read(n)
    if len(b) != n:
        raise EOFError
    return struct.unpack(fmt, b)


def rd_str(f):
    (n,) = rd(f, "<Q")
    return f.read(n).decode("utf-8", "replace")


def skip_val(f, t):
    # GGUF metadata value types
    fixed = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    if t in fixed:
        f.read(fixed[t])
    elif t == 8:  # string
        rd_str(f)
    elif t == 9:  # array
        (et,) = rd(f, "<I")
        (n,) = rd(f, "<Q")
        for _ in range(n):
            skip_val(f, et)
    else:
        raise ValueError(f"unknown gguf value type {t}")


def scan(path):
    with open(path, "rb") as f:
        magic, ver = rd(f, "<4sI")
        if magic != b"GGUF":
            raise ValueError(f"not GGUF: {magic!r}")
        (n_tensors,) = rd(f, "<Q")
        (n_kv,) = rd(f, "<Q")
        for _ in range(n_kv):
            rd_str(f)
            (t,) = rd(f, "<I")
            skip_val(f, t)
        hist = {}
        nvfp4_2d = 0
        nvfp4_names = []
        for _ in range(n_tensors):
            name = rd_str(f)
            (ndim,) = rd(f, "<I")
            dims = [rd(f, "<Q")[0] for _ in range(ndim)]
            (ttype,) = rd(f, "<I")
            rd(f, "<Q")  # offset
            hist[ttype] = hist.get(ttype, 0) + 1
            if ttype == NVFP4:
                nvfp4_2d += 1
                if len(nvfp4_names) < 4:
                    nvfp4_names.append(f"{name}{dims}")
        return n_tensors, hist, nvfp4_2d, nvfp4_names


if __name__ == "__main__":
    for path in sys.argv[1:]:
        try:
            n, hist, nv, names = scan(path)
        except Exception as e:  # noqa: BLE001
            print(f"{path}\tERROR {e}")
            continue
        h = " ".join(
            f"{NAMES.get(t, t)}={c}" for t, c in sorted(hist.items(), key=lambda kv: -kv[1])
        )
        verdict = "F8F4-REACHABLE" if nv else "no-nvfp4 (seam unreachable)"
        print(f"{path}\n  tensors={n} nvfp4={nv} [{verdict}]\n  {h}")
        if names:
            print(f"  e.g. {', '.join(names)}")
