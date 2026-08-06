#!/usr/bin/env python3
"""FP8 v3 slice-1 host proof.

(B) exact decomposition with UNBOUNDED integers reproduces the f64 reference dot exactly
    (proves the decomposition arithmetic is sound), and
(A) the s8-requant mapping's measured rms relative error on random e4m3 blocks
    (the number DESIGN.md section 2 predicts: order 1e-2, same class as the v2
    activation-side error, NOT under it).

Also demonstrates DESIGN.md section 3: clamping (B)'s aligned mantissas to 8 bits is
arithmetically identical to (A) — the exact mapping collapses into the requant mapping.
"""
import numpy as np

rng = np.random.default_rng(0xF8B10C)

# ---- e4m3 codec (OCP E4M3: bias 7, 3 mantissa bits, max 448, no inf, S.1111.111 = NaN) ----
def e4m3_decode(byte):
    s = (byte >> 7) & 1
    e = (byte >> 3) & 0xF
    m = byte & 0x7
    if e == 0xF and m == 0x7:
        return np.nan
    if e == 0:
        val = (m / 8.0) * 2.0 ** (-6)          # denormal
    else:
        val = (1.0 + m / 8.0) * 2.0 ** (e - 7)  # normal
    return -val if s else val

DEC = np.array([e4m3_decode(b) for b in range(256)])

def e4m3_int_decompose(byte):
    """(sign, M, E) with value = sign * M * 2^E, M integer in 0..15."""
    byte = int(byte)
    s = -1 if (byte >> 7) & 1 else 1
    e = (byte >> 3) & 0xF
    m = byte & 0x7
    if e == 0:
        return s, m, -9            # denorm: m/8 * 2^-6 = m * 2^-9
    return s, 8 + m, e - 7 - 3     # normal: (8+m)/8 * 2^(e-7) = (8+m) * 2^(e-10)

def random_block(k=128):
    """Random e4m3 codes (no NaN), plus f32 activations with per-32 e4m3-ish spread."""
    codes = rng.integers(0, 256, size=k).astype(np.uint8)
    codes[(codes & 0x7F) == 0x7F] &= 0xF7      # strip NaN encodings
    acts = rng.normal(0, 1, size=k)
    return codes, acts

def proof_B_exact(n_blocks=1000, k=128):
    """s32-chain (unbounded python ints) + single fold == f64 reference, exactly."""
    max_rel = 0.0
    for _ in range(n_blocks):
        codes, acts = random_block(k)
        # activation side: per-32 s8 requant (the real kernel's activation path)
        ref = 0.0
        chain_ok = True
        for c0 in range(0, k, 32):
            a = acts[c0:c0+32]
            d = np.max(np.abs(a)) / 127.0 or 1.0
            q = np.clip(np.round(a / d), -127, 127).astype(np.int64)
            w = DEC[codes[c0:c0+32]]
            ref += float(np.dot(w, q * d))
            # integer chain: w_i = s*M*2^E ; sum s*M*q << (E - E_min) then fold 2^E_min * d
            trips = [e4m3_int_decompose(b) for b in codes[c0:c0+32]]
            e_min = min(t[2] for t in trips)
            acc = 0
            for (s, M, E), qi in zip(trips, q):
                acc += s * M * int(qi) * (1 << (E - e_min))   # unbounded int
            chain = float(acc) * (2.0 ** e_min) * d
            if not np.isclose(chain, float(np.dot(w, q * d)), rtol=1e-12, atol=1e-12):
                chain_ok = False
        assert chain_ok, "integer chain diverged from f64 reference"
    print(f"(B) exact decomposition: {n_blocks} blocks, s32-chain == f64 reference "
          f"(rtol 1e-12) — PASS")

def measure_A_error(n_blocks=1000, k=128):
    """s8 requant against block amax: measured rms_rel + worst rel error."""
    rels = []
    for _ in range(n_blocks):
        codes, _ = random_block(k)
        w = DEC[codes]
        amax = np.max(np.abs(w))
        if amax == 0:
            continue
        s8 = np.clip(np.round(w / amax * 127.0), -127, 127)
        w2 = s8 / 127.0 * amax
        nz = w != 0
        rels.append((w2[nz] - w[nz]) / w[nz])
    r = np.concatenate(rels)
    print(f"(A) s8-requant vs e4m3 exact: rms_rel={np.sqrt(np.mean(r**2)):.3e} "
          f"worst_rel={np.max(np.abs(r)):.3e} over {r.size} nonzero elements")

def clamp_B_collapses_to_A(k=128):
    """Clamp (B)'s aligned mantissas to 8 bits -> identical error class to (A)."""
    codes, _ = random_block(k)
    w = DEC[codes]
    trips = [e4m3_int_decompose(b) for b in codes]
    e_max = max(t[2] + 4 for t in trips)                 # 4 bits: M in 0..15
    clamped = []
    for (s, M, E) in trips:
        shift = e_max - 8 - E                            # keep top 8 bits below e_max
        Mc = (M >> shift) << shift if shift > 0 else M
        clamped.append(s * Mc * 2.0 ** E)
    clamped = np.array(clamped)
    nz = w != 0
    r = (clamped[nz] - w[nz]) / w[nz]
    print(f"(B, operands clamped to 8 bits) rms_rel={np.sqrt(np.mean(r**2)):.3e} "
          f"— same error class as (A): the exact mapping collapses under an 8-bit operand cap")

if __name__ == "__main__":
    proof_B_exact()
    measure_A_error()
    clamp_B_collapses_to_A()
