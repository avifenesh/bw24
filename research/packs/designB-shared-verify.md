# Design-B Shared-KV Verify Walk — Exactness Proof

## Context

The multi-row verify FA kernel `fa_decode_vec_q_rows` (flash_attn.cu:1440) launches
`grid.z = row` where each row r independently computes its own split-K decode attention
over `T_kv_r = t_kv_base + r + 1` keys. Design-B proposes a SINGLE walker that processes
the full KV range once, updating K+1 rows' online-softmax states in parallel, to amortize
the KV read across all verify rows (they share a common KV prefix, differing only in the
last key each row attends to).

## The Split-Key Formula (from code)

```
// flash_attn.cu:1453
const int T_kv     = t_kv_base + r + 1;
const int n_splits = (T_kv + split_keys - 1) / split_keys;  // ceil(T_kv / split_keys)

// flash_attn.cu:1464
const int per  = (T_kv + n_splits - 1) / n_splits;          // keys per split (may differ from split_keys!)
const int t_lo = split * per;
const int t_hi = min(T_kv, t_lo + per);
```

**Critical distinction:** `split_keys` is 64 (the fixed constant from `fa_split_keys`).
But `per` (the actual keys-per-split) is NOT always 64. It is `ceil(T_kv / n_splits)`
where `n_splits = ceil(T_kv / 64)`.

## Working the Math: Do Split Boundaries Align Across Rows?

### Example: t_kv_base = 130, K=3 (4 rows: r=0,1,2,3)

| Row r | T_kv_r | n_splits_r | per_r | Split boundaries |
|-------|--------|-----------|-------|------------------|
| 0 | 131 | ceil(131/64) = 3 | ceil(131/3) = 44 | [0,44), [44,88), [88,131) |
| 1 | 132 | ceil(132/64) = 3 | ceil(132/3) = 44 | [0,44), [44,88), [88,132) |
| 2 | 133 | ceil(133/64) = 3 | ceil(133/3) = 45 | [0,45), [45,90), [90,133) |
| 3 | 134 | ceil(134/64) = 3 | ceil(134/3) = 45 | [0,45), [45,90), [90,134) |

**Result: boundaries DIVERGE.** Row 0 processes keys [0,44), [44,88), [88,131) while
row 2 processes [0,45), [45,90), [90,133). The split boundaries are NOT aligned even though
`split_keys` is the same fixed constant for all rows.

### Why `per_r` is NOT simply split_keys

The formula `per = ceil(T_kv / n_splits)` redistributes keys to equalize split load:
- n_splits = ceil(T_kv / 64)
- per = ceil(T_kv / n_splits)

When T_kv is not a multiple of 64:
- n_splits = floor((T_kv + 63) / 64)
- per = ceiling(T_kv / n_splits)

This makes `per` depend on T_kv mod 64, which differs per row (since T_kv_r = T_kv_base + r + 1,
consecutive rows differ by 1). The result: per_r is either 64, 65, or some other ceiling value
that depends on the exact T_kv for that row.

### Systematic analysis over all T_kv mod 64

For T_kv = 64*q + m (m = 1..63):
- n_splits = q + 1
- per = ceil((64*q + m) / (q+1)) = 64*q/(q+1) + m/(q+1) rounded up

For m > 0 this is generally NOT 64 unless (64*q + m) is exactly divisible by (q+1).

Counter-example (the common case): T_kv = 65 (q=1, m=1):
- n_splits = 2
- per = ceil(65/2) = 33
- Boundaries: [0,33), [33,65)

NOT [0,64),[64,65). The formula actively re-balances.

## Consequence for Design-B

A shared walker processing keys 0..max_T_kv in fixed 64-key chunks CANNOT reproduce
each row's per-row accumulation order because:

1. **Split boundaries diverge across rows.** The online-softmax `alpha` rescale happens at
   each split boundary (the `fa_decode_combine_rows` merge). A shared walker accumulating
   all keys in one pass produces a SINGLE accumulation sequence per row, but the actual
   kernel's per-row accumulation is split into n_splits_r independent PARTIAL sums merged
   in `fa_decode_combine_rows`. These two orderings produce FP-different results.

2. **The combine step reintroduces per-split m_i/l_i independently.** Even if we could
   match the within-split accumulation order, the combine kernel (`fa_decode_combine_rows`)
   iterates `s = 0..n_splits-1` in order, applying `w = exp2((ms - m_global) * LOG2E)`.
   This weighted sum of partial O values does NOT equal a single-pass scan in a different
   order — the partial m_i and l_i per split are derived from different key ranges per row.

3. **Rows within the same walker iteration are at DIFFERENT states of their online-softmax**
   because each row has accumulated a different number of keys at any given point in the
   shared walk (rows with smaller T_kv_r have already "finished" while later rows continue).

## Verdict: INFEASIBLE (Exactness-Violating)

Design-B cannot reproduce the exact per-row accumulation order of `fa_decode_vec_q_rows` +
`fa_decode_combine_rows` because:

- The split boundaries are T_kv_r-dependent and NOT aligned across rows.
- The per-row combine is an independent weighted sum over per-split partials.
- A shared single-pass walker produces a fundamentally different FP accumulation sequence.

The only way to make Design-B exact would be to change BOTH the multi-row kernel AND the
single-step decode kernel to use the same accumulation order (e.g., always process keys in
fixed 64-key chunks without the per-row-balanced rebalancing). But this would break the
**spec-exactness law**: the verify decode MUST produce bit-identical results to the eager
single-step decode, and the single-step decode uses the same `per = ceil(T_kv/n_splits)`
formula (flash_attn.cu:1134, fa_decode_vec_q:1135-1136).

Changing the split formula to a fixed-chunk scheme (per = split_keys always, last split
partial) would invalidate the existing spec-exactness gates and require re-validating the
entire test battery — a non-trivial risk for the 4-8% bandwidth saving.

## Register Budget Analysis (moot given infeasibility, but included for completeness)

If Design-B WERE feasible (e.g., with a fixed-chunk scheme), the register budget for K+1=4
rows at head_dim=256, dpl=8 would be:

| State | Per row | x4 rows | Regs/lane |
|-------|---------|---------|-----------|
| acc[8] f32 | 8 | 32 | 32 |
| q_reg[8] f32 | 8 | 32 | 32 |
| m_i f32 | 1 | 4 | 4 |
| l_i f32 | 1 | 4 | 4 |
| **Total per-row state** | | | **72** |
| KV dequant temps | | | ~8 |
| Loop control | | | ~4 |
| **Grand total** | | | **~84** |

sm_120a register file: 256 regs/lane (65536 regs / 256 threads per SM). At 84 regs/lane
the kernel would need max 84 regs which is feasible (occupancy ~3 warps/SM at 84 regs vs
the current ~4 warps/SM at 64 regs for fa_decode_vec_q).

However, the q_reg is PER HEAD (not per row!) since all rows share the same Q for a given
GQA head — wait, NO: in verify, each row r has its OWN Q vector (it's a different token).
So the full 4x q_reg IS needed. At 84 regs/lane, occupancy is adequate but not the
bottleneck — the exactness violation is the killer.

## Recommendation

**Do not pursue Design-B.** The per-row split boundaries are NOT aligned due to the
`per = ceil(T_kv_r / n_splits_r)` rebalancing formula, making a shared-walker approach
fundamentally incompatible with the spec-exactness law.

Alternative approaches for the C2 (~16%) bandwidth cost of the multi-row verify:
1. L2 cache reuse is already happening (the register-dequant rewrite, 2026-07-03 comment in
   flash_attn.cu:1162-1163 says "4 GQA warps re-read the same KV bytes; L2 serves the reuse").
2. Accept the bandwidth cost as the price of exactness.
3. If exactness could be relaxed (e.g., argmax-only equivalence), a shared-walker with per-row
   masking at the natural 64-key chunk level would be viable — but this is a policy decision
   that would require re-gating the entire spec battery.
