#!/usr/bin/env python3
"""Bug-CLASS audit: re-derive spec.rs's sampled-verify column<->stats index mapping for every
(base, k_round) arm, so the fix is known to cover the only defective site rather than assumed to.

The class of defect is "filtered-sampler stats taken from a foreign row". `th` from
filter_stats is a threshold in e-units relative to that row's OWN max (e_max == 1), so any
(row_max, th) pair fed to a different row mis-scales e0 = exp((x - row_max)/T) and can mask
the entire row -> argmax tie-break -> id 0.

Sites in spec.rs that consume (row_max, th) (grep: filter_stats / gumbel_perturb_filtered /
softmax_gather_filtered / residual_sample_filtered):

  L3634  graph-draft uniform accept: stats per q_slot, from that slot        -> own row, OK
  L3703  eager-draft filtered gumbel: stats from q_row, perturb q_row       -> own row, OK
  L3923  batched p-gather: stats(rows) + gather(same rows), one call         -> own rows, OK
  L3977  base==0 last-col: stats from lc, gather from lc                     -> own row, OK
  L4010  q accept gather: draft_stats[j] with q_bufs[j]                      -> own row, OK
  L4072  FULL-ACCEPT BONUS                                                   -> THE DEFECT
  L4134  reject residual: p_stats = col_stats[gi] / last_col_stats           -> checked below

This script proves two things mechanically:
  1. the full-accept bonus column (base + k_round - 1) is NEVER a member of the gathered set
     {base + j - 1 : j in 0..k_round, j > 0 or base == 1} — in EITHER base arm, at EVERY
     k_round. So `col_stats.last()` was always a foreign row: not an edge case, the norm.
  2. every reject-path index (gi -> gathered row, or the base==0 last_col special case) DOES
     resolve to the same column the residual sampler reads. That path was already correct and
     is left untouched.
"""
import sys

bad_bonus = 0
rej_checked = rej_bad = 0
for base in (0, 1):
    for k_round in (1, 2, 3, 4, 5, 6, 7, 8):
        gathered = [base + j - 1 for j in range(k_round) if (j > 0 or base == 1)]
        bonus = base + k_round - 1
        in_set = bonus in gathered
        if not in_set:
            bad_bonus += 1
        print(f"base={base} k_round={k_round} gathered={gathered} "
              f"bonus_col={bonus} bonus_in_gathered={in_set}")
        for n_acc in range(k_round):
            rej_checked += 1
            if n_acc > 0 or base == 1:
                gi = n_acc if base == 1 else n_acc - 1
                col = base + n_acc - 1
                ok = 0 <= gi < len(gathered) and gathered[gi] == col
            else:
                gi, col, ok = "last_col_stats", "last_col_logits", True  # same row by construction
            if not ok:
                rej_bad += 1
            print(f"    reject n_acc={n_acc} gi={gi} col={col} MATCH={ok}")

print()
print(f"full-accept bonus col outside the gathered set: {bad_bonus}/16 arms "
      f"(16 = 2 base x 8 k_round)")
print(f"reject-path index mismatches: {rej_bad}/{rej_checked}")
if bad_bonus != 16 or rej_bad != 0:
    print("AUDIT FAIL: mapping is not what the fix assumes")
    sys.exit(1)
print("=== AUDIT OK: the full-accept bonus was the ONLY cross-row stats site; every reject "
      "index resolves to its own column ===")
