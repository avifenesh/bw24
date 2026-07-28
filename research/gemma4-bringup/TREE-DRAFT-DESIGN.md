# Tree drafting for the gemma MTP loop — design (lane opened 2026-07-28)

## Why

Every spec cell now rides a single linear draft chain: K drafts, verify K+1 rows, accept
the longest prefix. The verify step is batch-row memory-bound — verifying 8 rows costs
barely more than 5 (the b16 tier reaches t=16). The wasted margin is the ACCEPTANCE
CLIFF: one wrong draft kills the whole tail. A tree spends the same verify batch on
multiple continuations at the fork points, so a single wrong guess no longer ends the
round. llama has no MTP-tree either — this is a pull-ahead-everywhere lane, not a parity
chase. EAGLE-2/3 report 1.2-1.5x over linear chains at equal verify budget.

## Where the win lives (measured shape, 2026-07-25..28 data)

- 26B depth: accept 0.826-0.839, tok/round 4.4-4.6 at K=6 — one extra accepted
  token/round ≈ +20% e2e. Deep positions miss most (p3 accept ~35% on 31B chat).
- Verify cost vs width: b16 tier exists; rows-batched kernels (`_b4/_b8/_b16` twins)
  are compiled and gated green. Width 8→12 is nearly free on the memory-bound models.
- The drafter itself is ~1ms of a ~13ms round on 26B (7%) — extra draft branches are
  cheap; the expensive resource is verify WIDTH, which the tree spends better.

## Shape

Fixed-topology tree (static, like EAGLE's frozen tree — no per-round host planning):

```
root ── d0 ── d1 ── d2 ── d3 ── d4          main spine (K deep)
         ├─ a1                              sibling at depth 1 (2nd-best of d0's logits)
         └─ (d1) ├─ b2                      sibling at depth 2
```

- Spine K deep + S sibling branches at the shallowest F fork depths (start: F=2 forks,
  1 sibling each → verify width K+1+S ≤ b16 tier).
- Sibling token = the draft head's SECOND argmax at that step (top-2 from the same
  logits — one extra `argmax2_token_device_col` kernel or a fused top2; no extra trunk
  step for the fork token itself).
- Each sibling then extends 1-2 tokens linearly (its own trunk steps, seeded by the
  sibling token) — the tree is a small set of ROOT-ANCHORED PATHS.

## Verify

The verify batch stays ONE causal batch if rows are ordered by tree path with per-row
POSITION ids (pos = depth) and an attention MASK that lets each row see only its
ancestors. Two options:
1. **Path-duplicated rows (v1, no kernel change):** verify each root path as its own
   contiguous row segment — the KV appended during verify is per-path garbage beyond
   the accepted path, but we already rollback KV to the keep-point every round
   (`spec_rollback` machinery exists). Path rows re-verify shared prefixes (wasted
   rows: shared spine prefix × extra paths). With F=2/S=2 short branches the waste is
   2-3 rows — still inside the b16 budget. ZERO new kernels.
2. **Tree-masked verify (v2):** custom mask in the fa rows kernel — no duplicate rows,
   max width efficiency. New kernel work; only if v1 shows the win.

Accept rule: for each root path, compute its longest-prefix match against the verify
argmaxes of ITS OWN rows; take the best path; commit that path's tokens; rollback the
rest (existing rollback path, keep-point = best path's last accepted row).

## Exactness

Unchanged contract: emitted tokens are verify argmaxes (main-model outputs). The tree
only changes WHICH candidates get verified. Stream agreement 128/128 stays the gate;
the accept rule must be deterministic (tie-break: spine first, then branch order).

## Doors and staging

- `BW24_SPEC_TREE=<S>` (0 = linear, default 0 until measured): number of sibling
  branches. v1 target: S=1..2, fork at depths 1-2, sibling extension 1.
- Stage 1 (v1): top-2 fork at depth 1 only, sibling extends 1 token → width K+3.
  Measure on 26B depth (accept 0.83 → expected tok/round +0.3-0.5) and 31B depth.
- Stage 2: second fork at depth 2, sibling extension 2.
- Stage 3 (only if v1/v2 win): tree-masked verify kernel, wider trees.

## Cost model (26B depth, K=6)

Round today: draft 6×~150µs + verify(7 rows) ≈ 13ms round, 4.5 tok/round.
v1 adds: 1 top-2 kernel (~10µs) + 1 sibling trunk step (~150µs) + verify width 7→9
(memory-bound: +~2-4%). Break-even: the sibling path needs to win ≥ ~0.05 extra
tokens/round. EAGLE data says fork-at-depth-1 recovers 20-40% of first-miss rounds —
at 17% miss rate that's +0.15-0.3 tok/round ≈ +3-6% e2e. Worth building.

## Open questions

- Sibling seeding: h_next for the sibling comes from the SAME trunk step (h_next is
  token-independent? NO — h_next feeds from the chosen token's embedding path; the
  sibling needs its own trunk step seeded with the sibling token → the fork costs one
  extra draft step, already in the cost model).
- Ring/commit: spec_ring_commit takes a linear accepted run — path commit maps onto it
  once the best path's tokens are packed contiguously (host repack before commit, or
  device gather).
- Burst/graph arms: linear-only forever; tree is an eager-arm feature (same as the
  in-round cut).
