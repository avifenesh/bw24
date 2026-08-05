# k27 rig-divergence — MECHANISM NAMED: the `fa_split_keys` SM rung (lib.rs:477)

Lane: lane/k27-divergence, 2026-08-05/06. The open red from the v0.70.0 release battery
(be85ae00): pod (188-SM RTX PRO 6000 WS) fast-gate k27 argmax diverged from the golden
pinned at d9e45d86 on the 82-SM 5090 Laptop. Artifact byte-identical both rigs
(md5 `0f5bdf74337f6a0b66415b38c979a4ab`, verified again this lane on both).

## Verdict in one paragraph

The divergence is the **`fa_split_keys` big-rig rung** — `fa_sm_count() >= 128`
(crates/memra-engine/src/lib.rs:477). k27 is a qwen35-arch hybrid with `n_head_kv=4`; at
t_kv 96–2048 the 82-SM rig's ladder returns split **8** (kv<=4 branch, t_kv<=512) while a
>=128-SM rig returns split **16**. FA-decode split size changes the split-K partition of
keys, i.e. the f32 combine/reduction segmentation, a ~1e-2 logit perturbation. The probe
prompt is 90 tokens, so generated index 6 is exactly t_kv=96 = `FA_VEC_MIN_TKV` — the first
step the vec split-K kernel runs at all — and the top-2 tokens there sit at a **near-tie**
(margin 0.0373 on the 5090 view / 0.2913 on the pod view, vs a 20-step median margin of
~9.4). The rung flips that coin: 7246 on <128-SM, 5638 on >=128-SM. Everything downstream
is prefix separation. **Verdict class: FLIP-NEARTIE (the documented cross-config drift
class), not a numeric defect** — at matched split the two rigs are logit-identical to every
printed digit. **Both prior hypotheses are dead**: not silicon class, not driver/toolkit
codegen — the same binary produces either stream on either rig depending only on the knob.

## First-div position + margins (mission item 1)

Teacher-forced localizer (`k27div-probe`, in-tree this lane): the 5090 golden tape fed
through each rig's arithmetic — bit-identical inputs at every position, so the flip is
measured at the position the reference visited, not inferred from separated streams.

| arm | first div | margin at flip | 1/20 positions disagree |
|---|---|---|---|
| pod default (sp16), old arith | step 6 (t_kv=96) | 0.2913 toward 5638 | yes — steps 7..19 all agree with the tape |
| 5090 default (sp8), old arith | none | (margin 0.0373 toward 7246 at step 6) | 0/20 |
| pod sp16, NEW arith (2d2618ef) | step 6 | — (see below, direction inverts) | — |

Step-margin distribution on the probe (21 margins): min 0.254 (prefill p1), p50 ~9.37.
The flip site is the *decode* near-tie at step 6, an order of magnitude under p10.

## One-variable kills (mission items 2–4) — all cells N>=2 where load-bearing, md5 receipts

Streams: `dc2eb17c…` = 7246@6 (the d9e45d86 golden, "5090-class");
`0da9c9b9…` = 5638@6 ("pod-class", and — see below — the CURRENT re-pinned golden).

| rig | binary | arith | MEMRA_FA_SPLIT | stream |
|---|---|---|---|---|
| pod | pod-built v0.70.0 | old | default (→16) | 0da9c9b9 (x3 runs, deterministic) |
| pod | pod-built v0.70.0 | old | 32 | 0da9c9b9 |
| pod | pod-built v0.70.0 | old | **8** | **dc2eb17c = golden** (x2) |
| pod | **5090-built** (glibc-shimmed loader) | old | default | 0da9c9b9 — **divergence follows the SM count, not the build/toolkit** |
| pod | 5090-built | old | 8 | dc2eb17c |
| 5090 | local build | old | default (→8) | dc2eb17c |
| 5090 | local build | old | **16** | **0da9c9b9 — the pod stream reproduced on the 5090** |
| pod | v0.70.0, MEMRA_FAST=0 + sp8 | old | 8 | 0da9c9b9 (oracle = Stage-A f32 arithmetic, its own near-tie coin — expected, not a counterexample; the golden was minted on the fast path) |

**Strongest receipt**: at matched split, the teacher-forced logs are **byte-identical
across rigs to every printed digit** (4-decimal logits, 6-decimal margins, 20 steps):
`tf-local-default.log` == `tf-pod-split8-5090build.log` and
`tf-local-split16.log` == `tf-pod-5090build.log` (diff = empty on all step lines).
188 vs 82 SMs, driver 570.211.01 vs 595.84, different build hosts — zero numeric
difference once the split key matches. There is no silicon or toolkit mystery left.

Why the release lane's arms missed it: it pinned MEMRA_FA_SPLIT=**32** — a valid pin, but
on the wrong side; the boundary in play is 8-vs-16 and 32 lands with 16 (0da9c9b9
measured). KS/Q80_G2/PRIME_CHUNK were never in play (NVFP4+Q5_K artifact, no Q8_0 trunk;
T=90 = single chunk), and the FAST=0 oracle changes arithmetic class entirely.

## SM-count-conditional dispatch census (mission item 3, exhaustive)

| site | keyed on | arithmetic-changing? |
|---|---|---|
| `lib.rs:477` fa_split_keys big_rig (>=128) | FA-decode split size | **YES — the mechanism** |
| `lib.rs:6981` Q8_0 g2 grid fill (`4*sm_count`) | grid shape only | no — documented bit-identical twin; and k27 has no Q8_0 trunk tensors |
| `lib.rs:7279+` batched_variant auto rules (`sms`) | variant pick | no — all auto variants bit-identical per (token,row); the k-order-shifting rpks family is banned from auto |
| `decode.rs:2489` graph budget key (>=180) | graph vs eager replay | no — replay is bit-identical (256-step gate); budget=20 < 48 doors it closed here anyway |

The MMQ stream-k autotune (timing-keyed, not SM-keyed) is prefill-only; prefill logits
agreed across rigs (step-0 margins identical), so it is not in play on this probe.

## Interaction with the 2d2618ef re-pin (new chunk-0 arithmetic) — the divergence DOES NOT vanish; it INVERTS

Measured with b1f7b84e-built binaries on both rigs:

| rig | arith | split | stream |
|---|---|---|---|
| 5090 | new | default (8) | 0da9c9b9 = **new golden** (re-pinned @869e0fcc) |
| 5090 | new | 16 | dc2eb17c |
| pod | new | default (16) | **dc2eb17c ≠ new golden** (x2) |
| pod | new | 8 | 0da9c9b9 = new golden |

The chunk-0 fix shifts the logits a few 1e-2 and the near-tie **flips direction** (pod
sp16 TF margin now 0.0141 toward 7246; sp8 0.0087 toward 5638). So the k27 gate on the
pod is **still red** under the new golden — the streams merely swapped roles. The rig
difference does NOT live in the f32-first-chunk path; it lives in the FA-decode split
rung, which the chunkinv-flip did not touch.

## Gate recommendation (mission item 5) — applied this lane

Golden-token probes are cross-rig-portable **only under a pinned FA split**: the goldens
are all minted on the 5090, and any >=128-SM rig runs different FA-decode arithmetic
naked. Fix applied: `MEMRA_FA_SPLIT=8` in the k27 row's extra-env (models.tsv) — 8 is the
82-SM ladder value the goldens (old AND re-pinned) were minted under, byte-identical
behavior on the 5090 (its default at these t_kv is already 8, md5-verified), and verified
this lane to produce the current golden on the pod under the new arithmetic (both the
run-gen stream and the TF probe). This makes k27 green on both rigs for the RIGHT reason:
same arithmetic, same tokens. The exactness contract wording stays honest: "one canonical
greedy output per prompt" holds **per FA-split configuration**; rigs whose SM count
crosses the 128 rung run a different (equally valid, near-tie-class) configuration naked.
Any future probe expected to run on >=128-SM rigs needs the same pin — or per-SM-class
goldens, which is more machinery for the same guarantee.

NOT recommended: flattening the fa_split_keys SM rung itself. It is a measured perf
default (g7e sweeps, +7-11% short-ctx on 188 SM) and the divergence it causes is
near-tie-class; the gate pin isolates exactness from the perf ladder.

## Hashes

- artifact: `0f5bdf74337f6a0b66415b38c979a4ab` (both rigs, re-verified)
- old golden stream (d9e45d86 pin): tokens-line md5 `dc2eb17c5578500318f57071bd7cf8a1`
- pod-divergent / NEW golden stream (869e0fcc pin): `0da9c9b9af970f788aa9818014ab1bd8`
- binaries: pod v0.70.0 run-gen `3dc712268a5aa57f904dfbcd990ec4ca`; 5090-built run-gen
  (ed815eee tree) `3aea6536cdb053c9387c05d22ce6266a`; k27div-probe (c0c02017+probe)
  `0aa0c437bbfb5bda12c9f8f52f317649`; new-arith (b1f7b84e) run-gen
  `26c482ced0b098abb00057dc0b7bebc2`, k27div-probe `5ce52912e841356c827376265ebd5a9b`
- raw logs: `pod/` (rsynced from the pod's research/k27-divergence-20260805/) and
  `local5090/` in this directory. Pod cells are exactness rows (community-pod caveat
  applies to perf only; no perf is claimed here).
