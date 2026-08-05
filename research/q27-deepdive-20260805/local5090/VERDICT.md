# 5090-arbiter gate + transfer verdict — q27 deep-dive phase-1 levers

Rig: RTX 5090 Laptop GPU, **82 SM**, 24GB, platform_profile=balanced, driver mem clock
14001 MHz throughout. GPU serialized via `flock /tmp/gpu5090.lock`. Only co-resident: the
CPU-bound bge embedding llama-server (`-ngl 0`, 332 MiB, the local-ci allowed class).
All logs in this directory (`final-*` = the shipped SM-gated tree; earlier files = the
pre-SM-gate tree that exposed the transfer failure).

## Gate battery on the FINAL tree (SM-gated key + lever 1 + kernel-check m=5/8 arms)

| gate | arm | result | receipt |
|---|---|---|---|
| `kernel-check` FULL naked | final tree | **ALL GREEN** | `final-kc-naked.log` |
| `kernel-check` + q35 weights (Q8-FUSED2/3 + batched m=2,3,4,**5**,**8**) | final tree | **ALL GREEN**, 24 bits=true cells incl. the new fused2_b8 m=5/8 arms, rel=0.00e0 | `final-kc-q35.log` |
| `run-gen` argmax (prefill-vs-decode + batched-prime-vs-tokenwise) | q9 | **MATCH** both | `final-rungen-q9.log` |
| `run-gen` argmax | q27 (NVFP4-MTP, the local 27B form) | **MATCH** both | `final-rungen-q27.log` |
| `run-spec` K=1..3 self-consistency | q9 | **PASS all 3** | `final-runspec-q9-K{1,2,3}.log` |
| `graph-decode-gate` 256 steps | q27 | **BIT-IDENTICAL**, buckets=30, captures=2 | `final-gdg-k27.log` |
| `graph-session-gate` | q27 | **ALL GREEN** | `final-gsg-k27.log` |
| `decode-batch-gate` exactness battery | judge Q8_0 (all-Q8_0 trunk) | **ALL GREEN** (gates 1-3 incl. device sampling + lean-logits identity) | `final-dbg-judge.log` |
| `fast-gate --diff 265ccc3c` (tier 0 FULL kernel-check + tier 1) | final tree | tier 0 GREEN, tier 1 **6/6 PASS** (q9, q35, g12, o35, q35slru, q35spec — golden token-identical) | /tmp/fast-gate-20260805-082645 (goldens pinned in-repo) |

The kernel-check bit-identity arm for the new `qmatvec_q8_0_mmvq_fused2_b8` wrapper was
**already added by the lane** (kernel_check.rs extends the Q8-FUSED2-B loop to m=5/8) — no gap.

## Transfer A/B — the 82-SM verdict (tg128 d512, N=3, interleaved, order alternated, warmup discarded)

### Lever 2 (graph key 256→48): DOES NOT TRANSFER — REFUTED on 82 SM, SM-gate added

q27 NVFP4-MTP, pre-SM-gate binary (naked = graph at n=128 vs forced eager), `ab-k27-*`:

| rep | eager | naked(graph) | Δ |
|---|---|---|---|
| r1 (eager first) | 46.43 | 45.23 | −2.58% |
| r2 (naked first) | 45.83 | 45.11 | −1.57% |
| r3 | 45.86 | 45.12 | −1.61% |
| **median** | **45.86** | **45.12** | **−1.61%** |

3/3 pairs lose, both orderings. Token streams IDENTICAL (md5-equal), so it is pure perf.
Crossover sweep (`cross-k27-*`): graph stays negative at n=256 (median eager 46.11 / graph
45.92... paired −1.07%) and n=512 (45.77 / 45.55, −0.59%) — the crossover on this rig, if it
exists, sits above 512, i.e. the OLD 256 key was already past this rig's measured envelope
and the 48 key would have regressed every naked run ≥48 tokens.

**Resolution:** the key is now SM-gated in decode.rs — `budget >= 48` at `sm_count() >= 180`
(measured: 188-SM PRO 6000), `budget >= 256` otherwise (82-SM refuted; 132-SM H100 and
170-SM desktop 5090 unmeasured at sub-256 budgets keep their shipped key — rig-divergence law).
Post-gate verification (`smgate-k27-*`): naked now tracks eager within noise (r2 46.40/46.12,
r3 45.96/45.90 — thermal-declining set, naked ≥ eager in every pair).

### Lever 1 (Q8_0 dense-FFN gate+up fuse at m=1): FLAT on 82 SM — kept as default (bit-identical, big-rig-positive)

qwen3.5-9B judge all-Q8_0 (the local proxy — the pod's exact 27B-Q8_0 artifact is 26.6 GiB
and does not fit 24GB; same `matmul_pre_dual_noscale` dispatch class). N=5 with graph door
forced off both arms (`f5-*`), order alternated:

| rep | off | on |
|---|---|---|
| r1 | 84.58 | 84.00 |
| r2 | 83.11 | 83.61 |
| r3 | 82.81 | 82.70 |
| r4 | 82.50 | 82.56 |
| r5 | 82.52 | 82.47 |
| **median** | 82.81 | 82.70 |

Paired deltas −0.69/+0.60/−0.13/+0.07/−0.06% — sign-flipping, order-paired mean −0.04%,
inside this laptop board's thermal spread (the set drifts −2.5% monotone run-over-run).
**Verdict: FLAT on 82 SM, +0.94% (5/5 pairs) on 188 SM.** Kept as the default because the
fusion is BIT-IDENTICAL (kernel-check `bits=true`, stream md5s equal across all four arms in
`ab-judge-*`), strictly removes 64 launches/token, wins on big silicon, and costs nothing
here. The q35 flagship (Q8_0-trunk attn_qkv+attn_gate rides the same fused2 family) also
measured flat (183.6-185.2 both arms, `q35-fuse-*`) with stream identity.

### Local same-board denominators (this directory)

- q27 NVFP4-MTP tg128 d512: ~45.9 tok/s eager (thermal band 45.1-46.4 across the session)
- judge-9B Q8_0 tg128 d512: ~82.5-84.6 tok/s (declining thermal band)
- All runs: mem clock pinned 14001, temp 66-80C, profile balanced.

## What ships

- Lever 1 (fuse2 m=1 arm + `MEMRA_Q8_FFN_FUSE2` seam): default ON everywhere (bit-identical,
  big-rig win, small-rig flat).
- Lever 2: SM-gated — 48 key only at >=180 SM; 82-SM naked behavior is byte-for-byte the
  pre-lane default (256 key).
- fused2_b8 wrapper + `matmul_q8_fused2_t` 2..=8 widening + kernel-check m=5/8 arms: gate
  infrastructure, no default dispatch change (lever-3 call site stayed reverted).
