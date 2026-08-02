# M2 ppN — N-stage pipeline parallel on the 8xH100 box (2026-08-02)

Box: darklanes capacity block, p5.48xlarge (8xH100 SXM, NVSwitch all-pairs P2P), window
2026-08-02T11:30Z → 2026-08-03T11:30Z. Repo `~/memra` @ lane/m2-pp8 (source commit
`a70a13c2` + this session's fixes, synced file-level; box tree is not a git checkout).
Model under gate/bench: Qwen3.5-9B-Q8_0 (q9), gemma-4-12b q4_0 (g12) for the serial arm.

**Scope shipped this window:** M2 increments 1–3 — PpNRt N-stage generalization (N up to
8), per-stage weight sharding (`MEMRA_PP_DEVICES` + sharded loader, `MEMRA_PP_SHARD=0`
rollback), deferred-readback pipelining (`decode_step_h_ppn_deferred` → `PendingLogits`).
**Microbatch (batch>1 through the pp pipeline) has no implementation in this lane — there
is no arm to gate; it is the explicit next increment,** not a silent omission.

## Verdict summary (final build, battery 3, 16:30–16:56Z)

`run-m2-gates.sh` (this dir): **28/28 verdict lines PASS, script-detected failures: 0.**

| gate | configs | verdict |
|---|---|---|
| ppn-gate q9 serial | N=2/4/8 × {singledev, dev0..N-1} × {shard, noshard} + asym splits 5,16,27 + split5 + overlap + streams0 | **PASS — BIT-IDENTICAL, 48 steps, all 13 arms** |
| ppn-gate q9 pipelined (deferred, window 3, overlap forced) | same minus streams0 (serial-only by design, note printed) | **PASS — BIT-IDENTICAL, all 11 arms** (see standing flake below) |
| ppn-gate g12 serial (gemma4 arm, N=2) | singledev + dev01 | **PASS — BIT-IDENTICAL** |
| pp2-gate legacy (M1 binary semantics) | singledev + dev01 | **PASS — BIT-IDENTICAL** |
| pp-transport-smoke | 8 devices, all 56 peer pairs `cuDeviceCanAccessPeer=1` | **PASS** |
| kernel-check | full battery | **ALL GREEN: kernels match CPU reference** (0 FAIL lines) |
| run-gen naked | q9 + g12 | **argmax MATCH both** (door shut untouched) |

Representative quoted lines (full logs beside this file):

```
ppn gate PASS [serial]: 48 steps (16 prime + 32 gen) BIT-IDENTICAL logits (n_vocab=248320,
  fence=[0, 4, 8, 12, 16, 20, 24, 28, 32]; stages=8 streams=per-stage overlap=0
  devices=0,1,2,3,4,5,6,7 splits=default(even) shard=per-stage)
ppn gate PASS [pipelined]: 48 steps ... stages=8 ... devices=0,1,2,3,4,5,6,7 ... shard=per-stage
pp-transport-smoke PASS
ALL GREEN: kernels match CPU reference.
```

## What the session-restart state actually was (battery archaeology)

- **Battery 1** (`prefix-run-1355/`, binaries built 13:55): **14 failures.** The previous
  agent's mid-flight source fixes (per-stage `pos_d`, load barrier — found uncommitted in
  the local worktree, file-mtime 14:02) were on disk but **never compiled** before the
  session died. Battery 1 is therefore the *bug evidence* for those fixes: dev01 serial
  1-step graze (`FAIL [serial]: 1/48 ... first @ step 0 idx 79492`), split5 serial
  all-steps with `ref=0.0` (half-built head mirror poisoning step-0 KV), overlap serial
  graze, pipelined FAIL everywhere, and the two streams0 arms aborting with
  `Error: "ppn deferred needs per-stage streams"` before printing any serial verdict.
- **Battery 2** (`prefix-run-1442/`, 14:21 binaries = mid-flight fixes + the streams0 gate
  fix compiled in): **serial ALL PASS** (load barrier + per-stage pos_d confirmed),
  pipelined still FAIL on 8 arms.
- **Battery 3** (final build): all green, above.

## Root-caused fixes landed this session (each with its receipt)

1. **ppn-gate streams0 no-verdict** — the deferred API correctly refuses
   `MEMRA_PP_STREAMS=0`, but the gate binary propagated the error and the *serial* verdict
   was never printed (battery-1 streams0 logs). Fix: skip the pipelined arm with a printed
   NOTE. Battery 2/3: `ppn gate PASS [serial]` + `NOTE: pipelined arm skipped`.
2. **Boundary-slot first-use race** (the deterministic pipelined FAIL class): the slot
   buffer's lazy `alloc_zeros` memset enqueues on the **RX** stage stream, the TX copy on
   the **TX** stream, and on first use `ev_rx` was never recorded — nothing orders them.
   With ≥2 tokens in flight the RX stream is still busy, the memset lands **after** the TX
   copy, and the boundary residual is zeroed. Bisect receipt: window=1 PASS, window≥2 FAIL
   at the slot-1 first-use step; `-overlap` arms passed because the serial arm pre-warmed
   both slots. Fix: one-time `s_rx.synchronize()` per slot allocation (≤2·(N−1) syncs per
   process, all during prime). Post-fix: dev01 pipelined PASS deterministic.
3. **Per-stage Engine isolation on the primary device**: `Engine` owns lazily-grown
   stable-pointer scratch pools (`fa_part_pool`, `argmax_partials`, `fa_vf16_scratch`, …)
   — safe single-stream by design, a data race once two stage streams run concurrently
   through the *same* Engine (singledev placement + deferred readback: token t+1 stage-0
   fa memsets the partials while token t stage-s fa still reads them). Cross-device arms
   were immune because remote stages already had their own Engine — exactly the observed
   battery-2 split. Fix: every stage s>0 builds its own Engine even on the primary device
   (same retained CUcontext, so the per-context CUmodule cache keeps it cheap).
4. **Cache-birth barrier**: with the door open but no device placement, `Cache::new`
   memsets enqueue on the primary worker stream while first KV appends run on per-stage
   streams — unordered. `pp::new_cache` now syncs every stage context after creation
   (same class as the load barrier).

## Standing NEGATIVE (open): single-device pipelined intermittent

After all four fixes, a **x20 same-build soak** of the singledev N=2 pipelined arm
(`soak-singledev/`) lands at **13/20 PASS, 7/20 FAIL (35% flake)** — every failure a
different first-divergence step (6, 11, 13, 14, 16, 25, 28), the classic signature of a
timing-dependent cross-stream race, not a logic bug. Combined with battery 3 + the x5
probe: singledev pipelined = 21/34 clean, cause not yet isolated (lead suspects: device
default-mempool cross-stream reuse, or a remaining Engine-shared surface reachable when
both stage streams live on one device).

The controlled twin: a **x10 dev01 soak** (`soak-dev01/`) on the same build, same window,
same model is **10/10 PASS.** Cross-device pipelined — the configuration multi-GPU PP
exists for — has **0 failures across all post-fix runs** (battery 3 all placements +
dev01 x5 probe + this soak: 18/18). Verdict discipline: the pipelined arm is validated
**cross-device only**; single-device multi-stage keeps a repro'd 35% intermittent and
must not be claimed clean or benched.

## Throughput (final build only; interleaved ×5 at the invocation level)

`run-m2-bench.sh`, P=32 G=128, five outer rounds, arm rows appended per round — raw JSONL
in `bench-q9-*.log`. Baseline on GPU1 (`CUDA_VISIBLE_DEVICES=1`, unsharded load, door
shut); N=2/4 on devices 1.. (GPU0 kept free); N=8 takes all eight. Same-session
denominator; medians of N=5, full-power regime (box otherwise idle per
`gpu-state-pre-bench.txt`).

| config | serial-pp median tok/s | pipelined-pp median tok/s | vs baseline 185.8 |
|---|---|---|---|
| baseline (1 GPU, door shut) | 185.83 [184.49–188.24] | — | 1.00x |
| N=2 sharded (dev 1,2) | 185.39 [184.81–185.85] | **346.87** [345.71–350.60] | 1.00x / **1.87x** |
| N=4 sharded (dev 1–4) | 167.73 [167.18–167.91] | **350.04** [342.23–351.28] | 0.90x / **1.88x** |
| N=8 sharded (dev 0–7) | 165.12 [164.75–165.71] | **332.17** [325.74–333.17] | 0.89x / **1.79x** |
| N=2 noshard (peer-read) | 55.53 | 64.40 | 0.30x / 0.35x |
| N=4 noshard | 42.76 | 51.95 | 0.23x / 0.28x |
| N=8 noshard | 38.65 | 46.18 | 0.21x / 0.25x |

Reads:

- **Deferred readback is the M2 payoff: 1.87–1.88x single-stream decode at N=2/4.** The
  win saturates at N=2 — it comes from tokens-in-flight (window 3 in the bench loop), not
  stage count; N=8 pays more boundary hops for the same window (1.79x).
- **Serial cross-device N=2 is free** (185.4 vs 185.8 — inside the round-to-round band),
  confirming M0's 0.3–0.5% tick prediction at N=2. **N=4/8 serial costs ~10%** — 3–7
  boundary crossings per token with no overlap to hide them. NEGATIVE row, recorded.
- **Peer-read weight placement is a 3–4x cliff** (row 5–7). Sharding (increment 2) is not
  an optimization, it is the difference between a working and a broken multi-GPU config;
  `MEMRA_PP_SHARD=0` stays rollback-only.
- An earlier bench pass (archived `bench-prefix-broken-pipelined/`) ran on a pre-fix build
  whose pipelined arm computed **wrong logits** (e.g. N=2 292.8 tok/s) — those rows are
  excluded from every claim; wrong math is not a data point.

## fa-deep sm_90a validation (v0.67 H100-side evidence, rode this box)

The fa-deep rewrite (`55736e4f`, post-dates this lane's branch point) was staged as a
separate tree (`~/memra-fadeep`) and gated behind the same GPU lock — receipts in
`fadeep-h100/`:

```
fa_decode_v4_deep vs v4 (eager) t_kv=512:  bitdiff=0 OK
fa_decode_v4_deep vs v4 (dc)    t_kv=512  bucket=512/640:   bitdiff=0 OK
fa_decode_v4_deep vs v4 (eager) t_kv=3071: bitdiff=0 OK
fa_decode_v4_deep vs v4 (dc)    t_kv=3071 bucket=3071/3199: bitdiff=0 OK
ALL GREEN: kernels match CPU reference.        (kernel-check, 0 FAIL lines)
prefill argmax=268  decode argmax=268 ... MATCH (run-gen q9 naked)
```

## Files

- `run-m2-gates.sh` / `run-m2-bench.sh` — the drivers (params baked as literals).
- `ppn-*.log`, `pp2-*.log`, `pp-transport-smoke.log`, `kernel-check.log`,
  `run-gen-*.log`, `gates-driver.log` — battery 3 (the shipped verdicts).
- `bench-q9-*.log`, `bench-driver.log` — the ×5 interleaved bench (raw JSONL rows).
- `prefix-run-1355/`, `prefix-run-1442/` — batteries 1–2 (the failure record).
- `bench-prefix-broken-pipelined/` — the excluded pre-fix bench pass.
- `fadeep-h100/` — fa-deep sm_90a gate receipts.
- `build-m2.log`, `build-m2-fix.log` — release builds (sm_90a auto-detect).

## Next increments (explicitly not in this window's code)

1. Microbatch boundary slots (+ graph capture) — the M2 mission item with no lane
   implementation yet; requires batched residual slots and a batched deferred API.
2. Root-cause the singledev pipelined intermittent (mempool cross-stream reuse is the
   lead suspect); until closed, single-device multi-stage stays serial-arm-only guidance.
3. 5090-rig re-gate before any runtime default moves (G7e/H100 evidence ≠ default flip).
