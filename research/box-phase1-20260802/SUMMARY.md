# box-phase1 — 2026-08-02 (8xH100 p5.48xlarge us-east-2, capacity block cr-02251913b8f9ea0a6)

Phase 1 of `tools/box-aug2-mission.md` on the Aug-2 box (13.58.112.171). Box tree
`~/memra` = BOX-COMMIT **c041f70e** plus the two cross-device fixes below (this lane's
commit carries them; the box's `BOX-COMMIT.txt` is annotated). Raw logs: `receipts/`
(rsync'd from `~/receipts/{m1-pp2,m0-a2a,battery}` on the box — tee'd first, parsed
second). GPU discipline held: GPU 0 bench-only, every run pinned `CUDA_VISIBLE_DEVICES`,
multi-GPU holds under `flock /tmp/gpu-box.lock`. Fabric: full NV18 NVSwitch mesh
(`receipts/m0-a2a/topo.txt`).

## Verdicts

| Item | Verdict |
|---|---|
| M1-FINISH (mission §1b list, final binary) | **DONE — all six gates PASS** |
| M0 a2a re-confirm (§1c) | **CONFIRMED — curve reproduces, M0 verdict stands** |
| Validation battery `--quick` (§1d, GPU 0) | **ALL GATES GREEN** (x3: pristine, fix-1, final) |
| Boundary cost interleaved x5 | seam +1.44%/tick; cross-dev 3.12x (correctness-mode placement, see below) |

## The two cross-device bugs THE gate caught (and their fixes)

`MEMRA_PP_DEVICES=0,1 pp2-gate` ran cross-device for the FIRST time ever on this box (the
5090 rig has one GPU; `devices=0,0` is the same-context degenerate). It failed twice,
loudly, before any perf number — exactly what the gate exists for:

1. **`launch_pdl rms_norm_q8_1: CUDA_ERROR_INVALID_HANDLE`** (`receipts/m1-pp2/q9-dev01.log`).
   `pdl_func`/`pdl_func_flash` cached their duplicate CUmodule/CUfunction handles in
   PROCESS-WIDE statics. CUmodules are context-scoped; the remote stage-1 Engine lives in
   dev1's primary context, got dev0's handles, and every stage-1 `launch_pdl*` died.
   Fix: per-context caches keyed by the engine's CUcontext, module loaded with that
   context current then the caller's context restored (`crates/memra-engine/src/lib.rs`).
2. **`pdl module load qmatvec.fatbin: CUDA_ERROR_ILLEGAL_ADDRESS`**
   (`receipts/m1-pp2/q9-dev01-postfix.log`, after fix 1). A sticky async fault reported at
   the next API call in the poisoned context: stage-1 kernels dereference dev0 weights,
   but every engine allocation is STREAM-ORDERED POOL memory (`cuMemAllocAsync`, device
   default pool — memra-runtime configures it), and `cuCtxEnablePeerAccess` does NOT map
   pool allocations. Fix: `cuMemPoolSetAccess` both ways on the two devices' default
   pools at `Pp2Rt::build` (`crates/memra-engine/src/pp.rs`; the grant is pool-wide and
   covers allocations made before the build — weights load first).

Patch receipts: `receipts/m1-pp2/pdl-perctx.patch` (fix 1 as shipped mid-window),
`receipts/m1-pp2/pp2-crossdev-fixes.patch` (both fixes = this lane's code diff).

## 1b. M1-FINISH gate list (final binary, `*-r3.log`; pre-fix failures kept as evidence)

| Gate | Result |
|---|---|
| pp-transport-smoke | **PASS** — CanAccessPeer=1 on all 56 ordered pairs |
| pp2-gate single-device sanity | **PASS** 48 steps BIT-IDENTICAL |
| `MEMRA_PP_DEVICES=0,1` (THE cross-device gate) | **PASS** 48 steps BIT-IDENTICAL |
| `MEMRA_PP_DEVICES=0,1` split=5 (off-center) | **PASS** 48 steps BIT-IDENTICAL |
| `MEMRA_PP_DEVICES=0,1` + `MEMRA_PP_OVERLAP=1` | **PASS** 48 steps BIT-IDENTICAL |
| `MEMRA_PP_DEVICES=0,7` (non-adjacent pair) | **PASS** 48 steps BIT-IDENTICAL |

**M1 is DECLARED DONE.** (On this NV18 full mesh, 0,7 is topologically equivalent to 0,1 —
the gate still proves the placement plumbing at the far ordinal.)

## Boundary-cost receipt (interleaved x5, `receipts/m1-pp2/boundary/*-r3-*`, boundary-stats.py)

Vehicle: `run-gen` q9 gen-only tok/s, 128 greedy tokens, `MEMRA_QWEN_DC=0` on ALL arms
(the pp2 door lives in eager `decode_step`; dc/graph serving loops are not pp2-wired —
mission-doc placement note — so the three arms share one loop and differ ONLY by pp2
door/placement). Single window, arms interleaved pass-wise, N=5, warmed page cache, idle
box. The 128-token stream is md5-IDENTICAL across all arms and passes (a second
bit-identity witness on top of the gate).

| arm | median tok/s | per-tick cost vs naked |
|---|---|---|
| naked eager (dev0) | 164.21 | — (6.090 ms/tick) |
| pp2 same-device (seam only) | 161.85 | **+1.44% (+88.8 µs/tick)** |
| pp2 `MEMRA_PP_DEVICES=0,1` | 52.57 | +212% (3.12x; +12.93 ms/tick) |

vs M0's 0.3-0.5% prediction: the same-device seam (streams + events + boundary
materialization + copy, no cross-device transport) measures **1.44%/tick — ~3x above the
prediction band**; the prediction priced the boundary TRANSPORT, and the current eager
choreography adds per-step stream/event overhead that M2's pipelined loop (deferred
readback, microbatch slots, graph capture) is designed to hide. The cross-device arm is
**correctness-mode placement**: weights stay on dev0 and stage 1 PEER-READS ~half the
model over NVLink every tick (mission §1b's explicit caveat) — 3.12x is the expected
HBM-vs-NVLink bandwidth signature, NOT a seam verdict. Weight sharding is M2 scope
(`research/m1-inc2-20260801/` next-increment list).

## 1c. M0 a2a re-confirm (`receipts/m0-a2a/`, compare-a2a.py)

New box NVSwitch fabric vs the Jul-31 Mumbai receipts. Median per-a2a µs @64 KiB/peer,
N=5 reps per cell, same committed sources (`research/m0-nccl-20260801/src/`), NCCL 22707
at `/opt/pytorch/cuda/lib/libnccl.so.2` (re-verified; this DLAMI ships no unversioned
`libnccl.so` — link with `-l:libnccl.so.2`, `receipts/m0-a2a/env.txt` has the first
`-lnccl` failure):

| set | Jul-31 Mumbai | Aug-2 use2 box | delta |
|---|---|---|---|
| eager-NCCL a2a n=2 | 10.12 | 10.85 | +7.2% |
| eager-NCCL a2a n=4 | 27.89 | 29.55 | +5.9% |
| eager-NCCL a2a n=8 | 83.89 | 79.94 | -4.7% |
| graph-NCCL a2a n=2 | 19.71 | 21.49 | +9.0% |
| graph-NCCL a2a n=4 | 31.06 | 30.84 | -0.7% |
| graph-NCCL a2a n=8 | 55.78 | 55.38 | -0.7% |

The mission's reference curve (19.7/30.8/55.8 µs at n=2/4/8) is the graph-NCCL set:
n=4/n=8 reproduce within 1%; n=2 +9% (+1.8 µs absolute, different pair — [0,1] here vs
[0,3] Mumbai, both single NVSwitch hops). **Not materially different: the M0 verdict
(EP≤4 + graph-captured a2a mandatory) stands; M2 may lean on it.**

## 1d. Validation battery (GPU 0, `receipts/battery/`)

Three quick-battery runs, each preceded by the in-script rebuild (`.cu` touch defeats
rsync-stale fatbins), q9 = Qwen3.5-9B-Q8_0:

- `validate-q9-quick.log` — pristine c041f70e: **VALIDATE-H100: ALL GATES GREEN**. First
  on-box run of the gate1 fraction rule (decode-batch config mode) + prime-gate line: no
  FAILs.
- `validate-q9-quick-postfix.log` — after fix 1: **ALL GATES GREEN**.
- `validate-q9-quick-r3.log` — final binary (both fixes): **ALL GATES GREEN** — the
  per-context caches and pool grants are single-context no-ops, regated.

## Box state left behind (as of ~14:00Z Aug 2)

- All 8 GPUs idle (0 MiB / 0%); no processes left running by this lane.
- `~/memra` = c041f70e + `pp2-crossdev-fixes.patch` (annotated in `~/memra/BOX-COMMIT.txt`);
  `target/release` binaries built from that state, battery-green.
- `~/receipts/{m1-pp2,m0-a2a,battery}` intact on-box for the phase-3 sweep; driver
  scripts copied into `~/receipts/` for provenance. Nothing evidence-like in `/tmp`.
- `~/m0-nccl/` build dir (commbench binaries + copied sources) — rsync-only tree,
  contents byte-derived from the committed `research/m0-nccl-20260801/src/`.
- Mumbai rsyncs (Hy3 payload + Qwen3.6-35B) still streaming into `/opt/dlami/nvme/models`
  — untouched by this lane; no bandwidth-heavy copies were started.
- `flock /tmp/gpu-box.lock` convention in use; lock free at exit.
