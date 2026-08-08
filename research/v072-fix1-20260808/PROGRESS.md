# v0.72 fix 1: restore tickinv35c canary teeth

Train tip: `a131e8c75849bcf1efe1b234039a14d71bb089a3`

Fix commit: `73c65c91b0d7728882b3abb2475e897b3039f6e2`

Verdict: **PASS.** `tickinv35` is bit-exact naked and `tickinv35c` now produces a real
tick-dependent result under `MEMRA_PRIME_CALLLOCAL=1`.

## Fix

Lever A changed step35 SWA prefill from the f32 floor to
`fa_prefill_view_ws_w_hd128`. The old `MEMRA_PRIME_CALLLOCAL` seam still restored the
per-call `seq_end` predicate, but windowed and unwindowed FA are bit-identical where that
predicate can differ, so the canary went inert.

The seam now restores both halves of the pre-fix segmentation arithmetic:

1. Per-call `seq_end = cache.pos + t`, ignoring `queued_after`.
2. The raw, unaligned SWA view offset, restoring the current FA path's
   call-boundary-dependent tile grid.

The naked path is unchanged: request-level `seq_end` plus the BK=32-aligned absolute view
origin.

## Local checks

- `cargo check -p memra-engine --lib`: PASS.
- `cargo test -p memra-engine --lib`: 48 passed, 0 failed, 1 CUDA-only test ignored.
- `git diff --check`: PASS.
- Repo-wide `cargo fmt --all -- --check` remains red on broad pre-existing formatting
  drift; no unrelated formatting changes were applied.

## Verification rig

Box2 (`13.59.112.147`) was still receiving shard 2 at the scheduled `02:45Z` poll
(`11,499,712,512 / 46,999,941,600` bytes), with shard 3 and the Q8 MTP head absent.
Per the mission fallback, verification moved to the pair box once its existing locked
battery completed:

- Host: `18.195.123.14`
- GPUs: 2x NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GB
- CUDA: 13.2, auto-detected `sm_120a`
- Lock: one hold on `/tmp/memra-gpu.lock`, `02:54:07Z` to `03:09:12Z`
- Source: dedicated `~/step37/v072fix-memra`; no existing remote tree was modified

The copied source base was the v0.72 battery tree at `6afc4f65`. Its engine and registered
tick gate hashes match the `a131e8c7` train; the intervening train commits only change
server/fleet/docs surfaces. The fix files were overlaid byte-for-byte:

| file | SHA-256 |
|---|---|
| `crates/memra-engine/src/hybrid_forward.rs` | `47ea7c235f0eda446ca45a7693113469428f7c22ce157bbfcafb31c74a1861f5` |
| `docs/FLAGS.md` | `3e14869a56904ddc4f2715a638c548080128fa330e0e875877cf06ff49c475b4` |
| `tools/tick-invariance-gate.sh` | `0c9f13386a03eb17b4942c69d833509e69aa75d5c0c9aa1579261c31f406fcf0` |

Artifact byte sizes were checked before the lock:

| artifact | bytes |
|---|---:|
| IQ4_XS shard 1 | 46,483,327,296 |
| IQ4_XS shard 2 | 46,999,941,600 |
| IQ4_XS shard 3 | 11,510,293,728 |
| MTP Q8_0 head | 3,707,276,416 |

## Gate results

Invocation is pinned in `verify-box2.sh`: PP-2 on devices `0,1`, prompt
`prompt-pp6257.txt`, budgets `0,1024,513,512,256,64`, splits `64,256,512`, 24 generated
steps, and seam `MEMRA_PRIME_CALLLOCAL`.

### tickinv35 naked

PASS. Every budget and split arm was exact:

| arms | result |
|---|---|
| `1024,513,512,256,64` | `EXACT`, maxdiff `0.000e0`, stream identical |
| `sp64,sp256,sp512` | `EXACT`, maxdiff `0.000e0`, stream identical |

### tickinv35c

PASS. The canary broke the invariant as required:

| arm | first divergent row | maxdiff | stream divergence |
|---|---:|---:|---|
| `1024` | 1024 | `1.310e0` | step 6 |
| `513` | 513 | `1.115e0` | step 3 |
| `512` | 512 | `1.164e0` | step 6 |
| `256` | 512 | `1.164e0` | step 6 |
| `64` | 512 | `1.164e0` | step 6 |
| `sp64` | 4096 | `1.204e0` | step 6 |
| `sp256` | 4096 | `1.533e0` | step 6 |
| `sp512` | 512 | `1.164e0` | step 6 |

This is the required O(1) canary signature at budgets at or below 512. It is not
digit-identical to the retired `1.813e0` floor-vs-FA signature: the live default is now FA,
so the restored mechanism is the FA view tile grid. The raw offset also makes the 1024 and
513 call boundaries variant; that is intentional canary-only behavior. The shipped naked
path remains exact across every arm.

## Raw receipts

- `raw/build.log`
- `raw/verify-20260808T025407Z.log`
- `raw/tickinv35-summary.log`
- `raw/tickinv35-probe-raw.log`
- `raw/tickinv35c-summary.log`
- `raw/tickinv35c-probe-raw.log`
- `raw/SHA256SUMS`
