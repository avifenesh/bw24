# pp2-hardening — PP-2 on 2x RTX PRO 6000 (g7e.12xlarge, 2026-08-06)

Lane `lane/pp2-hardening` off `restructure/public-split` @ **4b4dc7b1**.
Box: g7e.12xlarge SPOT `18.195.123.14`, eu-central-1a, 2x RTX PRO 6000 Blackwell **Server
Edition** 96GB (cc 12.0, 512-bit bus, 102.0 GB reported total), 48 vCPU, 499 GB RAM,
387 GB root (339 free), 250 GB /dev/shm. Driver 595.71.05, CUDA 13.2.
Box tree `~/memra` = BOX-COMMIT.txt `4b4dc7b1`, file-level rsync (box is not a git checkout).
All receipts under `logs/`, rsync'd from `~/receipts/pp2` on the box; **committed locally, never
from the box.**

Owner order 2026-08-06: *"p2p is a nececary lane"* — PP-2 serving bill in scope, launch in 21 days.

---

## Phase 0 — provision + first receipt: DONE

### Provisioning notes (for the next box of this family)

- DLAMI `Deep Learning Base OSS Nvidia Driver GPU AMI Ubuntu 24.04 20260724` **does** ship
  CUDA toolkits — 12.8, 12.9, 13.0, **13.2** (`/usr/local/cuda -> cuda-13.2`). The
  "no toolkit" expectation from prior boxes of this family is stale for this AMI date.
- **`crates/memra-engine/build.rs:61` hardcodes `/usr/local/cuda-13.1/bin/nvcc`** as the
  default. On this box that path does not exist and the build dies with
  `panicked at build.rs:126: spawn nvcc: Os { code: 2, kind: NotFound }` — *not* an obvious
  "toolkit missing" message. Workaround used: `MEMRA_NVCC=/usr/local/cuda-13.2/bin/nvcc`.
  (Fix candidate for a later lane: resolve `nvcc` from `PATH`/`CUDA_HOME` before falling back
  to a pinned version. Out of scope here — no code change without a gate.)
- `cargo build --release --bins`: **3m58s**, arch auto-detected `120a` (compute_cap 12.0).
  0 errors. Toolchain: rustup stable (minimal), `build-essential pkg-config libssl-dev cmake`.

### THE P2P TRANSPORT RECEIPT (`logs/p2p-probe.log`, `logs/topo.txt`, `logs/pcie.csv`)

Topology: **`GPU0 <-> GPU1 = PIX`** — at most a single PCIe bridge, same NUMA node (0),
CPU affinity 0-47 both. The best possible non-NVLink arrangement. `lspci -tv` shows both
cards under one bridge chain (`0000:01 -> 00.0 -> [02-23]`).

**P2P is ON, natively, on the stock driver — both directions:**

```
canAccessPeer 0->1 = 1
canAccessPeer 1->0 = 1
peer_enabled=1
```

This closes the open question in `research/model-192gb-20260806/ASSESSMENT.md` §0 for THIS
silicon (previously inferred from the 5090-validation NOTE's PRO 6000 datapoint on driver
580.95 — now confirmed first-hand on 595.71.05, Server Edition, and measured).

`p2pBandwidthLatencyTest`-class numbers, `cudaMemcpyPeerAsync` vs a host-staged pinned
bounce (D2H then H2D — what a non-P2P PP boundary pays). Bandwidth iters 2000 (<1MB) /
300 (<16MB) / 40 (bulk); latency = mean of 5000 enqueue+sync rounds:

| payload | P2P uni GB/s | P2P bidir GB/s | host-bounce GB/s | P2P advantage |
|---|---|---|---|---|
| 4 KB | 3.24 | 3.10 | 0.276 | **11.7x** |
| **12 KB** (122B boundary) | **10.21** | 9.39 | **0.751** | **13.6x** |
| **16 KB** (q27 n_embd 4096 f32) | **13.35** | 12.66 | **0.954** | **14.0x** |
| 64 KB | 45.76 | 49.55 | 3.86 | 11.9x |
| 256 KB | 53.83 | 105.45 | 11.02 | 4.9x |
| 1 MB | 56.02 | 107.69 | 20.06 | 2.8x |
| 4 MB | 56.58 | 108.12 | 25.60 | 2.2x |
| 16 MB | 56.23 | 107.13 | 27.50 | 2.0x |
| 64 MB | 55.48 | 106.08 | 28.00 | 2.0x |
| 256 MB | 54.53 | 104.60 | 28.15 | 1.9x |

One-way latency (single copy, enqueue+sync):

| payload | P2P us | host-bounce us | P2P advantage |
|---|---|---|---|
| 4 KB | 6.56 | 14.53 | 2.2x |
| 12 KB | 6.83 | 16.43 | 2.4x |
| 16 KB | 7.02 | 17.22 | 2.5x |
| 64 KB | 7.65 | 16.83 | 2.2x |

**Reading of the transport verdict:**

1. **P2P saturates at ~56 GB/s uni / ~107 GB/s bidir.** Full duplex works (bidir ≈ 1.9x uni),
   so the link is not shared-half-duplex. ~56 GB/s is PCIe **Gen5 x16**-class (~63 GB/s
   theoretical); the pair reaches ~89% of theoretical. Note `pcie.link.gen.current = 1` in
   `logs/pcie.csv` and `LnkSta: Speed 2.5GT/s (downgraded)` in `logs/acs.txt` — that is
   **idle-state power management** (both cards were at P8/30W when sampled), not a real
   downgrade: `LnkCap` reports `Speed 32GT/s, Width x16` and the measured 56 GB/s could not
   happen on a genuine Gen1 link. Recording this so nobody re-derives a false "the box is
   PCIe-gen-capped" conclusion from an idle `nvidia-smi` sample.
2. **At PP-boundary payload the win is not 2x, it is 13-14x.** The engine's boundary is
   `[n_embd] f32` per token — 16 KB for q27 (n_embd 4096), 12 KB for the 122B. At exactly
   those sizes bounce delivers 0.75-0.95 GB/s vs P2P's 10-13 GB/s. Small copies are
   latency-dominated and the bounce pays *two* sequential transfers plus two syncs.
3. **Per-token boundary cost, absolute:** a 16 KB P2P boundary copy is ~7.0 us one-way; the
   bounce is ~17.2 us. At a 12 ms/token decode tick (q27-class c=1) that is 0.06% vs 0.14%
   — both noise. The bounce penalty only becomes structural under **microbatch/batched
   decode**, where payload scales with batch (c=32 x 16 KB = 512 KB → 105 GB/s P2P vs
   11 GB/s bounce, a 9.6x gap on a per-tick-serial copy) — i.e. the transport arm matters
   for exactly the door that is unwired (batched decode over PP).
4. **Consequence for the bill:** `pp.rs` already ships `cudaMemcpyPeerAsync` as the
   cross-device boundary and grants peer + `cuMemPoolSetAccess` both ways at build. On this
   pair that path is **live and correct by construction** — there is no host-staged fallback
   to enable/wire, and no work item here. The ASSESSMENT's "transport is NOT the problem on
   this pair" is now measured, not projected. **Phase-2 item 1 is therefore CLOSED as
   already-shipped**, and the bill's weight moves entirely to the serving loops.

### Probe source

`logs/p2p_probe.cu` — kept beside the log so the numbers are reproducible. Built
`nvcc -O3 -arch=sm_120`. (Patched once at bring-up: `cudaDeviceProp::memoryClockRate` was
removed in CUDA 13.x — dropped the derived theoretical-BW print.)

---

## Phase 1 — PP-2 gate battery on target silicon: **GREEN, 0 failures**

Vehicle: Qwen3.5-9B-NVFP4-MTP (5.7 GB, n_layers 32, n_vocab 248320) — the same model class
the M1 `pp2-gate` and M2 `ppn-gate` receipts were minted on, so this is a same-vehicle
cross-rig comparison. Staged to `/scratch-models` on the box's local root NVMe.
Driver: `run-pp2-gates.sh` (committed here), receipts `logs/gates/`.

**This is the FIRST time the PP gates have ever run on an RTX PRO 6000 pair.** Prior
receipts: M1/M2 on 8xH100 NVSwitch, single-GPU degenerate on the 5090. The 192 GB
assessment named this as mandatory-before-listing. It is now done.

| Gate | Config | Verdict |
|---|---|---|
| `kernel-check` (full battery) | naked | **ALL GREEN — kernels match CPU reference**, 0 FAIL lines |
| `pp-transport-smoke` | 2 devices | **PASS** — canAccessPeer 1/1, peer-arm bytediff=0, 4 boundary roundtrips bytediff=0 |
| `ppn-gate` N=2 singledev | serial | **PASS BIT-IDENTICAL** 48 steps (pipelined correctly skipped — quarantine NOTE fires) |
| `ppn-gate` N=2 **dev01** | serial + pipelined | **PASS BIT-IDENTICAL** both arms |
| `ppn-gate` N=2 **dev10** (reversed placement) | serial + pipelined | **PASS BIT-IDENTICAL** both arms |
| `ppn-gate` N=2 dev01 `SHARD=0` | serial + pipelined | **PASS BIT-IDENTICAL** both arms (bring-up peer-read placement) |
| `ppn-gate` N=2 dev01 `OVERLAP=1` | serial + pipelined | **PASS BIT-IDENTICAL** both arms |
| `ppn-gate` N=2 dev01 `SPLITS=5` (off-center) | serial + pipelined | **PASS BIT-IDENTICAL** both arms |
| `ppn-gate` N=2 `STREAMS=0` (inc-1 seam) | serial | **PASS BIT-IDENTICAL** (pipelined skipped by design) |
| `ppn-gate` N=4 `dev=0,1,0,1` | serial | **PASS BIT-IDENTICAL** fence [0,8,16,24,32] |
| `ppn-gate` N=4 `dev=0,0,1,1` | serial | **PASS BIT-IDENTICAL** |
| `pp2-gate` legacy singledev | M1 semantics | **PASS BIT-IDENTICAL** |
| `pp2-gate` legacy dev01 | M1 semantics | **PASS BIT-IDENTICAL** |
| `run-gen` naked argmax (door shut) | q9 | **MATCH** (`prefill argmax=268 decode argmax=268`) |

`script-detected failures: 0`.

Notes worth keeping:
- **Exactness vs single-GPU is proven the strong way, not the weak way.** `ppn-gate`'s
  method is not "PP-2 output looks like 1-GPU output" — it records full `n_vocab` f32 logits
  per step with the door OFF, then replays the identical token stream with the door ON and
  compares every f32 *bit*. 48 steps x 248,320 logits, zero differing bits, on every arm.
- **Both placement orders pass.** `dev01` and `dev10` are not symmetric in the engine (stage
  0 owns the embed table + `pos_d`, the last stage owns `output_norm` + lm head, and the
  primary engine's device is a third variable). `dev10` had never been gated; it passes.
- **N=4 over 2 cards works on the serial arm** — the 2-stages-per-card shape a deeper split
  on a pair would take. Its pipelined arm is correctly refused by the same-device quarantine
  (2+ stage streams on one device), which is the right behavior, not a gap in this shape.
- The quarantine plumbing behaves exactly as designed on new silicon: the NOTE fires on
  `singledev`, `dev0101`, and `dev0011`; no arm silently ran an unsound placement.

### Flake-arm repro on P2P silicon — **the flake does NOT reproduce on this pair**

Both quarantines in `research/m2-pp8-20260802/RESULTS.md` were minted on **8xH100 SXM /
NVSwitch** (the 5090 rig has one GPU, so `devices=0,0` there is the degenerate case). Two
questions, both answered on-target. Drivers `run-pp2-soak.sh` + `run-pp2-forced.sh`,
receipts `logs/soak/`, one log per run.

| Soak arm | Placement | H100 record | **PRO 6000 pair result** |
|---|---|---|---|
| cross-device pipelined | `dev01`, x40 + x40 (two batches) | ~1 FAIL in ~190 (~0.5%), root cause open | **80/80 PASS, 0 FAIL** |
| cross-device serial | `dev01`, x40 | clean | **40/40 PASS, 0 FAIL** |
| same-device serial | singledev, x20 | clean | **20/20 PASS**, quarantine NOTE 20/20 |
| **same-device pipelined, FORCED** | `MEMRA_PP_FORCE_SAME_DEV_PIPELINED=1`, x20 | **13/20 PASS, 7/20 FAIL (35% flake)** | **20/20 PASS, 0 FAIL** |

**The forced-arm row is the headline.** The 35%-flake placement — 2+ stage streams on one
device, refused by default since 2026-08-02 — ran **20/20 bit-identical** here, serial and
pipelined both. Under the H100-measured p=0.35 failure rate, 20 consecutive clean runs has
probability `0.65^20 = 2.1e-4`. So the H100 failure *rate* is **statistically refuted on
this silicon** (p < 0.001); this is not "we got lucky."

What that does and does not license:

- **It does NOT unquarantine anything.** The root cause named in the M2 write-up is real and
  code-visible: `Engine` owns lazily-grown stable-pointer scratch pools (`fa_part_pool`,
  `argmax_partials`, `fa_vf16_scratch`) that are single-stream-safe by design, and the
  same-device pipelined placement runs two stage streams through concurrently. A race whose
  window closed on different silicon is still a race — clocks, kernel durations, and SM
  counts differ, and this box's cards are *idle-clocked at P8 between runs*. Absence of
  failure in 20 runs is **not** absence of the unsound surface. Wrong-logits-fast stays
  un-shipped.
- **It DOES change the risk picture for the arm this lane actually needs.** The cross-device
  pipelined arm — one device per stage, which is exactly the PP-2-on-a-pair serving shape —
  now carries **40/40 clean on target silicon** on top of the H100's ~189/190. The single
  H100 cross-device flake remains recorded evidence that the mechanism is reachable
  cross-device at low probability; nothing here refutes it (80 runs cannot see a 0.5% event
  with confidence — P(0 failures | p=0.005, n=80) = 0.67, so this soak is simply not
  powered to detect it). **Honest statement: the cross-device flake is neither reproduced
  nor excluded; the same-device 35% rate IS excluded.**
- **Consequence for the bill:** the deferred-pipelined arm (the 1.87x prize) is still not
  serving-default-cleared, and this lane does not clear it. But the blocker is now
  explicitly a *root-cause* requirement (fix the shared-Engine scratch surface, or prove per
  stage-stream isolation), not "we don't know if it happens on our hardware."

### Why the flake window likely differs here (mechanism note, not a claim)

The M2 bisect found `MEMRA_PDL=0` went 20/20 clean and PDL narrowed but did not eliminate
the window — i.e. programmatic dependent launch widens the race. PDL's early-launch overlap
scales with how long a kernel's tail leaves the SMs partly idle, and both the SM count
(H100 SXM 132 vs PRO 6000's Blackwell config) and the per-kernel durations at q9's shape
differ. A narrower window on this silicon is consistent with the recorded mechanism. Not
measured here; recorded so nobody reads "20/20 clean" as "the bug was H100-specific."

---

## Phase 2 — the serving bill

### Item 1: P2P transport arm — **CLOSED, already shipped** (Phase 0 receipt)

The task brief asked to "wire/enable the direct-copy arm in pp.rs where the host-staged path
exists." **There is no host-staged path in `pp.rs` to replace.** `PpNRt::tx` (pp.rs:569)
selects transport per boundary: `memcpy_dtod` when the boundary is same-device, and
`cudarc::driver::result::memcpy_peer_async` with explicit src/dst contexts when it crosses.
Peer access + `cuMemPoolSetAccess` are granted between every distinct device pair at
`PpNRt::build`. On this pair `canAccessPeer` is 1 both ways, so the cross arm is live and is
what all 13 Phase-1 gates and all 80 cross-device soak runs exercised. Measured advantage
over the bounce a non-P2P box would need: **13-14x at boundary payload** (Phase 0 table).
No work item. Note for the record: the host-staged bounce in issue-#67 class was a *5090*
problem, and the correct conclusion is that it never needs building for owned PRO 6000
hardware.

### Item 2: batched decode over PP-2 — **the real finding: the door is not unwired, it is
### SILENTLY WRONG-SHAPED (fails open, not closed)**

The assessment and the `pp.rs` header both say batch/dc/graph/spec are "unwired" and that
`warn_unwired_once` fires. **Both are wrong about the batch path.** Ground truth from the
source:

- `warn_unwired_once` has exactly **two** call sites, and both are gemma4-specific:
  `decode.rs:615` (gemma4-e4b eager) and `hybrid_forward.rs:6462` (gemma4 eager N>2).
  **Neither is on the batched path.**
- `decode_step_batch`'s only pp-awareness is `decode_batch.rs:361` — `pp_cuts(...).is_none()`
  as one condition among six gating the **B=1 fast path**. Its effect is the opposite of a
  guard: with the pp door open it *disables* the B=1 fast path and falls through into the
  batched body, which then runs the **entire trunk** (`for (il, layer) in
  self.layers.iter().enumerate()`, decode_batch.rs:514, `lo=0..n_layers`) on the **primary
  engine's stream**, with no stage split, no boundary, and no `rt.enter()`.
- Under `MEMRA_PP_DEVICES=0,1` with sharding on (the default), stage 1's weights **are on
  dev1**. So the batched body dereferences dev1 weights from a dev0 kernel — legal, because
  `PpNRt::build` granted peer + pool access — and every projection for the back half of the
  model streams over PCIe per step.

**Probe result (`logs/batchprobe/`, `run-pp2-batchprobe.sh`): every arm PASSES.**

| `decode-batch-gate` arm (q9, 32 steps, --mode config) | gate1 | gate2 | gate3 |
|---|---|---|---|
| door SHUT B=4 (baseline) | PASS (0/6 early draws) | PASS | PASS |
| door OPEN `stages=2` singledev B=4 | PASS | PASS | PASS |
| **door OPEN `stages=2` dev01 SHARDED B=4** | **PASS** | **PASS** | **PASS** |
| door OPEN dev01 B=1 | PASS | PASS | PASS |

**This is the dangerous answer.** Peer reads return identical bytes, so exactness is
genuinely preserved — the gates are not lying, and there is no correctness bug to fix. But
that means:

1. **Nothing fails, warns, or refuses.** A serving deployment that opens the pp door and
   serves batched traffic gets *silently* the worst of both worlds: half the weights read
   over PCIe every step, zero pipeline parallelism, and a green exactness battery. The
   quarantine discipline that protects the pipelined arm (a loud `Err` with a full
   explanation, decode.rs:977) has **no counterpart on the batch path**.
2. **The docs understate the bill in one direction and overstate it in another.** Batched
   decode over PP-2 is not "an arm that doesn't exist yet and errors if you try" — it is an
   arm that *runs*, passes gates, and quietly costs performance. That is worse than unwired.
3. **The fix has two halves and they are not the same size.** (a) A refusal — make
   `decode_step_batch` fail closed under an open pp door with device placement, matching the
   pipelined arm's precedent. Small, mechanical, gate-able today. (b) The actual wiring —
   stage-split the batched trunk. That is the weeks-class item, and it is real work:
   `decode_step_batch`'s body is a single 250-line `lo=0..n_layers` loop over a
   pointer-table (`lin_base`/`attn_base`, built once for all B across all layers,
   decode_batch.rs:447-500) plus a batched `pos_d` vector; splitting it means per-stage
   pointer tables, per-stage `pos_d`, an `[B, n_embd]` boundary (16 KB x B — where the
   Phase-0 bidir 105 GB/s number starts to matter), and per-stage engines threaded through
   ~15 `e.` call sites. The eager arm got a clean `decode_layers_eager(e, x, lo, hi, ...)`
   seam to reuse; **the batched body has no equivalent `lo..hi` helper** — that extraction is
   the first increment of any real wiring.

#### Cost of the silent peer-read — MEASURED

`decode-batch-bench`, q9, 32 prompt / 128 gen, four arms **interleaved rep-major N=5**
(driver `run-pp2-batchcost.sh`, receipts `logs/batchcost/`, one log per rep per arm).
Medians, aggregate tok/s across the batch:

| arm | B=1 | B=4 | B=8 | vs door-shut |
|---|---|---|---|---|
| A door SHUT (single-GPU baseline) | 208.9 | 491.3 | 657.0 | 1.00x |
| B door OPEN `stages=2` singledev | 178.5 | 491.2 | 655.7 | 0.854x / 1.000x / 0.998x |
| **C door OPEN `stages=2` dev01 SHARDED** | **7.4** | **29.8** | **47.4** | **0.035x / 0.061x / 0.072x** |
| D door OPEN `stages=2` dev01 `SHARD=0` | 178.5 | 491.1 | 656.6 | 0.854x / 1.000x / 0.999x |

Run-to-run spread ≤0.63% on every cell (most ≤0.35%), so these are not noisy numbers.

Three reads, all of which the refusal recommendation rests on:

1. **The real serving shape (arm C) costs 28x at B=1 and 14x at B=8.** 7.4 tok/s aggregate
   for a 9B model on two PRO 6000s. Every projection for layers 32..63 is fetched over
   PCIe *per batched step*, and there is no pipeline overlap to hide it because the trunk
   runs as one unsplit loop on the primary stream.
2. **Arm D isolates the cause exactly.** Same open door, same two devices, only
   `MEMRA_PP_SHARD=0` (all weights home on the primary) — and it is bit-for-bit the same
   throughput as arm B/singledev (178.5 / 491.1 / 656.6). So the entire 28x is the
   **peer read of stage-1 weights**, not the door, not the placement plumbing, not the
   `pp::new_cache` stage-owned KV. This matches M2's H100 finding that unsharded
   peer-read weight placement is a 3-4x cliff — on this pair, at batch, it is a 14-28x
   cliff, because PCIe Gen5 x16 (56 GB/s) against 96 GB HBM bandwidth is a far worse
   ratio than NVLink was.
3. **The `-15%` at B=1 in arms B/D is the lost fast path, and it is the door's only
   *visible* effect.** 178.5 vs 208.9 = the `decode_step_b1_fast` bypass that
   `decode_batch.rs:361` disables. At B>=4 it vanishes (491.2 vs 491.3) because the
   batched body is what runs in both cases. So a well-placed operator who opens the door
   with `SHARD=0` sees only a 15% B=1 regression and nothing else — no warning that PP
   is not happening at all.

**Recommendation, now with numbers: `decode_step_batch` must fail closed under an open pp
door with multi-device placement.** The pipelined arm refuses its unsound config with a
full `Err` (decode.rs:977) plus a measurement override; the batch path has no counterpart
and silently delivers 3.5% of baseline. Shipped as a code change this lane — see Item 4.

### Item 3: the eager PP-2 throughput story on a PRO 6000 pair — **the 1.87x transfers, and it is 1.91x here**

`ppn-bench`, 32 prompt / 128 gen, N=5 reps interleaved rep-major in-process. Two
invocations per model because **the pp door is a load-time decision** (a sharded load
timed as a "baseline" would time peer-reads — the bench's own header law): invocation 1 =
door shut, unsharded, `serial-off`; invocation 2 = `MEMRA_PP_STAGES=2
MEMRA_PP_DEVICES=0,1`, which interleaves `serial-pp` and `pipelined-pp`. Both under
`flock /tmp/memra-gpu.lock`. Driver `run-pp2-eagerbench.sh`, receipts `logs/eagerbench/`.
Cards went 180 MHz/30 W idle to 2400/2317 MHz, 266/291 W (`gpu-pre.csv`/`gpu-post.csv`) —
full-power regime, box otherwise empty of compute apps.

| model | baseline (door shut, 1 GPU) | serial-pp dev01 | pipelined-pp dev01 | serial vs base | **pipelined vs base** |
|---|---|---|---|---|---|
| q9 (Qwen3.5-9B-NVFP4-MTP, 32L, 5.7 GB) | 211.77 tok/s | 210.10 | **378.18** | 0.992x | **1.786x** |
| q27 (Qwen3.6-27B-NVFP4-Q4_K_M-mtp, 64L, 15.7 GB) | 76.03 tok/s | 75.73 | **144.82** | 0.996x | **1.905x** |

Spread is tight (q27 pipelined: all 5 reps inside 0.5 us/tok, 0.007%). Fences: q9
`[0,16,32]`, q27 `[0,32,64]`. Both door-open logs carry the cross-device transport line
(`cudaMemcpyPeerAsync per cross boundary; ... weight home: per-stage (sharded loader)`).

Reads:

1. **The M2 prize transfers to PCIe P2P silicon, and improves with model depth.** H100
   SXM/NVSwitch measured 1.87x at N=2 on q9; this pair does 1.786x on q9 and **1.905x on
   q27**. The *bigger* model winning more is the expected shape — the boundary copy is a
   fixed ~7 us against a per-stage compute half that grows with layer count, so the
   overlap fraction improves. It also confirms the Phase-0 verdict under load: at these
   payloads the interconnect is not the limiter, and NVLink's absence costs ~nothing here.
2. **Serial cross-device PP-2 is free (0.99x both models)** — reproducing M2's H100 result
   and M0's 0.3-0.5%/tick prediction on new silicon. So the pure *capacity* use of PP-2
   (fit a model that does not fit one card) costs ~0.4% of single-card decode. That is the
   number the 192 GB listing decision needs, now measured on the target pair.
3. **Ceiling check.** 1.905x is near the structural max for a 2-stage split with
   tokens-in-flight 3 (each stage busy ~half the serial tick), so little is left in *this*
   mechanism at N=2 — and M2 already showed N=4/8 do not add single-stream speed.

**The caveat that governs how this number may be used** (recorded because the M2 write-up
does not state it and the 192 GB assessment repeats "1.87x" unqualified): the deferred
window is measured on a **pre-recorded greedy token stream**. `ppn-bench` and `ppn-gate`
both replay an `inputs` vector captured from a door-off reference run, so step t+1 is
enqueued before step t's logits are read back. **Plain autoregressive serving cannot do
that** — it needs token t to pick token t+1. So 1.905x is not a free speedup for
single-stream greedy serving; it is the pipeline's throughput *when something supplies the
next token early*. What legitimately supplies it: **speculative/MTP decode** (draft tokens
are known ahead, and the artifact already carries an MTP head), **batched/multi-sequence
serving** (independent sequences fill the window), and prefill. Single-stream greedy gets
the *serial* 0.996x row, not the pipelined row. This does not weaken the result — it names
which serving modes monetize it, and both are the remaining bill items. It does mean the
deferred arm's value is **gated behind spec-over-PP2 or batch-over-PP2**, on top of its
existing root-cause quarantine.

---

## Cohabitation with the step37-p2 lane (coordinator directive 2026-08-06)

The box is shared with the Step-3.7-Flash bring-up lane (owner doctrine: 3.7-Flash serves on
this pair over PP-2). Rules honored on my side:

- **Tree isolation:** my work is confined to `~/memra`, `~/receipts/pp2`, and
  `/scratch-models`. `~/step37` is not touched.
- **GPU lock:** every measurement window from the eagerbench onward runs under
  `flock /tmp/memra-gpu.lock`. The reason is not courtesy — an interleaved A/B whose arms
  straddle another lane's model load is a cross-run comparison, which this repo's H100 LAWS
  forbid outright. `run-pp2-eagerbench.sh` carries the wrapper plus an in-script comment so
  the next hand does not drop it.
- **Contention actually observed: none.** `nvidia-smi --query-compute-apps` was empty before
  and after every window in this report, and the cards idled at P8/30 W immediately before
  the eagerbench. Every receipt here is window-clean.
- **Disk:** root went 74 GB to **128 GB of 387 GB used (34%, 259 GB free)** while `~/step37`
  grew to 54 GB en route to ~105 GB. Headroom is fine for both lanes (my models total 21 GB
  and are already staged), but a *third* lane staging a 100 GB-class artifact would not fit.
  Flagged per directive; not blocking.

---

## Item 4: the refusal — **SHIPPED with a gate** (code change, not a recommendation)

`decode_step_batch` now **fails closed** when the ppN door is open across 2+ distinct
devices with the sharded loader on. Three pieces:

- **`pp.rs::pp_sharded_cross_device()`** — env-only predicate (open door AND `MEMRA_PP_SHARD`
  not 0 AND `MEMRA_PP_DEVICES` naming 2+ distinct devices), i.e. exactly "some layers' weights
  live on a device other than the primary". Deliberately NOT the same predicate as
  `pp_multi_stream_same_device()`: that one guards a *race*, this one guards a *cliff*, and
  their true-sets are near-complements (`0,0` is unsafe for the pipelined arm and fine here;
  `0,1` is the reverse).
- **`decode_batch.rs`** — refusal at the top of `decode_step_batch_sampled_lean_masked`, with
  the measured numbers, the three escapes, and the receipt path inline in the `Err` string, so
  an operator who trips it does not need to find this file.
  `MEMRA_PP_ALLOW_UNSPLIT_BATCH=1` overrides for measurement (the batchcost arm above must
  stay reproducible).
- **`pp.rs` module header + `docs/FLAGS.md`** — the "`warn_unwired_once` fires" claim for
  batch/dc/graph/spec is corrected in place, since that sentence is what made the fail-open
  behavior look accounted-for.

### Gate battery for the refusal (`logs/refusal/`, driver `run-pp2-refusal.sh`)

Both halves gated: it fires where it must, and it does not fire anywhere else.

| # | Config | Expected | Result |
|---|---|---|---|
| 1 | door SHUT | PASS | **gate1+2+3 PASS**, 0/6 early draws |
| 2 | door OPEN `stages=2`, no placement (singledev) | PASS — not cross-device | **gate1+2+3 PASS** |
| 3 | door OPEN `stages=2 devices=0,1` SHARDED | **REFUSE** | **exit=1, full `Err` text** |
| 4 | #3 + `MEMRA_PP_SHARD=0` (documented escape) | PASS | **gate1+2+3 PASS** |
| 5 | #3 + `MEMRA_PP_ALLOW_UNSPLIT_BATCH=1` (override) | PASS | **gate1+2+3 PASS** |
| 6 | door OPEN `devices=0,0` (repeated device) | PASS — same-device, nothing remote | **gate1+2+3 PASS** |
| 7 | regression: `ppn-gate` N=2 dev01 (the eager pp arm) | untouched | **PASS BIT-IDENTICAL both arms**, 48 steps |
| 8 | regression: `kernel-check` | untouched | **ALL GREEN**, 0 FAIL lines |

Arm 6 is the one that would catch an over-broad predicate (`0,0` names two devices textually
but places nothing remotely); arm 5 keeps this lane's own measurement reproducible; arm 7
proves the change did not leak into the arm that actually works.

**What this does and does not fix.** It converts a silent 28x into a loud error with three
named escapes — that is the whole claim. It does **not** make batched decode work over PP-2;
that remains the weeks-class item below. The value is that a serving deployment can no longer
reach 3.5% of baseline with a green battery, which is the failure mode this box was rented to
find.

---

## What is left of the PP-2 serving bill

Ordered by what a serving deployment on this pair actually needs:

1. **Stage-split the batched trunk** (weeks-class, the real item). `decode_step_batch`'s body
   is one `lo=0..n_layers` loop over pointer tables (`lin_base`/`attn_base`, built once for all
   B across all layers, `decode_batch.rs:447-500`) plus a batched `pos_d`. Splitting it needs
   per-stage pointer tables, per-stage `pos_d`, an `[B, n_embd]` boundary (16 KB x B — where
   Phase 0's 105 GB/s bidir number starts to matter, and where the P2P-vs-bounce gap becomes
   structural rather than noise), and per-stage engines threaded through ~15 `e.` call sites.
   **First increment: extract a `decode_batch_layers(e, .., lo, hi, ..)` seam.** The eager arm
   had `decode_layers_eager` to reuse; the batched body has no equivalent, and every later
   increment depends on that extraction.
2. **Root-cause the shared-Engine scratch race** — the only thing between the deferred arm and
   a serving default. Named surface: `Engine`'s lazily-grown stable-pointer pools
   (`fa_part_pool`, `argmax_partials`, `fa_vf16_scratch`) are single-stream-safe by design and
   the pipelined arm runs 2+ stage streams through them. Either isolate per stage-stream or
   prove they cannot alias. This lane refuted the H100 *rate* (20/20 forced-clean, p<0.001) but
   an unsound surface is not a fixed one, and 80 runs cannot see a 0.5% event.
3. **Spec/MTP over PP-2** — brief, not built (Item 5 below). This is the arm that *monetizes*
   the 1.905x, because draft tokens are the legitimate way to fill the deferred window in
   single-stream serving.
4. **dc/graph over PP-2** — both still "run unsplit, silently", same class of bug the batch
   path had. Neither is audited. `decode_step_dc` and the graph-capture path should get the
   same fail-closed treatment (cheap) before anyone measures them on a pair.
5. **`build.rs` nvcc resolution** — resolve from `PATH`/`CUDA_HOME` before the pinned
   `/usr/local/cuda-13.1`. Cost me the first build on this box; will cost the next lane too.

### Item 5: spec/MTP over PP-2 — the brief (not built, per "don't sink days")

Why it is the highest-value remaining perf item and not just another wiring job: the deferred
window's 1.905x needs token t+1 enqueued before token t's logits land, which plain
autoregressive decode structurally cannot do — but speculative decode **already has** the next
K tokens (the draft proposes them). So spec-over-PP2 is the one arm where the measured 1.9x is
reachable in real single-stream serving. The artifact already carries an MTP head.

Shape of the work, in the order it has to happen:

1. **Draft placement decision.** The MTP/NextN layers map to the LAST stage today
   (`pp::new_cache`'s trailing-layer rule). For spec that means the draft runs on stage N-1
   while the verify trunk starts on stage 0 — a full boundary hop per draft token, but also
   natural draft/verify overlap on two devices. The alternative (draft on stage 0) makes
   drafting local but serializes against the trunk. **Neither is measured; this is the
   experiment, not a foregone conclusion.**
2. **Verify is a batched forward of K+1 tokens** — so it lands on item 1 above. Spec-over-PP2
   cannot ship before the batched trunk is stage-split. That ordering is the point of this
   brief: item 1 is not optional infrastructure, it is spec's prerequisite.
3. **Rollback/accept state crosses stages.** Accept length is decided at the last stage (where
   the lm head is) but rollback touches every stage's KV. Today's spec loop assumes one engine
   owns all caches. Needs a stage-fan-out accept/rollback, event-ordered like `tx`/`rx`.
4. **Gate before perf, as always:** `run-spec` K=1..8 self-consistency over the door, plus a
   PP-2-vs-1-GPU accepted-token-stream identity check. A spec arm that changes acceptance is
   changing math, not scheduling.

Estimated shape: item 1 is the bulk; given a `decode_batch_layers` seam, spec's own PP wiring
is days, not weeks. Not started this lane.
