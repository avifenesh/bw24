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
| cross-device pipelined | `dev01`, x40 | ~1 FAIL in ~190 (~0.5%), root cause open | **40/40 PASS, 0 FAIL** |
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
  cross-device at low probability; nothing here refutes it (40 runs cannot see a 0.5% event
  with confidence — P(0 failures | p=0.005, n=40) = 0.82, so this soak is simply not
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
