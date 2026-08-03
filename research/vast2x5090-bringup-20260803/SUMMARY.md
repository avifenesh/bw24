# vast.ai 2x RTX 5090 bring-up — 2026-08-03 (first desktop-GB202 session)

**Box:** vast.ai, 2x RTX 5090 32 GB (GB202 desktop, compute_cap 12.0, 575 W cap, PCIe
NODE topology, **no P2P**), 256-thread EPYC, 503 GB RAM, Ubuntu 24.04 + CUDA 13.0.1
devel. $0.748/hr. Engine tree: rsync of `restructure/public-split` worktree at
`24d19a04` (8bit-decision merge). Self-competition doctrine: **no llama arm anywhere
in this session** — laptop rows below are frozen cross-rig context with a stated
non-interleaved caveat.

**Model:** q27 = Qwen3.6-27B **Q8_0** (unsloth/Qwen3.6-27B-GGUF, 28.6 GB) — the 8-bit
prod serving arm per the 2026-08-03 decision. These are the **first q27-at-8-bit cells
in the repo**, measured on the deployment silicon. Drafter: bench-repo own-trim
(`drafts/qwen36-27b-nvfp4/draft-owntrim-nvfp4head-q4blk.gguf`).

## 1. Toolchain + build

- rustup stable 1.97.1, nvcc 13.0.88. `MEMRA_CUDA_ARCH` auto-detect landed **120a**
  from compute_cap 12.0 — no override needed.
- One friction: `build.rs` defaults nvcc to `/usr/local/cuda-13.1/bin/nvcc`; this
  image ships 13.0 at `/usr/local/cuda` → first build died `spawn nvcc: NotFound`.
  Fixed with `MEMRA_NVCC=/usr/local/cuda/bin/nvcc` (documented seam, no code change).
- `cargo build --release` wall: **5 m 35 s** (256 threads).

## 2. Gates before numbers — ALL GREEN (the headline)

| Gate | Verdict |
|---|---|
| kernel-check GPU 0 | **ALL GREEN** |
| kernel-check GPU 1 | **ALL GREEN** |
| run-gen argmax (pp512) | **MATCH** (prefill=decode=198, maxdiff 5.9e-1); batched-prime **MATCH** |
| run-spec K=1/2/3 | **PASS** token-identical, acceptance 77.8 / 75.0 / 71.4 % |
| ppn-gate PP-2 same-device serial | **PASS** 48 steps BIT-IDENTICAL |

Exactness transfers to desktop GB202 (170 SM vs laptop 82) with zero code change.

## 3. Single-card anchor cells (GPU 0, flock'd, N=5 process reps, medians)

| Cell | rig2x5090 Q8_0 | laptop board (NVFP4, frozen 2026-08-02) | note |
|---|---|---|---|
| plain tg128 @ d512 | **53.63** | 47.6 | 1.13x — cross-rig AND cross-quant (Q8_0 = 2x bytes/token of NVFP4) |
| plain tg128 @ d6257 | **52.75** | 46.2 | 1.14x; depth droop only −1.6% (laptop −2.9%) |
| prefill pp512 | **4151** | ~788 (0.6499 s/512, 2026-07-04 jsonl baseline, older code) | ~5x-class, stale-baseline caveat |
| prefill pp6257 | **4072** | — | flat vs pp512 (−1.9%) |
| spec best-K (pp512-cont.) | **K=4: 137.4** (acc 0.646) | board K=3 | best-K moved 3→4 on desktop; K=3 reads 132.0 |
| spec p1 K=3 | **125.9** (acc 0.75) | 116.4 | 1.08x |
| spec p2 K=3 | **109.1** (acc 0.567) | 101.2 | 1.08x |
| spec p3 K=3 | **94.4** (acc 0.52) | 86.0 | 1.10x |

Laptop spec rows are the board's NVFP4+trimmed-draft cells — cross-rig, cross-quant,
non-interleaved: **context, not verdicts.** Self-competition reading: desktop Q8_0
lands 1.08–1.14x above the laptop NVFP4 rows despite double the weight bytes —
Q8_0 serving on this silicon starts ABOVE the published laptop board.

**Spec-shape finding:** deeper K pays on desktop (K=4 beats K=3 by +4% on the
continuation class). Verify is relatively cheaper on 170 SMs. Per-class K re-sweep
due before publishing a rig2x5090 spec board.

**OOM finding (bench harness, not engine):** run-gen's batched-prime gate allocates a
second full cache; at d6257 on a 32 GB card with the 28.6 GB Q8_0 resident, the
timing pass OOM'd 5/5 (captured `Error: DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out
of memory")`; both argmax gates had already PASSED; VRAM 32096 MiB peak). Re-ran with
`MEMRA_PRIME_GATE=0` (documented diagnostics seam) → 5/5 clean at 52.73–52.76.

## 4. Two-GPU topologies

### (a) Two independent replicas (fleet doctrine) — WORKS, ~1.8x aggregate

1 replica/GPU (28.6 GB model ⇒ pairs-per-GPU off the table on 32 GB), serve-proxy cap
16/replica, load-serve c=8/16/32, N=3 passes, 1018 requests, **0 errors, 0 shed**:

| c | agg tok/s | p50 | p95 |
|---|---|---|---|
| 8 | 97.5 | 10.4 s | 10.5 s |
| 16 | 97.2 | 20.9 s | 21.1 s |
| 32 | 96.6 | 42.1 s | 42.4 s |

Aggregate ≈ 1.8x single-card plain decode, flat 8→32 (decode-bound saturation),
p95/p50 ≈ 1.008 everywhere — no tail inflation under the admission cap. (c8-pass1
117.1 was the cold-start outlier; steady band quoted.)

### (b) PP-2 (M2 path) — BLOCKED BY HOST: no PCIe P2P

`cuDeviceCanAccessPeer=0` both directions on this vast host (NODE topology,
consumer-board/ACS-class block). The M2 guard refused loudly, as designed:

```
device 0 cannot peer-access device 1 (cuDeviceCanAccessPeer=0); ppN cross-device
needs P2P — refusing a silently-staged path
```

Same-device serial PP-2 gate: **PASS, bit-identical** — the engine path is healthy;
it's the host, not the code. Consequence: **PP-2 q27 on rented 2x5090 requires a
P2P-enabled host — verify `nvidia-smi topo -m` before renting.** The deferred-readback
1.87x prize stays H100-box evidence for now. Fleet-of-replicas is the working 2-GPU
shape on commodity vast hosts and needs no P2P. No cross-device transport banner was
printable (init refuses before transport selection); topo matrix in
`pp2/topo-matrix.txt`.

## 5. M2 ride-along cell: desktop J/token decode A/B (FP8-ST program)

nvidia/Qwen3.6-27B-NVFP4 safetensors (MIXED: F8 attn + NVFP4 MLP), run-gen decode
tg128 @ pp512, N=3 interleaved A/B pairs:

| arm | tok/s (3 reps) | median |
|---|---|---|
| A: ST default (F8→Q8_0 re-encode) | 75.06 / 75.13 / 75.05 | 75.06 |
| B: `MEMRA_ST_E4M3=1` (e4m3-direct) | 76.19 / 76.16 / 76.18 | **76.18 (+1.5%)** |

argmax MATCH maxdiff 0.0 both arms, all reps. **The desktop 5090 reads box-side**:
e4m3-direct decode is +1.5% (laptop 171 W was −7%, G7e 600 W +7.1%) — the J/token
law holds direction; decode runs at ~66–94 W, far under the 575 W wall, so the
magnitude is small and the big FP8 win stays prefill-side. Program verdict: no
decode-side penalty on the deployment card — FP8-ST pursuit loses nothing at decode.

## 6. Thermals / clocks (unknown chassis — watched per rep)

No throttle at any point: anchor battery 31–68 °C, SM clocks pinned 2400–2925 MHz,
power peak 578 W (momentary, = cap, prefill only). Decode sits at ~80–94 W. This
host's cooling is fine; the vast-host thermal risk did not materialize.

## 7. Receipts map

- `rig2x5090.jsonl` — 6 rows (build, gates, anchors, fleet, pp2, e4m3-ab), rig5090 schema.
- `anchor/` — per-rep raw logs (plain/pp/spec), driver logs with per-rep
  temp/clock/power, 1 Hz GPU CSVs. `fleet2/` — replica + proxy logs, `points.jsonl`
  (9 load points), per-request JSONLs, 1 Hz 2-GPU CSV. `pp2/` — both gate arms' logs,
  `topo-matrix.txt`. `e4m3-ab/` — A/B logs + 1 Hz CSV.
- `gate1-kernel-check-gpu{0,1}.log`, `gate2-rungen-argmax.log`,
  `gate3-runspec-k123.log`, `build.log`, `dl-*.log` at top level.

## 8. What this founds

rig2x5090 board rows start from: plain 53.6/52.8, pp 4151/4072, spec 137.4 best-K,
p1/p2/p3 125.9/109.1/94.4 — all q27 Q8_0, all gates green. Open next: per-class spec
K re-sweep, a P2P-capable host for PP-2, and the FP8-ST tuning program with these
Q8_0 cells as the baseline it must beat (decode sign on product silicon: +1.5%, safe).
