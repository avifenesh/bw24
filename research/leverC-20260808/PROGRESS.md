# Lever C - grouped Step35 prefill experts

Branch: `lane/cx-moe-grouped-prefill`
Train base: `13f5ddb8`
Pipeline dependency: local `lane/cx-pipeline-prime` tip `62fac3c0`

## Increment 0 - read-first mechanism contract

### The cost being removed

The Step35 prefill path reaches the sequential MoE fallback because its sigmoid router denies
the pairs and device-router arms. The anatomy receipt in
`research/pp-prefill-20260807/PROGRESS.md` measured that fallback class at 28% of prime time:
about 835K expert-kernel launches per pp4096 prime, with expert work dispatched one token at a
time even though routing is already known for the whole prefill chunk.

This lane will group routed rows by expert inside the existing Step35-legal host-routing family.
It will not relax the sigmoid-router predicates and will not route Step35 through pairs, device
routing, grouped-decode fusion, or any uniform-slab fallback that bypasses expert metadata.

### Chosen grouped mechanism

For each prefill MoE layer:

1. Compute router logits with the same `router_gemv` decision used by the sequential path.
   The cuBLASLt `m=t` router is forbidden because its logits and selected experts are
   chunk-size-dependent.
2. Apply the existing host sigmoid-router oracle, including correction bias, top-k ordering,
   scaling, and renormalization.
3. Bucket routed token rows by expert while retaining each route's original top-k slot.
4. Gather one `[m_e, n_embd]` activation matrix per expert and run the existing row-parallel q8
   expert kernels at `m=m_e`: `quantize_q8_1`, gate/up `qmatvec_expert_q8`,
   clamp-aware `ffn_act_lim`, a second `quantize_q8_1`, then down
   `qmatvec_expert_q8`.
5. Scatter each result back to its original token and top-k slot, then reduce slots in original
   top-k order with the same fused multiply-add chain as sequential `axpy_into`.

The local resident expert slab is legal input because it is only a storage source for the same
per-expert q8 kernels. Mixed-layout banks continue through metadata-aware cache/staging and never
enter uniform-only fused kernels.

### Why this targets exact arithmetic

- Router logits, route selection, weights, and route order come from the same m-invariant helper
  and host sigmoid code as the sequential oracle.
- `quantize_q8_1` and `qmatvec_expert_q8` index token rows independently. Raising `m` changes the
  launch grid, not any row's reduction program or quantized bytes.
- Step35's layer-specific SwiGLU clamp and macro scales stay on `ffn_act_lim`.
- Slot scatter preserves the router's original top-k position. Slot reduction therefore performs
  the same ordered `fmaf(weight, expert_output, accumulator)` sequence as the token loop.
- The shared-expert branch is unchanged.

The older `MEMRA_MOE_GROUPED` prototype is not the default implementation for this lane: its
expert arithmetic uses f32 dequantized `qmatvec_view`, while served Step35 uses q8 expert math.
That documented numeric-class difference is unnecessary here because the existing q8 expert
kernel already supports `m>1`.

### Dispatch seam and chunk uniformity

The default promotion is restricted to the served prime walker, not decode/spec:

- every Step35 prefill chunk with more than one row uses the grouped q8 arm;
- `MEMRA_MOE_GROUPED=0` restores the sequential prefill oracle;
- explicit `MEMRA_MOE_GROUPED=1` retains the existing opt-in meaning outside prefill;
- no chunk-local size threshold chooses between arithmetic classes.

Thus all normal chunks of one request use the same dispatch class. The final single-row tail, if
present, may use the sequential path because grouping cannot change a one-row launch and both
paths are required to be bit-identical.

### Planned gates

| Gate | Required verdict |
|---|---|
| grouped-vs-sequential MoE oracle | bit-identical model-backed output |
| `kernel-check` | ALL GREEN |
| `chunkinv35` / `tickinv35` | PASS, with canaries retaining teeth |
| `ppsplit` | unsplit, serial split, and pipelined split bit-identical and live |
| `run-gen` PP-2 | argmax MATCH |
| `run-spec` K=1..8 | 8/8 PASS at pinned acceptance |
| performance | pp512/2048/4096, N=5 interleaved against the 417.6 tok/s pipeline baseline |

Raw build, gate, and performance logs will be retained under `research/leverC-20260808/raw/`.
Every reported median will state N and thermal regime. The supplied steering path
`~/.lanectl/inbox/cx-leverC.md` was absent at lane start; no alternate Lever C steering file was
present under `~/.lanectl`.

## Increment 1 - grouped q8 prefill implementation

The local pipeline-prime tip `62fac3c0` was merged first, preserving its measured 417.6 tok/s
pp4096 baseline and current `ppsplit`/chunk/tick gate surfaces.

The grouped implementation now has two exact storage cases behind one host-routed dispatch:

- **Resident uniform Step35 bank:** the host sigmoid oracle produces the original pair ids,
  expert ids, and weights. An expert CSR feeds the existing `moe_pairs_matvec_q8_dec` kernel,
  which decodes one expert weight group and applies it to all routed rows in that expert segment.
  Gate/up outputs remain indexed by original pair id; `ffn_act_lim` applies the layer's Step35
  clamp; down uses the same CSR; `moe_pairs_scatter` performs the original slot-ordered FMA chain.
  This reuses only the pairs arm's arithmetic kernel, not its softmax router or plain-SiLU
  dispatch policy.
- **Spill, remote-slab, or mixed-layout bank:** the existing A2 expert groups remain
  metadata-aware. Uniform q8-supported experts run `quantize_q8_1` plus
  `qmatvec_expert_q8` at `m=m_e` from a local slab or live SLRU slot. Direct no-cache/frozen
  staging and mixed layouts retain the sequential oracle's f32 class, with each expert's
  authoritative `qtype`, `row_bytes`, `len`, and source.

`moe_router_logits` is now shared by sequential and grouped dispatch, so both use
`router_gemv` under the exact-prefill policy. All prefill entry points call the prefill wrapper;
decode and speculative verification keep the original wrapper and dispatch class.
`MEMRA_MOE_GROUPED=0` is the live rollback seam, while an explicit nonzero value preserves the
old opt-in behavior for other callers.

Local proof:

| Check | Verdict |
|---|---|
| `cargo check -p memra-engine` | PASS, CUDA 13.1, auto-detected sm_120a |
| worktree scope | source + flag catalog + this increment only |

Target-rig exactness and performance remain pending; no winner claim is made from the local
compile.

## Increment 2 - first Box2 oracle is red

Box2 ran commit `a1e04b43` on the model-backed Step35 artifact. The first grouped layer took the
intended resident arm, but `MEMRA_MOE_GATE=1` rejected its output:

| Check | Verdict |
|---|---|
| grouped arm engagement | `il=3 t=19 dispatch=resident-q8` |
| grouped vs sequential bytes | **FAIL**, 55,032 / 77,824 f32 elements differ, max diff 1.358427e-5 |
| model-backed `kernel-check` | `ALL GREEN` |

The remaining acceptance battery was stopped after preserving the failure and kernel receipts.
No invariance or performance claim is made from this build.

The failed exactness assumption was the whole-layer arithmetic class. The sequential resident
Step35 path at an unclamped layer runs the fused per-token q8 pair
`moe_gate_up_silu8_q8` plus `moe_down8_fma_q8`. The first Lever C arm instead ran
`moe_pairs_matvec_q8_dec` for each projection, a separate activation kernel, a separate down
matvec, and `moe_pairs_scatter`. Matching row-dot and slot-reduction descriptions were not enough
to make those two complete kernel chains byte-identical.

The correction is to batch the existing fused per-token program over the prefill token axis:
host sigmoid routing still supplies the exact `sel` and `w`, while the established
`moe_gate_up_silu8_dev_q8_rows` / `moe_down8_fma_dev_q8_rows_g` family executes all tokens in one
launch pair. Those kernels are rows twins of the sequential fused program, not the denied
softmax router. Step35's clamped layers cannot enter the plain-SiLU rows arm and retain the
clamp-aware grouped q8 fallback.

## Increment 3 - rows path exact, clamped fallback isolated

A dedicated Box2 worktree at `/home/ubuntu/memra-cx-leverC` was pinned to `72a929ec` after the
shared `~/memra` checkout changed commits while an earlier command waited on the GPU lock. Only
results from the pinned worktree are accepted below.

The rebuilt model-backed oracle proved the batched rows correction byte-identical for every
unclamped MoE layer from `il=3` through `il=42`. The first clamped layer isolated the remaining
red:

| Layer class | Dispatch | Verdict |
|---|---|---|
| `il=3..42`, unclamped | `resident-q8-rows` | `BYTE-IDENTICAL` at every layer |
| `il=43`, clamped | first expert-major fallback | **FAIL**, 52,168 / 77,824 differ, max diff 6.203651e-4 |

This disproves the remaining decode-once assumption for the clamped path. Its correction keeps
the same host sigmoid routes, clamp kernel, batched row quantization, and slot-ordered scatter,
but uses the pair-major `moe_pairs_matvec_q8` body for gate, up, and down. That kernel is the
literal `qmatvec_expert_q8` row program with a pair-indexed weight pointer, matching the
sequential clamped oracle without relying on the decode-once extractor's claimed equivalence.

## Increment 4 - model-backed byte oracle green

Commit `adfa5a5e` was rebuilt from the pinned Box2 worktree and rerun with
`MEMRA_MOE_GATE=1 MEMRA_MOE_STATS=1`. The short `run-gen` protocol performs several prefill
comparisons internally; all 210 grouped layer comparisons were byte-identical and the log
contains zero `MISMATCH` rows.

| Layer class | Live dispatch | Oracle verdict |
|---|---|---|
| unclamped Step35 MoE | `resident-q8-rows` | `BYTE-IDENTICAL` |
| clamped Step35 MoE | `resident-q8-clamped-pairs` | `BYTE-IDENTICAL` |
| process result | model-backed `run-gen`, PP-2 | exit 0 |

The grouped dispatch now stays entirely inside the Step35-legal family: routing and weights come
from the m-invariant host sigmoid oracle; the rows and pair-batched kernels consume those fixed
routes and never invoke the softmax device router.

## Increment 5 - full Box2 correctness battery green

The committed gate driver ran from the dedicated Box2 worktree at `b341c109` under one exclusive
GPU lock. That commit adds evidence only; the release binaries are source-identical to the
`adfa5a5e` implementation. The target was two NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs
with PP stages pinned to devices `0,1`.

| Gate | Verdict |
|---|---|
| grouped-vs-sequential model oracle | PASS: 210 `BYTE-IDENTICAL` layer comparisons, zero `MISMATCH`; live `resident-q8-rows` and `resident-q8-clamped-pairs` dispatch |
| model-backed `kernel-check` | `ALL GREEN` |
| `ppsplit` | PASS: unsplit, serial split, and pipelined split bit-identical with split/overlap liveness |
| `ppsplit` canary | PASS: serial-pipeline seam broke overlap liveness as required |
| `chunkinv35` | PASS: invariant for chunks `4096,513,512,256,64` |
| `chunkinv35` canary | PASS: legacy seam produced a variant result as required |
| `tickinv35` | PASS: invariant for budgets `0,1024,513,512,256,64` and splits `64,256,512` |
| `tickinv35` canary | PASS: legacy seam produced a variant result as required |
| `run-gen`, PP-2, 64 generated tokens | PASS: prefill/decode argmax `6776` MATCH; batched-prime/tokenwise argmax `6776` MATCH |
| `run-spec`, PP-2, K=1..8 | 8/8 self-consistency PASS |
| complete battery | exit 0 |

Pinned speculative acceptance was K1 `14/17` (82.4%), K2 `15/32` (46.9%), K3 `15/48`
(31.2%), K4 `15/64` (23.4%), K5 `15/80` (18.8%), K6 `15/96` (15.6%), K7 `15/112`
(13.4%), and K8 `15/128` (11.7%).

The battery ran serially under `/tmp/memra-gpu.lock`, with no competing compute process present at
the recorded boundaries. Across 44 before/after snapshots, GPU temperature ranged from 33 to
44 C and observed SM clocks ranged from 180 to 2422 MHz, including idle boundary samples.

Raw logs and their checksum manifest are retained in
`research/leverC-20260808/raw/gates-b341c109/`.

## Increment 6 - grouped prefill is the Box2 winner

The final mechanism keeps Step35 inside its legal host-sigmoid routing family:

- `router_gemv` computes m-invariant logits and the existing host oracle fixes expert ids,
  weights, and top-k order for the whole prefill chunk;
- unclamped layers batch the sequential fused q8 program over token rows with
  `moe_gate_up_silu8_dev_q8_rows` and `moe_down8_fma_dev_q8_rows_g`;
- clamped layers batch the literal `qmatvec_expert_q8` row program by routed pair, retain
  `ffn_act_lim`, requantize each row, and scatter in original slot order;
- decode and speculative verification remain on their prior dispatch, and
  `MEMRA_MOE_GROUPED=0` is the live prefill rollback seam.

Box2 measured commit `41f0af6f` on two RTX PRO 6000 Blackwell Server Edition GPUs. Each cell
used five independent timed processes with one warmup per process; grouped-default and rollback
order alternated by repetition under one GPU-lock hold. The same prompt bytes and pipeline source
as `research/pipeprime-20260808` were used.

| shape | realized T / effective auto geometry | rollback median, N=5 | grouped median, N=5 | grouped vs rollback | historical pipeline | grouped vs historical |
|---|---|---:|---:|---:|---:|---:|
| pp512 class | 461 / chunk 128 x 4 | 324.5 tok/s | **497.5 tok/s** | **+53.3%** | 330.0 tok/s | +50.8% |
| pp2048 class | 1833 / chunk 230 x 8 | 402.3 tok/s | **639.2 tok/s** | **+58.9%** | 401.8 tok/s | +59.1% |
| pp4096 | 4096 / chunk 512 x 8 | 426.9 tok/s | **697.6 tok/s** | **+63.4%** | 417.6 tok/s | **+67.0%** |

The same-commit rollback is the controlled A/B result; the historical pipeline column is context.
The raw `ppprime` footer prints `chunk=4096(default)` when `MEMRA_PRIME_CHUNK` is unset, but that
is the raw environment label, not the effective chunk. The naked PP-2 policy in
`prime_chunk_tokens()` resolves the three prompts to the geometry shown above, and the preceding
`ppsplit` gate proves the default pipeline and overlap counters are live.

All 30 timed arms exited zero. Independent parsing of the individual logs reproduced every TSV
median. Across 62 before/after snapshots, temperatures were 36 to 43 C and SM clocks were 2272 to
2370 MHz; all boundaries showed 0 MiB used and no competing compute process.

The grouped arm therefore remains the Step35 prefill default in this lane. No merge, tag, release,
or origin push is made here: the local RTX 5090 proof-rig correctness, memory, and throughput
battery remains required before this default can ship.

The lane steering file was first observed during the final receipt check, after the Box2 campaign
had completed. Its `2026-08-08T11:15:33Z` coordination note reserves Box2 for serving, forbids new
long runs there, and explicitly exempts lanes already in final receipts. No further Box2 GPU work
was started after reading it.

Raw logs, prompts, results, transfer provenance, and checksums are retained in
`research/leverC-20260808/raw/perf-41f0af6f/`.

## Increment 7 - local RTX 5090 proof gate rejects the default flip

The final proof-rig battery ran commit `3a491bcf` on the local RTX 5090 Laptop under
`/tmp/gpu5090.lock`. The entry snapshot showed the expected resident Hermes embedder only:
394 MiB of compute memory, 0% GPU utilization. The desktop stayed on the `performance` profile,
and every throughput process ran sequentially at `nice -n 10`.

Step35 is box-only, so its Box2 byte-oracle and model receipts remain authoritative for that
artifact. The local dispatch/correctness set was the available IQ/MoE set from the k32 receipts:
KAT-Coder, Gemma 4 26B-A4B, and Qwen 3.6 35B-A3B.

| Gate | Local 5090 verdict |
|---|---|
| release build | PASS: CUDA 13.1, auto-detected sm_120a |
| model-backed `kernel-check` on KAT | PASS: `ALL GREEN` |
| KAT `run-gen`, explicit grouped path | PASS: prefill/decode argmax `271` MATCH; batched-prime/tokenwise `271` MATCH; resident-q8-rows dispatch observed |
| Gemma 26B `run-gen` | PASS: prefill/decode argmax `236786` MATCH; batched-prime/tokenwise `236786` MATCH |
| Qwen 35B `run-gen`, explicit grouped path | PASS: prefill/decode argmax `8160` MATCH; batched-prime/tokenwise `8160` MATCH; resident-q8-rows dispatch observed |
| qwen `chunkinv` | PASS: both pinned prompts bit-identical at chunks `2048,64,32` |
| qwen `chunkinv` canary | PASS: legacy seam restored chunk-dependent output on both prompts |
| post-rejection rollback rebuild and naked-default probe | PASS: release rebuild green; unset `MEMRA_MOE_GROUPED` produced both MATCH lines, zero `moe-grouped` records, and zero OOMs |

KAT was the largest requested local model that both fit resident and entered the explicit grouped
path. Peak VRAM was sampled every 100 ms across model load, one warmup, and one timed pp2048
prime:

| arm | peak VRAM | headroom on 24,463 MiB | OOM |
|---|---:|---:|---|
| rollback (`MEMRA_MOE_GROUPED=0`) | 19,606 MiB | 4,857 MiB | no |
| grouped (`MEMRA_MOE_GROUPED=1`) | 19,542 MiB | 4,921 MiB | no |

The required throughput transfer gate was red. Five independent processes per arm ran as adjacent
interleaved pairs in one lock hold, with alternating order and one warmup per process:

| local model / shape | rollback median, N=5 | grouped median, N=5 | grouped vs rollback | pairwise |
|---|---:|---:|---:|---:|
| KAT-Coder pp2048 | **4027.1 tok/s** | 992.7 tok/s | **-75.3%** | 0/5 wins |

All ten processes exited zero. The measured thermal window was 57 to 63 C and observed SM clocks
were 1590 to 1597 MHz. This is a performance rejection, not an OOM or correctness failure.

By the proof-rig rule, the Box2 win remains research evidence but cannot flip the runtime default.
`MEMRA_MOE_GROUPED` therefore returns to default-off; explicit `=1` preserves the grouped research
arm and explicit `=0` selects the established path. The rollback-default edit was rebuilt in
release mode, then a short locked KAT probe with `MEMRA_MOE_GROUPED` removed from the environment
confirmed that the naked command takes the established path. No merge, tag, release, or origin
push is made.

Raw logs, 100 ms memory samples, the N=5 TSV, and checksums are retained in
`research/leverC-20260808/raw/5090/`.
