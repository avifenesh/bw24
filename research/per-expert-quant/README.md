# Hy3 spilling and quantization research pack

This lane owns two deliverables: spill-path improvements for large expert banks and a controlled
five-arm quantization study. Every retained routed expert is quantized; there is no BF16 expert
evaluation arm or BF16 expert fallback. Model loading, CUDA correctness, artifact generation,
research measurement, calibration, and public evaluation happen on the provisioned G7e machine.
The local RTX 5090 rig remains bw24's deployment and final performance target; runtime defaults are
not flipped until the completed code and artifacts pass the same correctness, memory, and throughput
gates there.

Mmap remains the default and correctness fallback, not the throughput endpoint. The cold G7e result
and bounded explicit-read promotion probe are recorded in
[`evidence/spill-prefetch-g7e-20260710.md`](evidence/spill-prefetch-g7e-20260710.md). The runtime now
also provides blocking `pread` as the explicit-I/O oracle and a bounded worker backend that reads
ahead into CUDA-pinned buffers while the caller thread retains all H2D and cache-publication work.
On the first five frozen calibration requests, the depth-8 worker run was 2.866x faster than mmap
with identical responses and 395 byte-identical routing rows; see
[`evidence/spill-worker-ab-g7e-20260710.md`](evidence/spill-worker-ab-g7e-20260710.md). Test buffered
and `O_DIRECT` reads through the same state machine; add io_uring only if it beats this worker
baseline. The deferred ring design is specified in
[`io-uring-spill-design.md`](io-uring-spill-design.md).

The first Hy3 quality attempt exposed and fixed an architecture-scoping bug in RMSNorm loading.
The official-reference layer gate, coherent generation smoke, exact mmap/pread parity result, and
the hashes of traces invalidated by that fix are recorded in
[`evidence/hy3-runtime-correctness-g7e-20260710.md`](evidence/hy3-runtime-correctness-g7e-20260710.md).
Do not use a routing trace captured before commit `38a5b08`.

GGUF remains bw24's general runtime and delivery focus. This study reads the pinned Hy3
safetensors checkpoint as common quantization source material and uses repack overlays to represent
per-expert precision experimentally. Spill, cache, prefetch, and dispatch changes must stay in the
shared expert-serving path, preserve GGUF behavior, and pass the existing GGUF gates before release.

## Target model and frozen recipes

The source model is `tencent/Hy3` with 192 routed experts per MoE layer, top-8 sigmoid routing, and
MoE layers 1 through 79. The frozen REAP50 mask retains 96 experts per layer. The public checkpoint
renumbered them to 0..95 without publishing the original ids, so `recover_hy3_reap_mask.py` matches
its 8-bit router rows back to the pinned BF16 router and confirms every match with the untouched
correction bias. Exact model, REAP-method provenance, and reference revisions live in
`arms.lock.json`.

The scored arms are fixed:

1. `plain_quant`: full 192-expert bank, uniform NVFP4, no pruning.
2. `plain_reap_quant`: frozen REAP50 96-expert bank, uniform NVFP4.
3. `plain_reap_mix_quant`: the same REAP50 mask, with the least-used 48 experts Q2_K and the
   remaining 48 NVFP4.
4. `mix_quant`: full bank ranked separately per layer; hottest 25% NVFP4, middle 50% Q3_K,
   coldest 25% Q2_K, and zero-count experts pruned.
5. `mix_quant_prune25`: full bank ranked separately per layer; hottest 25% NVFP4, next 25% Q3_K,
   next 25% Q2_K, and coldest 25% pruned.

The promoted follow-up `traffic_mix_quant` replaces arbitrary quartiles with boundaries measured
from the corrected full-bank trace: hottest 16 experts Q8_0, next 37 NVFP4, next 73 Q2_K, and
coldest 66 pruned per layer. Those boundaries capture 22.32%, 49.85%, and 84.98% cumulative routing
traffic respectively.

BF16 Hy3 is common source material only. It is never scored. The public MLX REAP50 checkpoint is a
mask donor only; none of its already-quantized expert weights enter a scored artifact.

Exact validated overlay, staged-directory, logical payload, tier, and prune counts are frozen in
[`evidence/five-arm-artifact-sizes-g7e-20260710.md`](evidence/five-arm-artifact-sizes-g7e-20260710.md).

Q8 means GGUF Q8_0 (8.5 effective bits/weight), Q2 means GGUF Q2_K (2.625 effective bits/weight),
Q3 means Q3_K (3.4375 bits/weight), and
NVFP4 is bw24's 64-value/36-byte block format (4.5 bits/weight). The mixed path is correctness
first: Q2_K uses the generic staged f32-dequant kernel until a dedicated target-rig-gated fast
kernel exists.

## What is implemented

- BW24_MOE_TRACE=/path records routed expert ids without changing normal runs.
- BW24_MOE_WEIGHT_TRACE=/path records the same selections with normalized router weights.
- BW24_CONFIDENCE_TRACE=/path switches calibration requests to tokenwise teacher forcing and
  records reference-token log probability, correctness, margin, and entropy. It is an opt-in
  research path; normal generation and batched prefill are unchanged.
- tools/build_expert_tier_plan.py emits calibration-independent uniform plans or aggregates frozen
  calibration traces for usage-ranked plans, accepts original-id expert masks, and records all
  trace/mask hashes.
- tools/recover_hy3_reap_mask.py reconstructs the public REAP50 original-id mask from router rows,
  requires one-to-one high-margin matches, and independently checks correction biases.
- tools/prepare_mixed_expert_repack.py streams BF16/F16/F32 or stacked MLX-affine experts on CPU
  and writes Q8_0, Q2_K, Q3_K, and NVFP4 byte ranges. Bounded `--workers` parallelism preserves exact
  expert order and is byte-compared against the single-worker path. `--resume` only reuses files
  whose atomic completion receipt matches the exact plan, layout, source identity, shape, and byte
  count. Every active expert projection must be assigned.
- A v2 overlay can reuse a complete manifest repack for dense, attention, router, tokenizer, and
  shared-expert tensors. Expert data is stored in one mixed file per layer/projection.
- Per-expert overlay entries remain zero-copy mmap windows in `HostExps`; the 161 GB full-bank
  control does not materialize an impossible second copy in 124 GB host RAM.
- Contiguous all-active/all-NVFP4 entries are coalesced back into a uniform mmap slab, preserving
  the uniform fused dispatch path for `plain_quant`; pruned or mixed arms retain per-expert layouts.
- `BW24_MOE_PAGE_PREFETCH=1` issues best-effort `MADV_WILLNEED` one expert ahead for mmap-backed
  GGUF and repack ranges in both sequential and grouped dispatch. It is off by default until matched
  cold-cache G7e measurements and final RTX 5090 gates prove a win.
- `BW24_SPILL_IO=mmap|pread|worker` selects the expert storage path. Mmap remains the default;
  `pread` is the blocking byte/H2D oracle, while `worker` moves exact positioned reads through a
  bounded CUDA-pinned pool and CPU read workers. Grouped prefill queues known-next expert reads, but
  only the caller thread submits H2D and publishes cache residency. Any initialization/read/ring
  degradation retains the validated mmap extent as fallback.
- `BW24_SPILL_PREAD_DEPTH` bounds worker threads and pinned buffers (`2` by default; the G7e A/B used
  `8`). `BW24_SPILL_STATS=1` logs cumulative reads, bytes, errors, short reads, mmap fallbacks,
  buffer waits, ring-full events, expert-cache hits/misses, H2D staged bytes, and slot count when
  server requests finish. Public-eval metadata records start/end snapshots and per-run deltas.
- Optional pruned_experts masks preserve original router width and expert ids. Masked experts are
  excluded before top-k and have no weight bytes in the artifact.
- HostExps carries qtype, row bytes, byte extent, and offset per expert. Mixed/pruned layers stay
  on metadata-aware staged, SLRU-cache, or grouped paths; uniform fused kernels remain
  uniform-only.
- validate_artifact.py checks the embedded-plan hash, expert coverage, allowed qtypes,
  non-overlapping byte ranges, total bytes, contamination metadata, and optional source
  fingerprints.
- The public eval suite is pinned and stores the served artifact manifest/hash with each run.

## Owned spill track

The full-bank arms are also the spill stress cases. Treat them as an end-to-end data-movement and
GPU-compute problem: combine mmap/zero-copy views, local-NVMe locality, pinned host buffers, SLRU
residency, asynchronous prefetch/overlap, PCIe transfer, and mixed-layout GPU dispatch without
changing the frozen precision plans. Record spill hit/miss counts, fault/read bytes, H2D bytes,
stage timing, peak host/VRAM, and throughput separately from quality.
Public eval examples must never tune cache size, prefetch policy, REAP masks, or precision tiers.

`/data` is the durable artifact store on the target host. Before calibration, public evaluation, or
spill measurement, copy the selected artifact to `/scratch/artifacts/<arm>` on the G7e local NVMe
and confirm its `manifest.json` hash matches the durable copy. The persistent EBS volume is suitable
for sequential artifact construction but its 4 KiB mmap-fault throughput is not a valid bw24 spill
benchmark.

### Backend ladder beyond mmap

1. Done: `pread` into a bounded CUDA-pinned pool is the blocking correctness and storage-ceiling
   proof, not the throughput endpoint.
2. Implemented and G7e-gated: a small disk worker pool fills the same pinned buffers while the CUDA
   owner continues compute; only the CUDA owner submits H2D and publishes GPU-cache residency. Use
   depth 2 as the current local-safe default and depth 8 for the measured G7e configuration. The
   first-five A/B cut wall time by 65.1%; the local 5090 gate is still required before any default
   change.
3. A/B buffered reads against `O_DIRECT`. Every offset and length in all five current Hy3 staged
   artifacts is 4 KiB aligned, while `/scratch` reports 512-byte direct-I/O alignment, so this study
   needs no artifact rewrite. General GGUF must query `STATX_DIOALIGN` and use aligned over-read or
   an aligned sidecar when a tensor extent is not directly readable.
4. Compare io_uring registered files/buffers against the winning worker implementation. The local
   8 MiB memlock limit bounds the current 3,538,944-byte expert ring to depth 2; G7e is unlimited.
5. Test mapped pinned-host reads only as a cold, read-once bypass. Recurrent experts belong in HBM;
   direct kernel reads from host memory repeatedly cross PCIe.

Do not plan around cuFile/GDS. Both current hosts report compatibility rather than direct mode, the
local target is GeForce, and the currently qualified NVMe P2PDMA list does not cover either target
GPU. Reconsider only if `gdscheck` proves an actual direct path on the deployment machine.

## Calibration and plan generation

Calibration data and public evaluation data must be disjoint. Use a representative private or
training-side corpus for routing counts; never use IFEval, GSM8K, BBH, DROP, HumanEval, or MBPP
examples to select tiers.

`calibration.lock.json` freezes 32 examples from each of the six training-side strata recommended
by REAP (192 requests total), the exact Hub revisions, seed, shuffle buffer, and 1,024-token cap.
The token cap keeps the routing run practical while still yielding tens of millions of layer/expert
assignments across Hy3. Freeze prompt ids once so the full-bank and REAP50 controls see identical
tokens:

    /data/src/reap/.venv/bin/python research/per-expert-quant/prepare_calibration.py \
      --tokenizer /data/models/hy3-source \
      --cache-dir /data/cache/huggingface/datasets \
      --out-dir /data/calibration/hy3-routing-v1

The locked result is 192 requests / 163,409 prompt tokens (103,274,488 routed-expert assignments
per control) with `requests.jsonl` SHA-256
`b23225e14d70947bc39d1ed92795d66deb365a69538cdb124b5c85e2b7daee04`. The builder fails on any
drift. Its generated manifest also records every source id/content hash and prompt length; keep it
with the traces and final report.

Capture enough requests to cover the intended deployment distribution:

    BW24_SERVE_SPEC=0 \
    BW24_KV_REUSE=0 \
    BW24_CTX=1032 \
    BW24_MOE_GROUPED=1 \
    BW24_MOE_TRACE=/data/runs/hy3-calibration.trace \
    BW24_MODELS=plain_quant=/data/artifacts/plain-quant \
    ./target/release/bw24-server

Spec decode and KV reuse are disabled so a zero-generation request still primes and traces every
frozen prompt token. With the server ready, submit the prompt ids. Use a fresh trace/output pair for
each uniform control; do not append one arm to the other:

    /data/src/reap/.venv/bin/python research/per-expert-quant/capture_calibration.py \
      --requests /data/calibration/hy3-routing-v1/requests.jsonl \
      --model plain_quant \
      --out /data/calibration/hy3-routing-v1/plain-quant-requests.jsonl

Repeat with `BW24_MODELS=plain_reap_quant=/data/artifacts/plain-reap-quant`,
`BW24_MOE_TRACE=/data/runs/hy3-reap50-calibration.trace`, and `--model plain_reap_quant`.

The trace format is one line per layer/forward: layer, token count, then comma-separated expert
ids. Multiple trace files may be passed and their SHA-256 hashes are frozen into the plan.

Recover the frozen REAP mask after both pinned downloads complete:

    python3 tools/recover_hy3_reap_mask.py \
      --base /data/models/hy3-source \
      --reference /data/models/hy3-reap50-mlx-reference \
      --base-revision 716aa7241bd6d95896be4ebfc761162a9c4d49ef \
      --reference-revision e054317b43aa601484a219a53e33e02e46caa970 \
      --out /data/plans/hy3-reap50-mask.json

Generate the two uniform controls without reading a calibration trace. Both consume the same BF16
source and preserve original expert ids:

    python3 tools/build_expert_tier_plan.py \
      --recipe uniform-nvfp4 --expert-count 192 --original-expert-count 192 \
      --top-k 8 --layers 1-79 --out /data/plans/plain-quant.json

    python3 tools/build_expert_tier_plan.py \
      --recipe uniform-nvfp4 --expert-count 192 --original-expert-count 192 \
      --mask /data/plans/hy3-reap50-mask.json \
      --top-k 8 --layers 1-79 --out /data/plans/plain-reap-quant.json

For a full 192-expert Hy3 source, build the usage pyramid and prune zero-count experts:

    python3 tools/build_expert_tier_plan.py \
      --trace /data/runs/hy3-calibration.trace \
      --recipe usage-pyramid \
      --expert-count 192 \
      --original-expert-count 192 \
      --top-k 8 \
      --expected-tokens 163409 \
      --layers 1-79 \
      --hot-fraction 0.25 \
      --low-fraction 0.25 \
      --prune-unused \
      --out /data/plans/mix-quant.json

Build the fixed usage-quartile variant from the same full-bank trace. Each layer has exactly 48
NVFP4, 48 Q3_K, 48 Q2_K, and 48 pruned experts:

    python3 tools/build_expert_tier_plan.py \
      --trace /data/runs/hy3-calibration.trace \
      --recipe quartile-prune \
      --expert-count 192 \
      --original-expert-count 192 \
      --top-k 8 \
      --expected-tokens 163409 \
      --layers 1-79 \
      --out /data/plans/mix-quant-prune25.json

Build the promoted traffic ladder from the corrected full-bank trace. Each layer has exactly 16
Q8_0, 37 NVFP4, 73 Q2_K, and 66 pruned experts:

    python3 tools/build_expert_tier_plan.py \
      --trace /data/runs/hy3-calibration-normfix-66394bf-worker-d8-full.trace \
      --recipe traffic-ladder \
      --expert-count 192 \
      --original-expert-count 192 \
      --top-k 8 \
      --expected-tokens 163409 \
      --layers 1-79 \
      --q8-count 16 \
      --nvfp4-count 37 \
      --q2-count 73 \
      --out /data/plans/traffic-mix-quant.json

The same recipe can cover the full bank when pruning is not desired. The evidence-driven
no-prune follow-up keeps the 16 Q8_0 and 37 NVFP4 traffic boundaries, then assigns all remaining
139 experts to Q2_K:

    python3 tools/build_expert_tier_plan.py \
      --trace /data/runs/hy3-calibration-normfix-66394bf-worker-d8-full.trace \
      --recipe traffic-ladder \
      --expert-count 192 \
      --original-expert-count 192 \
      --top-k 8 \
      --expected-tokens 163409 \
      --layers 1-79 \
      --q8-count 16 \
      --nvfp4-count 37 \
      --q2-count 139 \
      --out /data/plans/traffic-mix-quant-no-prune.json

The leaner ablation keeps only the hottest 16 experts at Q8_0 and assigns the other 176 to Q2_K.
Set the middle-tier count to zero explicitly:

    python3 tools/build_expert_tier_plan.py \
      --trace /data/runs/hy3-calibration-normfix-66394bf-worker-d8-full.trace \
      --recipe traffic-ladder \
      --expert-count 192 \
      --original-expert-count 192 \
      --top-k 8 \
      --expected-tokens 163409 \
      --layers 1-79 \
      --q8-count 16 \
      --nvfp4-count 0 \
      --q2-count 176 \
      --out /data/plans/traffic-q8-q2-no-prune.json

### Confidence-conditioned no-prune follow-up

Freeze a smaller domain-balanced subset from the already seeded private calibration corpus. Four
requests from each of its six strata keeps this teacher-forced pass practical while retaining the
same public-eval exclusion contract:

    python3 research/per-expert-quant/select_confidence_calibration.py \
      --source /data/calibration/hy3-routing-v1/requests.jsonl \
      --per-stratum 4 \
      --out /data/calibration/hy3-confidence-v1/requests.jsonl \
      --manifest /data/calibration/hy3-confidence-v1/calibration.lock.json

Launch the full-bank control with speculative decode and KV reuse disabled. Confidence tracing
also enforces one active request and tokenwise prefill so each routed-expert set aligns with the
reference token predicted by that forward:

    BW24_SERVE_SPEC=0 \
    BW24_KV_REUSE=0 \
    BW24_CTX=1032 \
    BW24_CONFIDENCE_TRACE=/data/calibration/hy3-confidence-v1/confidence.jsonl \
    BW24_MOE_WEIGHT_TRACE=/data/calibration/hy3-confidence-v1/routes-weighted.trace \
    BW24_MODELS=plain_quant=/scratch/artifacts/plain-quant \
    ./target/release/bw24-server

Submit the frozen prompt ids with the existing capture script. Its request ordinal becomes the
stable trace id used to join confidence and routing rows:

    python3 research/per-expert-quant/capture_calibration.py \
      --requests /data/calibration/hy3-confidence-v1/requests.jsonl \
      --model plain_quant \
      --out /data/calibration/hy3-confidence-v1/request-results.jsonl

Build the first candidate only after the two traffic no-prune screens choose their byte budget.
The counts below reproduce the 16 Q8_0 + 37 NVFP4 + 139 Q2_K budget. Allocation is ranked globally,
so individual layers may receive different counts while the artifact-wide totals remain exact:

    python3 tools/build_confidence_tier_plan.py \
      --requests /data/calibration/hy3-confidence-v1/requests.jsonl \
      --confidence-trace /data/calibration/hy3-confidence-v1/confidence.jsonl \
      --weight-trace /data/calibration/hy3-confidence-v1/routes-weighted.trace \
      --layers 1-79 --expert-count 192 --top-k 8 \
      --q8-count 16 --nvfp4-count 37 --q2-count 139 \
      --out /data/plans/confidence-mix-quant-no-prune.json

If the 16 Q8_0 + 176 Q2_K arm wins instead, set `--nvfp4-count 0 --q2-count 176`. The plan records
all three input hashes, domain-local confidence bands, frozen scoring constants, exact global tier
totals, and the top score diagnostics. Public eval results never enter allocation.

Run both new self-tests before capture or plan generation:

    python3 research/per-expert-quant/select_confidence_calibration.py --self-test
    python3 tools/build_confidence_tier_plan.py --self-test

For the masked REAP50 bank, build the exact 48 Q2_K / 48 NVFP4 split. The trace retains original
expert ids because bw24 masks the full-width router instead of renumbering it:

    python3 tools/build_expert_tier_plan.py \
      --trace /data/runs/hy3-reap50-calibration.trace \
      --mask /data/plans/hy3-reap50-mask.json \
      --recipe reap50-plus25 \
      --expert-count 192 \
      --original-expert-count 192 \
      --top-k 8 \
      --expected-tokens 163409 \
      --layers 1-79 \
      --out /data/plans/plain-reap-mix-quant.json

Run the builder self-test before producing plans:

    python3 tools/build_expert_tier_plan.py --self-test

Create at least three matched random controls without changing tier counts or prune masks:

    for seed in 11 29 47; do
      python3 tools/make_random_tier_control.py /data/plans/mix-quant.json \
        --seed $seed --out /data/plans/mix-quant-random-$seed.json
    done

## Artifact preparation

Build all five scored artifacts from the same pinned BF16 source. The recovered mask controls which
original source experts are omitted; no intermediate BF16-pruned checkpoint and no MLX expert
weights are used.

For corrected scored builds, always use new empty output directories and omit `--resume`, as shown
below. Safe receipt-checked resume remains useful for an interrupted build of the same frozen plan;
a missing or mismatched per-file receipt forces that projection to be rebuilt.

    python3 tools/prepare_mixed_expert_repack.py test
    python3 tools/prepare_mixed_expert_repack.py probe /data/models/hy3-source \
      --layer 1 --expert 0 --projection gate

    python3 tools/prepare_mixed_expert_repack.py prepare \
      /data/models/hy3-source /data/artifacts/plain-quant \
      --fallback-dir /data/models/hy3-source --plan /data/plans/plain-quant.json \
      --workers 4

    python3 tools/prepare_mixed_expert_repack.py prepare \
      /data/models/hy3-source /data/artifacts/plain-reap-quant \
      --fallback-dir /data/models/hy3-source --plan /data/plans/plain-reap-quant.json \
      --workers 4

    python3 tools/prepare_mixed_expert_repack.py prepare \
      /data/models/hy3-source /data/artifacts/plain-reap-mix-quant \
      --fallback-dir /data/models/hy3-source --plan /data/plans/plain-reap-mix-quant.json \
      --workers 4

    python3 tools/prepare_mixed_expert_repack.py prepare \
      /data/models/hy3-source /data/artifacts/mix-quant \
      --fallback-dir /data/models/hy3-source --plan /data/plans/mix-quant.json \
      --workers 4

    python3 tools/prepare_mixed_expert_repack.py prepare \
      /data/models/hy3-source /data/artifacts/mix-quant-prune25 \
      --fallback-dir /data/models/hy3-source --plan /data/plans/mix-quant-prune25.json \
      --workers 4

    for arm in plain-quant plain-reap-quant plain-reap-mix-quant mix-quant mix-quant-prune25; do
      python3 research/per-expert-quant/validate_artifact.py \
        "/data/artifacts/$arm" --verify-sources
    done

Every retained expert must appear in the plan; omission is an error, not a BF16 fallback. All arms
resolve non-expert tensors from the same pinned source so router, attention, shared experts,
tokenizer, and prompt template are byte-identical.

Pruning through a v2 plan does not renumber experts or shrink router tensors. bw24 keeps the
original router width, masks declared ids before selection, and only loads retained weights. This
makes trace ids and cross-arm comparisons stable.

## Experimental arms

The only scored arms are `plain_quant`, `plain_reap_quant`, `plain_reap_mix_quant`, `mix_quant`, and
`mix_quant_prune25`, in that predeclared order. Matched random-budget controls are diagnostic
appendices, not replacements for the five arms. No BF16 arm is scored. Keep router, attention,
shared experts, tokenizer, prompt template, sampling, dense fallback, runtime commit, and
calibration trace fixed.

After that frozen screen, `traffic_mix_quant` is a promoted follow-up candidate and must reuse the
same runtime, harness, samples, and full-bank calibration trace. It does not retroactively change
the five-arm taxonomy.

## Target-machine bring-up

Build the exact feature commit and run CPU gates before loading a model:

    cargo test -p bw24-gguf --lib
    cargo test -p bw24-engine --lib mixed_expert_loader
    cargo build --release -p bw24-server -p bw24-engine

Serve one clean arm at a time:

    BW24_COMPAT=openai \
    BW24_MODELS=plain_reap_mix_quant=/data/artifacts/plain-reap-mix-quant \
    BW24_ADDR=127.0.0.1:8080 \
    ./target/release/bw24-server

Before public evaluation, retain raw logs from the required CUDA gates:

    ./target/release/kernel-check 2>&1 | tee kernel-check.log
    ./target/release/run-gen /data/artifacts/plain-reap-mix-quant --prompt "gate prompt" \
      2>&1 | tee run-gen.log
    for k in 1 2 3 4 5 6 7 8; do
      BW24_SPEC_K=$k ./target/release/run-spec /data/artifacts/plain-reap-mix-quant \
        2>&1 | tee "run-spec-k${k}.log"
    done

`kernel-check` includes a model-independent Q2_K CPU-vs-GPU oracle; its raw target-machine log is a
required artifact before trusting the Q2 tier. No correctness, quality, or throughput claim is made
from the development host.

Run the five arms through one matched performance invocation after staging them under the same
local-NVMe root:

    ARTIFACT_ROOT=/scratch/artifacts \
      RUN_ID=g7e-optimized-YYYYMMDD \
      research/per-expert-quant/run_performance_evals.sh

When spill is storage-bound, run a fresh-server, expert-file-only cold-cache A/B before the
capability panel. Use `SPILL_IO=mmap` to choose a page-readahead window, or `worker` plus an explicit
pinned-buffer depth to measure the implemented read/compute overlap path:

    ARTIFACT=/scratch/artifacts/plain-quant \
      PROMPT=/data/runs/spill-profile/mmlu-pro-history-doc0-5shot.txt \
      WINDOWS="8 1 4" \
      SPILL_IO=worker PREAD_DEPTH=8 \
      OUT_ROOT=/data/results/spill-prefetch \
      research/per-expert-quant/run_spill_prefetch_ab.sh

Before integrating a second runtime backend, measure bounded explicit reads against the same real
expert files. This is a read-only storage-side upper bound; it does not include H2D or inference:

    research/per-expert-quant/bench_explicit_reads.py \
      /scratch/artifacts/plain-quant/experts \
      --files 32 --workers 2 --chunk 3538944 --advice random

The matched performance runner fixes a 512-token synthetic prompt, three warmed prefill
repetitions, and three fresh 128-token eager-decode processes per arm. It enables the shared cache,
grouped dispatch, H2D prefetch, and mmap page prefetch for every arm, and records exact
manifest/directory bytes, the GPU
and code revision, peak process memory, and raw timing logs. G7e numbers are research-host results;
the local RTX 5090 run remains the final deployment-performance result.

## Public evaluation

Start with the directional candidate panel: one fixed example each from GPQA Diamond, MATH-500,
MMLU-Pro history/other knowledge, economics, law, and psychology (`LIMIT=1`). Candidate generations
are capped at 256 tokens so this remains a screening run; override `LIMIT` or `MAX_GEN_TOKS` only
when deliberately expanding it. Set `LIMIT=all` only for the final promoted artifacts to run every
available sample in the candidate tasks. This is a promotion gate, not a leaderboard score; run the
full suites only after the size/quality direction is clear.

With the revisions pinned in `suite.lock.json`, `LIMIT=all` is 4,746 requests per arm: 198 GPQA
Diamond, 500 MATH-500, and 4,048 across the five selected MMLU-Pro domains. Estimate runtime from
the completed matched screen before launching the final pair; this is intentionally much larger
than the 350-request `LIMIT=50` screen.

The pinned candidate tasks contain 4,746 evaluation documents per arm: GPQA Diamond 198,
MATH-500 500, and MMLU-Pro history/other/economics/law/psychology
381/924/844/1,101/798. At roughly 70 seconds per generation on the current G7e spill setup, a full
arm is about 93 hours. `LIMIT=all` therefore defaults to a 432,000-second (five-day) external
timeout; bounded screens retain the 14,400-second default. Override `EVAL_TIMEOUT_S` deliberately
when measured throughput changes. Every run receipt records the task list, requested limit, and
effective timeout.

Do not reuse that rough estimate after a matched bounded run completes. Derive each task's measured
seconds/request from the task boundaries in its frozen progress log, preserve source hashes, and
project the pinned full counts with:

    python3 research/per-expert-quant/project_full_runtime.py \
      --run-dir /data/results/per-expert-quant/promoted-n50/plain_quant/RUN_ID \
      --json-out /data/results/per-expert-quant/promoted-n50/_runs/RUN_ID/runtime-projection.json \
      --markdown-out /data/results/per-expert-quant/promoted-n50/_runs/RUN_ID/runtime-projection.md

The projector rejects incomplete runs, unexpected task order/counts, and missing task-boundary
snapshots. Recompute it after any accepted concurrency or spill-setting change.

Before committing multiple machine-days to the two full arms, first tune the bounded worker ring.
The clean depth-8 N=50 run showed material buffer waits and ring-full fallbacks, while each depth-8
pool uses only about 28 MiB of pinned memory. Run matched `LIMIT=5`, `NUM_CONCURRENT=1` preflights
at depths 8, 16, and 32 on the plain artifact, restarting the server for every depth. Require all 35
documents, identical raw and filtered generations, zero server/spill errors and short reads, and
completed receipts. Validate and select the setting with `select_transport_preflight.py`, retaining
depth 8 unless a larger passing pool improves wall time by at least 5%. Preserve the spill-counter
deltas because lower ring pressure without lower elapsed time is not sufficient promotion evidence.

    python3 research/per-expert-quant/select_transport_preflight.py \
      --baseline /data/results/per-expert-quant/promoted-n50/plain_quant/RUN_ID \
      --candidate d8=/data/results/per-expert-quant/spill-depth-preflight/plain-d8/RUN_ID \
      --candidate d16=/data/results/per-expert-quant/spill-depth-preflight/plain-d16/RUN_ID \
      --candidate d32=/data/results/per-expert-quant/spill-depth-preflight/plain-d32/RUN_ID \
      --expected-limit 5 --dimension spill_depth --safe-setting 8

At the selected depth, measure `NUM_CONCURRENT=1,2,4` on the plain artifact with `LIMIT=10`. Accept
a higher value only when the server has no errors, it improves matched wall time by at least 5%,
and the raw generations remain identical to the matched N=50 subset:

    python3 research/per-expert-quant/compare_eval_generations.py \
      --baseline /data/results/per-expert-quant/promoted-n50/plain_quant/RUN_ID \
      --candidate /data/results/per-expert-quant/concurrency-preflight/plain-c2/RUN_ID \
      --candidate-subset

    python3 research/per-expert-quant/select_transport_preflight.py \
      --baseline /data/results/per-expert-quant/promoted-n50/plain_quant/RUN_ID \
      --candidate c1=/data/results/per-expert-quant/concurrency-preflight/plain-c1/RUN_ID \
      --candidate c2=/data/results/per-expert-quant/concurrency-preflight/plain-c2/RUN_ID \
      --candidate c4=/data/results/per-expert-quant/concurrency-preflight/plain-c4/RUN_ID \
      --expected-limit 10 --dimension num_concurrent --safe-setting 1

The selector rejects incomplete receipts, wrong task/count/config settings, non-identical raw or
filtered generations, missing results/logs, retries and runtime errors, and invalid spill deltas.
It also requires one fixed spill depth across a concurrency comparison, records evidence hashes,
and refuses to overwrite JSON or Markdown reports.

Use the selected depth and fastest passing concurrency for both final arms and record both in each
run receipt. These are transport optimizations, not model-quality variables; fall back to depth 8
and concurrency 1 on any response mismatch, retry, server error, or spill error.

    ARM=plain_quant MODEL=plain_quant ARTIFACT=/scratch/artifacts/plain-quant \
      SERVER_BIN=/data/bin/bw24-server-0c9817c \
      SERVER_LOG=/data/logs/plain-quant-server.log \
      BW24_SPILL_IO=worker BW24_SPILL_PREAD_DEPTH=SELECTED_DEPTH \
      BW24_SPILL_STATS=1 BW24_SERVE_SPEC=0 \
      SUITE=candidate research/per-expert-quant/run_public_evals.sh

The runner rejects missing runtime declarations, non-worker spill, disabled telemetry, speculative
serving, a non-exact health model set, and mismatched `SHARD_ID`/`TASKS_OVERRIDE` before creating an
output directory. The final evidence-directory creation is one exclusive `mkdir`, so concurrent
launchers cannot both pass a stale existence check and merge evidence.

For the matched first-pass screen, use the orchestrator. It refuses existing outputs and stale
listeners, starts a fresh exactly-one-model server for each arm, fixes the spill/cache environment,
and wraps every harness invocation in an external timeout:

    OUT_ROOT=/data/results/per-expert-quant/candidate \
      CACHE_DIR=/data/cache/bw24-public-eval \
      research/per-expert-quant/run_five_arm_candidate_evals.sh

The first pass has one matched sample per task, so do not use the bootstrap comparison below. Build
the strict N=1 table (explicitly without confidence intervals) with:

    python3 research/per-expert-quant/summarize_directional_results.py \
      --out-root /data/results/per-expert-quant/candidate \
      --run-id RUN_ID

After promotion, keep one run ID and identical N for every surviving arm. The promoted summarizer
rejects missing, duplicate, or mismatched samples; reports Wilson and stratified paired-bootstrap
intervals; adds the frozen 24,999,514,624-byte non-expert GGUF body to each expert overlay; and
prints the point-estimate quality/size Pareto frontier. It also validates the exact lm-eval model,
endpoint, concurrency, retry, generation, seed, limit, harness/runtime and artifact identity
contracts; records SHA-256 evidence for every manifest, receipt, result and sample file; and refuses
to overwrite either report output. It also binds every arm to the same copied suite lock and
lm-eval task hashes/versions. The only accepted lock evolution is the later additive
`eval_documents` map, which is required exactly for `--expected-n all`:

    python3 research/per-expert-quant/summarize_promoted_results.py \
      --out-root /data/results/per-expert-quant/promoted-n50 \
      --run-id RUN_ID \
      --arms plain_quant,mix_quant_prune25,traffic_mix_quant \
      --expected-n 50

The N=50 down-selection is frozen in the report: among non-baseline candidates, choose the highest
macro point estimate, breaking an exact tie with the smaller logical model. The report always keeps
`plain_quant` as the full-eval baseline and labels the choice directional. The Pareto label is
descriptive, not evidence of equivalence. Use the paired intervals and the full promoted-candidate
evaluation before making the final quality-retention claim.

For `LIMIT=all`, replace `--expected-n 50` with `--expected-n all`. The summarizer then requires
the pinned per-task counts from `suite.lock.json`—198/500/381/924/844/1,101/798, totaling 4,746
documents per arm—and rejects any incomplete or differently sampled result.

Run a full arm as seven restartable task shards under one arm/run ID. `TASKS_OVERRIDE` may contain
only tasks from the selected suite, and `SHARD_ID` creates an isolated receipt/result directory:

    ARM=plain_quant MODEL=plain_quant ARTIFACT=/scratch/artifacts/plain-quant \
      OUT_ROOT=/data/results/per-expert-quant/final-full-candidate RUN_ID=RUN_ID \
      SUITE=candidate LIMIT=all TASKS_OVERRIDE=gpqa_diamond_cot_zeroshot \
      SHARD_ID=gpqa_diamond_cot_zeroshot EVAL_TIMEOUT_S=432000 \
      SERVER_BIN=/data/bin/bw24-server-0c9817c \
      SERVER_LOG=/data/logs/final-full-plain-quant-server.log \
      BW24_SPILL_IO=worker BW24_SPILL_PREAD_DEPTH=8 BW24_SPILL_STATS=1 \
      BW24_SERVE_SPEC=0 \
      research/per-expert-quant/run_public_evals.sh

Repeat for each pinned candidate task, skipping only a shard that already has a validated
`results_*.json`. The full summarizer accepts either one monolithic result or these task shards,
requires byte-identical artifact manifests across shards, and requires exactly one aggregate and
sample log for every pinned task. Never combine shards from different model, server, harness,
artifact, generation, concurrency, or spill configurations. Full reporting rejects missing receipt
fields and compares the declared server/harness/tooling hashes, timeout, generation cap,
concurrency, spill depth, and spec setting across every shard and both arms.
Each receipt is written as incomplete before generation and finalized with wall time plus evaluator
and log-pipeline exit codes. Full reporting rejects receipts that are unfinished, timed out, failed,
or missing a positive elapsed time; concurrency preflights use that recorded wall time. At shard
completion the runner snapshots the active server log into the immutable shard directory and binds
its SHA-256 in the receipt; the live source-log path remains a separate matched configuration field.
The runner refuses every pre-existing run/shard directory, whether complete or partial. Preserve a
failed attempt by renaming it with a timestamp before retrying; never overwrite or mix evidence.
Pass the active server log to every runner. Receipts capture spill counters before and after each
preflight/shard and store the delta. Full reporting requires positive reads/bytes, monotonic
counters, and zero read errors and short reads; fallbacks, waits, and ring-full events remain
reported performance evidence rather than automatic correctness failures.

SWE-bench Verified and Terminal-Bench 2.x use their containerized agent harnesses rather than
`lm-eval`. Use small, frozen task lists with the same agent scaffold and budgets for initial
screening, then run their complete public suites only for promoted artifacts.

The directional practical-task panel is pinned in `practical-evals.lock.json`: 12 SWE-bench
Verified instances covering all 12 repositories and four difficulty strata, plus 12 CPU-only
Terminal-Bench 2 tasks spanning at least nine categories. The lock records the SWE dataset,
parquet, original harness, Harbor adapter and per-task revisions and the Terminal-Bench Harbor
dataset, harness and per-task content digests. Both suites run through Harbor's same frozen
Terminus-2 scaffold. Validate downloaded source material before running any agent:

    python3 research/per-expert-quant/validate_practical_eval_lock.py \
      --swe-parquet /data/evals/SWE-bench_Verified/data/test-00000-of-00001.parquet \
      --swe-harbor-root /data/evals/harbor-swe-verified \
      --terminal-root /data/evals/terminal-bench-2

The practical server must include `/v1/chat/completions`; it renders the full system/user/assistant
history with the model's own GGUF chat template. Use `openai/$ARM`, API base
`http://127.0.0.1:8080/v1`, temperature 0, JSON Terminus parser, 20 turns, 8192 input tokens, 512
output tokens, one attempt, one concurrent trial, and zero harness retries exactly as locked. The
OpenAI client still needs a non-empty dummy API key when `BW24_API_KEY` is unset.

After source validation and candidate promotion, launch one immutable panel at a time:

    ARM=plain_quant PANEL=swe RUN_ID=practical-RUN-ID \
      ARTIFACT=/scratch/artifacts/plain-quant \
      SERVER_BIN=/data/bin/bw24-server-practical \
      SERVER_LOG=/data/logs/practical-plain-server.log \
      BW24_SPILL_IO=worker BW24_SPILL_PREAD_DEPTH=SELECTED_DEPTH \
      BW24_SPILL_STATS=1 BW24_SERVE_SPEC=0 \
      research/per-expert-quant/run_practical_evals.sh

The runner refuses existing output, requires Docker and Harbor 0.18.0, checks exact server health
and the chat route without generating tokens, pins all 12 task names by dataset digest, writes the
resolved Harbor config before execution, and finalizes a hashed run receipt. Use one server and one
runner only; the receipt binds the exact artifact manifest, server binary, transport declarations,
and per-panel spill deltas. Repeat the identical run for the selected finalist and then for
`PANEL=terminal`.

After both arms finish a panel, validate every receipt, resolved agent config, task/digest, trial,
reward, exception, retry and spill invariant and produce the paired size/quality table with:

    python3 research/per-expert-quant/summarize_practical_results.py \
      --baseline /data/results/practical/plain_quant/swe/RUN_ID \
      --candidate /data/results/practical/FINALIST/swe/RUN_ID \
      --panel swe --json-out practical-swe.json --markdown-out practical-swe.md

The report includes exact finished logical bytes, paired wins/losses/ties and an exact sign test.
It remains a directional 12-task panel, not evidence of full-suite equivalence.

Run the same deterministic scaffold, tool permissions, turn/token budgets, task order, transport
configuration, and one trial per task for both arms. Never execute the SWE harness, generated code,
or terminal agents on the host; Docker isolation is mandatory. These 12+12 scores are directional,
not full-benchmark estimates. Promote to the complete 500-task SWE-bench Verified and 89-task
Terminal-Bench 2 suites only after the matched panel supports the finalist.

The generation-only core suite contains IFEval, GSM8K CoT, BBH CoT few-shot, and DROP. HumanEval
and MBPP are isolated as a code suite because their scorers execute generated Python. Run that
lane only in a disposable sandbox.

    ARM=plain_reap_mix_quant \
    MODEL=plain_reap_mix_quant \
    ARTIFACT=/data/artifacts/plain-reap-mix-quant \
    research/per-expert-quant/run_public_evals.sh

    # Transport/config smoke:
    ARM=plain_reap_mix_quant MODEL=plain_reap_mix_quant \
      ARTIFACT=/data/artifacts/plain-reap-mix-quant LIMIT=2 \
      research/per-expert-quant/run_public_evals.sh

    # Unsafe code lane, inside a sandbox only:
    ARM=plain_reap_mix_quant MODEL=plain_reap_mix_quant \
      ARTIFACT=/data/artifacts/plain-reap-mix-quant \
    SUITE=code BW24_UNSAFE_EVALS=1 research/per-expert-quant/run_public_evals.sh

Run all five arms in the predeclared order. Compare them with:

    python3 research/per-expert-quant/summarize_results.py \
      --baseline research/per-expert-quant/results/plain_quant/RUN_ID \
      --candidate plain_reap_quant=research/per-expert-quant/results/plain_reap_quant/RUN_ID \
      --candidate plain_reap_mix_quant=research/per-expert-quant/results/plain_reap_mix_quant/RUN_ID \
      --candidate mix_quant=research/per-expert-quant/results/mix_quant/RUN_ID \
      --candidate mix_quant_prune25=research/per-expert-quant/results/mix_quant_prune25/RUN_ID \
      --out research/per-expert-quant/results/comparison.md

Publish per-task scores, paired 95% bootstrap intervals, artifact bytes, tier counts, pruned
counts, peak VRAM/RAM, prefill/decode throughput, N, thermal regime, failures, exclusions, exact
commits, trace/plan hashes, and manifests. Do not collapse the report to perplexity or one macro
average.

Primary references: [REAP](https://github.com/CerebrasResearch/reap),
[Hy3 REAP50 model card](https://huggingface.co/pipenetwork/Hy3-REAP50-MLX-4bit),
[llama.cpp tensor encodings](https://github.com/ggml-org/llama.cpp/wiki/Tensor-Encoding-Schemes),
and [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
