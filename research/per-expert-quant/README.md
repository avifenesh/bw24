# Hy3 spilling and quantization research pack

This lane owns two deliverables: spill-path improvements for large expert banks and a controlled
four-arm quantization study. Every retained routed expert is quantized; there is no BF16 expert
evaluation arm or BF16 expert fallback. Model loading, CUDA correctness, artifact generation,
research measurement, calibration, and public evaluation happen on the provisioned G7e machine.
The local RTX 5090 rig remains bw24's deployment and final performance target; runtime defaults are
not flipped until the completed code and artifacts pass the same correctness, memory, and throughput
gates there.

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

BF16 Hy3 is common source material only. It is never scored. The public MLX REAP50 checkpoint is a
mask donor only; none of its already-quantized expert weights enter a scored artifact.

Q2 means GGUF Q2_K (2.625 effective bits/weight), Q3 means Q3_K (3.4375 bits/weight), and
NVFP4 is bw24's 64-value/36-byte block format (4.5 bits/weight). The mixed path is correctness
first: Q2_K uses the generic staged f32-dequant kernel until a dedicated target-rig-gated fast
kernel exists.

## What is implemented

- BW24_MOE_TRACE=/path records routed expert ids without changing normal runs.
- tools/build_expert_tier_plan.py emits calibration-independent uniform plans or aggregates frozen
  calibration traces for usage-ranked plans, accepts original-id expert masks, and records all
  trace/mask hashes.
- tools/recover_hy3_reap_mask.py reconstructs the public REAP50 original-id mask from router rows,
  requires one-to-one high-margin matches, and independently checks correction biases.
- tools/prepare_mixed_expert_repack.py streams BF16/F16/F32 or stacked MLX-affine experts on CPU
  and writes Q2_K, Q3_K, and NVFP4 byte ranges. Bounded `--workers` parallelism preserves exact
  expert order and is byte-compared against the single-worker path. Every active expert projection
  must be assigned.
- A v2 overlay can reuse a complete manifest repack for dense, attention, router, tokenizer, and
  shared-expert tensors. Expert data is stored in one mixed file per layer/projection.
- Per-expert overlay entries remain zero-copy mmap windows in `HostExps`; the 161 GB full-bank
  control does not materialize an impossible second copy in 124 GB host RAM.
- Contiguous all-active/all-NVFP4 entries are coalesced back into a uniform mmap slab, preserving
  the uniform fused dispatch path for `plain_quant`; pruned or mixed arms retain per-expert layouts.
- Optional pruned_experts masks preserve original router width and expert ids. Masked experts are
  excluded before top-k and have no weight bytes in the artifact.
- HostExps carries qtype, row bytes, byte extent, and offset per expert. Mixed/pruned layers stay
  on metadata-aware staged, SLRU-cache, or grouped paths; uniform fused kernels remain
  uniform-only.
- validate_artifact.py checks expert coverage, allowed qtypes, non-overlapping byte ranges, total
  bytes, contamination metadata, and optional source fingerprints.
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

Build all four scored artifacts from the same pinned BF16 source. The recovered mask controls which
original source experts are omitted; no intermediate BF16-pruned checkpoint and no MLX expert
weights are used.

    python3 tools/prepare_mixed_expert_repack.py test
    python3 tools/prepare_mixed_expert_repack.py probe /data/models/hy3-source \
      --layer 1 --expert 0 --projection gate

    python3 tools/prepare_mixed_expert_repack.py prepare \
      /data/models/hy3-source /data/artifacts/plain-quant \
      --fallback-dir /data/models/hy3-source --plan /data/plans/plain-quant.json \
      --workers 4 --resume

    python3 tools/prepare_mixed_expert_repack.py prepare \
      /data/models/hy3-source /data/artifacts/plain-reap-quant \
      --fallback-dir /data/models/hy3-source --plan /data/plans/plain-reap-quant.json \
      --workers 4 --resume

    python3 tools/prepare_mixed_expert_repack.py prepare \
      /data/models/hy3-source /data/artifacts/plain-reap-mix-quant \
      --fallback-dir /data/models/hy3-source --plan /data/plans/plain-reap-mix-quant.json \
      --workers 4 --resume

    python3 tools/prepare_mixed_expert_repack.py prepare \
      /data/models/hy3-source /data/artifacts/mix-quant \
      --fallback-dir /data/models/hy3-source --plan /data/plans/mix-quant.json \
      --workers 4 --resume

    for arm in plain-quant plain-reap-quant plain-reap-mix-quant mix-quant; do
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

The only scored arms are `plain_quant`, `plain_reap_quant`, `plain_reap_mix_quant`, and
`mix_quant`, in that predeclared order. Matched random-budget controls are diagnostic appendices,
not replacements for the four arms. No BF16 arm is scored. Keep router, attention, shared experts,
tokenizer, prompt template, sampling, dense fallback, runtime commit, and calibration trace fixed.

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

## Public evaluation

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

Run all four arms in the predeclared order. Compare them with:

    python3 research/per-expert-quant/summarize_results.py \
      --baseline research/per-expert-quant/results/plain_quant/RUN_ID \
      --candidate plain_reap_quant=research/per-expert-quant/results/plain_reap_quant/RUN_ID \
      --candidate plain_reap_mix_quant=research/per-expert-quant/results/plain_reap_mix_quant/RUN_ID \
      --candidate mix_quant=research/per-expert-quant/results/mix_quant/RUN_ID \
      --out research/per-expert-quant/results/comparison.md

Publish per-task scores, paired 95% bootstrap intervals, artifact bytes, tier counts, pruned
counts, peak VRAM/RAM, prefill/decode throughput, N, thermal regime, failures, exclusions, exact
commits, trace/plan hashes, and manifests. Do not collapse the report to perplexity or one macro
average.

Primary references: [REAP](https://github.com/CerebrasResearch/reap),
[Hy3 REAP50 model card](https://huggingface.co/pipenetwork/Hy3-REAP50-MLX-4bit),
[llama.cpp tensor encodings](https://github.com/ggml-org/llama.cpp/wiki/Tensor-Encoding-Schemes),
and [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
