# Usage-tiered expert compression research pack

This lane tests per-expert quantization and pruning in bw24. Every retained routed expert is
quantized; there is no BF16 expert fallback. Preparation and CPU validation happen here. Model
loading, CUDA correctness, performance measurement, calibration, and public evaluation happen on
the provisioned research machine.

## Target model and frozen recipes

The checked Hy3 REAP50 checkpoint has 96 routed experts per MoE layer (already reduced from 192),
top-8 sigmoid routing, and MoE layers 1 through 79.

Two primary recipes are predeclared:

1. usage-pyramid: rank experts separately in each layer by calibration-set router selection
   count. The hottest 25% use NVFP4, the middle 50% use Q3_K, the coldest 25% use Q2_K, and
   zero-count experts are pruned. Fractions are parameters, but any change creates a new named
   plan before public scores are viewed.
2. reap50-plus25: start from the 96-expert REAP50 checkpoint. The least-used 25% of the original
   192-expert bank (48 experts per layer) use Q2_K; the other 48 retained experts use NVFP4.
   No Q3 tier and no additional prune are applied.

Q2 means GGUF Q2_K (2.625 effective bits/weight), Q3 means Q3_K (3.4375 bits/weight), and
NVFP4 is bw24's 64-value/36-byte block format (4.5 bits/weight). The mixed path is correctness
first: Q2_K uses the generic staged f32-dequant kernel until a dedicated target-rig-gated fast
kernel exists.

## What is implemented

- BW24_MOE_TRACE=/path records routed expert ids without changing normal runs.
- tools/build_expert_tier_plan.py aggregates frozen calibration traces, ranks each layer
  independently, and emits a complete immutable plan with trace hashes.
- tools/prepare_mixed_expert_repack.py streams BF16/F16/F32 or stacked MLX-affine experts on CPU
  and writes Q2_K, Q3_K, and NVFP4 byte ranges. Every active expert projection must be assigned.
- A v2 overlay can reuse a complete manifest repack for dense, attention, router, tokenizer, and
  shared-expert tensors. Expert data is stored in one mixed file per layer/projection.
- Optional pruned_experts masks preserve original router width and expert ids. Masked experts are
  excluded before top-k and have no weight bytes in the artifact.
- HostExps carries qtype, row bytes, byte extent, and offset per expert. Mixed/pruned layers stay
  on metadata-aware staged, SLRU-cache, or grouped paths; uniform fused kernels remain
  uniform-only.
- validate_artifact.py checks expert coverage, allowed qtypes, non-overlapping byte ranges, total
  bytes, contamination metadata, and optional source fingerprints.
- The public eval suite is pinned and stores the served artifact manifest/hash with each run.

## Calibration and plan generation

Calibration data and public evaluation data must be disjoint. Use a representative private or
training-side corpus for routing counts; never use IFEval, GSM8K, BBH, DROP, HumanEval, or MBPP
examples to select tiers.

Capture enough requests to cover the intended deployment distribution:

    BW24_MOE_TRACE=/runs/hy3-calibration.trace \
    BW24_MODELS=hy3=/models/hy3-source \
    ./target/release/bw24-server

The trace format is one line per layer/forward: layer, token count, then comma-separated expert
ids. Multiple trace files may be passed and their SHA-256 hashes are frozen into the plan.

For a full 192-expert Hy3 source, build the usage pyramid and prune zero-count experts:

    python3 tools/build_expert_tier_plan.py \
      --trace /runs/hy3-calibration.trace \
      --recipe usage-pyramid \
      --expert-count 192 \
      --original-expert-count 192 \
      --top-k 8 \
      --layers 1-79 \
      --hot-fraction 0.25 \
      --low-fraction 0.25 \
      --prune-unused \
      --out /plans/hy3-usage-pyramid.json

For the actual local REAP50 checkpoint, build the exact 48 Q2_K / 48 NVFP4 split:

    python3 tools/build_expert_tier_plan.py \
      --trace /runs/hy3-reap50-calibration.trace \
      --recipe reap50-plus25 \
      --expert-count 96 \
      --original-expert-count 192 \
      --top-k 8 \
      --layers 1-79 \
      --out /plans/hy3-reap50-plus25.json

Run the builder self-test before producing plans:

    python3 tools/build_expert_tier_plan.py --self-test

Create at least three matched random controls without changing tier counts or prune masks:

    for seed in 11 29 47; do
      python3 tools/make_random_tier_control.py /plans/hy3-usage-pyramid.json \
        --seed $seed --out /plans/hy3-usage-pyramid-random-$seed.json
    done

## Artifact preparation

The local REAP50 quantization source is already MLX 4-bit. Its dense/router fallback is the
existing complete bw24 Q4_K repack. The commands below quantize experts directly from the MLX
source rather than adding another Q4_K intermediate:

    python3 tools/prepare_mixed_expert_repack.py test
    python3 tools/prepare_mixed_expert_repack.py probe /models/hy3-reap50-mlx \
      --layer 1 --expert 0 --projection gate

    python3 tools/prepare_mixed_expert_repack.py prepare \
      /models/hy3-reap50-mlx \
      /models/hy3-reap50-plus25 \
      --fallback-dir /models/hy3-reap50-q4k-bw24 \
      --plan /plans/hy3-reap50-plus25.json \
      --max-work-mb 512 \
      --resume

    python3 research/per-expert-quant/validate_artifact.py \
      /models/hy3-reap50-plus25 --verify-sources

For the publication arm, prefer a BF16/FP16 REAP50 checkpoint produced from tencent/Hy3 with the
official REAP pipeline, then quantize each expert once. If the public MLX 4-bit checkpoint is used
as the source, label the arm double-quantized and keep it separate; an NVFP4 re-encode cannot
recover precision already removed by the MLX quantizer.

For an indexed BF16/F16/F32 source, fallback-dir may be the same checkpoint, but a complete
fixed-precision bw24 repack is preferred so non-expert tensors are identical across all arms.
Every retained expert must appear in the plan; omission is an error, not a BF16 fallback.

Pruning through a v2 plan does not renumber experts or shrink router tensors. bw24 keeps the
original router width, masks declared ids before selection, and only loads retained weights. This
makes trace ids and cross-arm comparisons stable.

## Experimental arms

Freeze all plans before looking at public scores:

1. uniform_q4k_control: the existing complete Q4_K bw24 repack.
2. usage_pyramid: NVFP4/Q3_K/Q2_K plus zero-count prune from the frozen full-bank trace.
3. reap50_plus25: REAP50, then 48 Q2_K and 48 NVFP4 experts per layer.
4. random_budget_seed_*: at least three per-layer random assignments with the same counts,
   pruned count, total bytes, projections, and fixed seeds as the corresponding candidate.

An uncompressed/BF16 model may be reported as a quality ceiling, but it is not a mixed artifact
and no retained expert in either candidate remains BF16. Keep router, attention, shared experts,
tokenizer, prompt template, sampling, dense fallback, runtime commit, and calibration trace fixed.

## Target-machine bring-up

Build the exact feature commit and run CPU gates before loading a model:

    cargo test -p bw24-gguf --lib
    cargo test -p bw24-engine --lib mixed_expert_loader
    cargo build --release -p bw24-server -p bw24-engine

Serve one clean arm at a time:

    BW24_COMPAT=openai \
    BW24_MODELS=reap50_plus25=/models/hy3-reap50-plus25 \
    BW24_ADDR=127.0.0.1:8080 \
    ./target/release/bw24-server

Before public evaluation, retain raw logs from the required CUDA gates:

    ./target/release/kernel-check 2>&1 | tee kernel-check.log
    ./target/release/run-gen /models/hy3-reap50-plus25 --prompt "gate prompt" \
      2>&1 | tee run-gen.log
    for k in 1 2 3 4 5 6 7 8; do
      BW24_SPEC_K=$k ./target/release/run-spec /models/hy3-reap50-plus25 \
        2>&1 | tee "run-spec-k${k}.log"
    done

Add a dedicated Q2_K GPU oracle to kernel-check on that machine before trusting the Q2 tier.
No correctness, quality, or throughput claim is made from this development host.

## Public evaluation

The generation-only core suite contains IFEval, GSM8K CoT, BBH CoT few-shot, and DROP. HumanEval
and MBPP are isolated as a code suite because their scorers execute generated Python. Run that
lane only in a disposable sandbox.

    ARM=reap50_plus25 \
    MODEL=reap50_plus25 \
    ARTIFACT=/models/hy3-reap50-plus25 \
    research/per-expert-quant/run_public_evals.sh

    # Transport/config smoke:
    ARM=reap50_plus25 MODEL=reap50_plus25 ARTIFACT=/models/hy3-reap50-plus25 LIMIT=2 \
      research/per-expert-quant/run_public_evals.sh

    # Unsafe code lane, inside a sandbox only:
    ARM=reap50_plus25 MODEL=reap50_plus25 ARTIFACT=/models/hy3-reap50-plus25 \
    SUITE=code BW24_UNSAFE_EVALS=1 research/per-expert-quant/run_public_evals.sh

Run the uniform control first, then candidates in a predeclared order. Compare them with:

    python3 research/per-expert-quant/summarize_results.py \
      --baseline research/per-expert-quant/results/uniform_q4k_control/RUN_ID \
      --candidate usage_pyramid=research/per-expert-quant/results/usage_pyramid/RUN_ID \
      --candidate reap50_plus25=research/per-expert-quant/results/reap50_plus25/RUN_ID \
      --out research/per-expert-quant/results/comparison.md

Publish per-task scores, paired 95% bootstrap intervals, artifact bytes, tier counts, pruned
counts, peak VRAM/RAM, prefill/decode throughput, N, thermal regime, failures, exclusions, exact
commits, trace/plan hashes, and manifests. Do not collapse the report to perplexity or one macro
average.

Primary references: [REAP](https://github.com/CerebrasResearch/reap),
[Hy3 REAP50 model card](https://huggingface.co/pipenetwork/Hy3-REAP50-MLX-4bit),
[llama.cpp tensor encodings](https://github.com/ggml-org/llama.cpp/wiki/Tensor-Encoding-Schemes),
and [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
