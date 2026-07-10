# Per-expert mixed-precision research pack

This lane asks whether individual MoE experts can be quantized while the remaining experts retain
their source precision. The code path is prepared here; no model-quality or GPU-performance claim
has been measured on the current machine.

## What is implemented

- `tools/prepare_mixed_expert_repack.py` creates a sparse overlay from an explicit JSON plan.
  Selected BF16 expert projections become Q4_K; unselected experts and all other tensors resolve
  from the untouched Hugging Face checkpoint.
- `HostExps` carries dtype, row size, byte extent, and offset per expert. Uniform checkpoints keep
  the existing resident/fused paths. Mixed layers use the generic staged, SLRU-cache, or grouped
  paths whose dispatch reads the selected expert's metadata.
- `bw24-server` accepts manifest-backed overlays, so the same OpenAI-compatible endpoint can serve
  the BF16 reference and every experimental arm.
- `suite.lock.json` pins the lm-evaluation-harness commit, public dataset revisions, task names, and
  primary metrics. `summarize_results.py` reports paired-bootstrap confidence intervals from the
  logged per-document samples.

The v1 producer supports a BF16 base with selected Q4_K expert projections. The runtime metadata
also supports the existing Q8_0, Q3/4/5/6_K, IQ3_S, IQ4_XS, NVFP4, F32, and BF16 matvec formats.

## Prepare an overlay

Start from an indexed Hugging Face safetensors checkpoint and copy
`plans/example.json` to a named, immutable experiment plan. Public benchmark data must not be used
to choose the experts; derive the plan from a separate calibration split or a predeclared random
seed.

```bash
python3 tools/prepare_mixed_expert_repack.py test

python3 tools/prepare_mixed_expert_repack.py prepare \
  /models/base-bf16 \
  /models/overlays/selected-q4k \
  --plan research/per-expert-quant/plans/selected-q4k.json \
  --max-work-mb 512 \
  --resume
```

The overlay manifest records the plan, plan hash, source config/index hashes, tensor mapping, and
the required `BW24_FULL_PREC=1` runtime setting. The base checkpoint is never modified.

## Experimental arms

Freeze these arms before looking at public-eval scores:

1. `bf16_reference`: the original checkpoint with `BW24_FULL_PREC=1`.
2. `all_q4k`: every routed expert quantized to Q4_K; this measures the conventional uniform policy.
3. `selected_q4k`: the calibration-selected experts quantized under the same per-layer bit budget.
4. `random_q4k_seed_*`: at least three random assignments matched to `selected_q4k` by layer,
   expert count, projections, and total bytes.

Keep router, shared-expert, attention, tokenizer, prompt formatting, sampler, and runtime commit
fixed. Report checkpoint/plan hashes and exact artifact bytes for every arm.

## Remote machine bring-up

The provisioned machine needs a CUDA-capable GPU supported by bw24, enough host/disk capacity for
the BF16 checkpoint plus overlays, Rust/CUDA build tools, Python 3 with NumPy, `uv`, `git`, and
`curl`. Build the exact feature-branch commit, then run the CPU checks before loading a model:

```bash
cargo test -p bw24-gguf --lib
cargo test -p bw24-engine --lib mixed_expert_loader_keeps_each_encoding_and_extent
cargo build --release -p bw24-server -p bw24-engine
```

For each arm, start one clean server process. The mixed overlay must be served with full-precision
fallback enabled:

```bash
BW24_FULL_PREC=1 \
BW24_COMPAT=openai \
BW24_MODELS=selected_q4k=/models/overlays/selected-q4k \
BW24_ADDR=127.0.0.1:8080 \
./target/release/bw24-server
```

Before public evaluation, capture the required target-machine gates:

```bash
./target/release/kernel-check 2>&1 | tee kernel-check.log
BW24_FULL_PREC=1 ./target/release/run-gen /models/overlays/selected-q4k --prompt "gate prompt" \
  2>&1 | tee run-gen.log
for k in 1 2 3 4 5 6 7 8; do
  BW24_FULL_PREC=1 BW24_SPEC_K=$k ./target/release/run-spec /models/overlays/selected-q4k \
    2>&1 | tee "run-spec-k${k}.log"
done
```

The exact commands may need the target model's normal prompt/model arguments. A mixed arm is not
merge- or publication-ready until kernel correctness, greedy argmax, coherent text, determinism,
and K=1..8 self-consistency are recorded on that machine. Always retain raw logs and the concurrent
GPU/process state.

## Public evaluation

The safe core suite uses generation-only tasks because bw24's completions endpoint does not expose
token log-probabilities:

- IFEval: instruction following.
- GSM8K CoT: mathematical reasoning.
- BBH CoT few-shot: broad compositional reasoning.
- DROP: reading comprehension and discrete reasoning.

HumanEval and MBPP are a separate `code` suite because their scorers execute generated Python.
Run that lane only inside an isolated disposable sandbox.

With the server running:

```bash
ARM=selected_q4k MODEL=selected_q4k \
  research/per-expert-quant/run_public_evals.sh

# Fast transport/configuration smoke before a full run:
ARM=selected_q4k MODEL=selected_q4k LIMIT=2 \
  research/per-expert-quant/run_public_evals.sh

# Unsafe code lane, inside a sandbox only:
ARM=selected_q4k MODEL=selected_q4k SUITE=code BW24_UNSAFE_EVALS=1 \
  research/per-expert-quant/run_public_evals.sh
```

Run the BF16 reference first, then every candidate with the same suite lock. Compare arms with:

```bash
python3 research/per-expert-quant/summarize_results.py \
  --baseline research/per-expert-quant/results/bf16_reference/RUN_ID \
  --candidate selected_q4k=research/per-expert-quant/results/selected_q4k/RUN_ID \
  --candidate random_seed_11=research/per-expert-quant/results/random_seed_11/RUN_ID \
  --out research/per-expert-quant/results/comparison.md
```

The report includes aggregate deltas and paired 95% bootstrap intervals. Publish per-task scores,
not only a macro average. Also report model/overlay bytes, peak VRAM/RAM, prefill/decode throughput,
N and thermal regime for performance runs, and every failed or excluded arm.

Upstream references: [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness),
[GSM8K](https://huggingface.co/datasets/openai/gsm8k),
[IFEval](https://huggingface.co/datasets/google/IFEval), and
[HumanEval safety guidance](https://github.com/openai/human-eval).
