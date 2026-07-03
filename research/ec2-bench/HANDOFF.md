# EC2 g6e.12xl handoff — vLLM/SGLang/llama.cpp three-way

## What this box is FOR
The head-to-head we've been blocked on 3 sessions: **vLLM vs SGLang vs llama.cpp** on one clean box.
Laptop blockers (broken venv, `sgl-kernel` pkg-name bug, Qwen3_5 multimodal-registry dead-end, vLLM
never installed) all evaporate on a fresh cloud box.

## What it is NOT for
- **No bw24.** bw24 kernels are compiled `sm_120a` (mxf4nvf4 block_scale, m16n8k32.s8). L40S is
  **sm_89 (Ada)** — those instructions don't exist there. bw24 won't load. No `kernel_check` here.
- **No exact-arch tuning.** That's irreducibly the laptop (RTX 5090, sm_120). Stays there.
- **No native FP4.** Ada = FP8 tensor cores, not Blackwell FP4. The NVFP4 daily model runs **bf16/FP8**
  here, not native NVFP4. That's fine — the goal is engine-vs-engine + reading their kernels.

## g6e.12xl = 4x L40S (sm_89, 48GB ea, 192GB total)
- Single-GPU parity vs laptop: `CUDA_VISIBLE_DEVICES=0`.
- Multi-GPU story (TP=4): run separately — shows the 27B/MoE scaling vLLM/SGLang get that llama doesn't.

## The bridge (ties L40S numbers back to the laptop)
llama.cpp runs the **same f16 GGUF on both boxes** (f16 works on sm_89 AND sm_120).
```
rank = [bw24 / llama]_laptop   composed with   [vLLM, SGLang, llama]_L40S
```
So even though bw24 can't run on Ada, we get a full ranking via the shared llama reference.

## Run order
1. `bash setup.sh`  — deps, torch venv, vLLM, SGLang, llama.cpp (CUDA arch 89), pinned to laptop commit c57607016.
2. download/scp models (see below).
3. `bash bench.sh`  — pp512 + tg128 each engine; results in `~/bench-results`.
4. scp `~/bench-results/*` back to laptop `research/ec2-bench/results/`.

## WHAT I NEED FROM YOU when you pass the box
- **Instance IP + which .pem** (you have many in ~/.ssh — tell me which one + the ssh user, likely `ubuntu`).
- **HF token** — `~/.cache/huggingface/token` is EMPTY on the laptop. vLLM/SGLang download from HF;
  gated Qwen repos need a token. Either `export HF_TOKEN=...` on the box, or scp the 19G HF dirs
  (`qwen35-9b-hf`, `qwen36-27b-text-nvfp4-mtp-hf`) from the laptop (slower uplink ~30min vs ~3min download).
- Confirm the **AMI** (g6e DLAMI ships CUDA 12.x + driver; if bare Ubuntu I install the driver too).

## High-value extra (the copy-then-tune mechanism)
Once engines run, **profile their kernels** with nsys/ncu on the L40S:
- vLLM/SGLang FP8 GEMM + paged-attention kernel structure → the structure is arch-portable even if the
  exact mma isn't. That's the "copy the best kernel per component" input for the laptop sm_120 port.
- Specifically: their decode attention (paged KV) and their prefill GEMM tiling/warp-spec.
