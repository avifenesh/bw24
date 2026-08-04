# Qwen 3.8 27B — release watch note (2026-08-03)

## 2026-08-04 re-check — NOT OUT, no new arch info

Checked 2026-08-04 (all fetched live):

- **HF org `Qwen`: no 3.8 repo, no stub.** Newest org creations remain the 2026-06-26
  ASR/ForcedAligner drops (`/api/models?author=Qwen&sort=createdAt` — nothing after
  June). HF-wide `search=Qwen3.8` returns only two unrelated community finetunes of
  Qwen3-4B ("Ma7ee7/Qwen3.8_4B_Distilled*", July 30 — NOT the release). unsloth org: no
  Qwen3.8 repos either.
- **Upstream code: zero signal.** GitHub search `qwen3.8` in `ggml-org/llama.cpp`
  issues/PRs: 0 hits. `qwen3_8`/`qwen3.8` in `huggingface/transformers` PRs: 0 hits.
  No converter/arch intel available yet.
- **Timeline consolidates on "week of 2026-08-10"**: TechTimes 2026-08-03 states both
  open-weight releases "expected the week of August 10 via Alibaba Cloud Model Studio";
  MarkTechPost 2026-08-03 repeats "next week" and confirms **no benchmark table, no
  license, no activated-parameter count published**. AINews/latent.space 2026-08-04
  coverage adds no arch facts. Qwen blog page (qwen.ai/blog?id=qwen3.8) renders empty
  to non-JS fetch — no arch summary grabbed yet.
- **Qwen3.8-Max is MoE (2.4T total, active count unpublished)** per all coverage — that
  is the *Max*, not the 27B; no source claims the 27B changed class. Pre-announcement
  preview coverage (aireiter, July 19 "Kaleb" LMArena codename) is Max-only, nothing
  on 27B arch.
- **FP8 precedent sharpened** (for the FP8-ST leg): `Qwen/Qwen3.6-27B-FP8` (created
  2026-04-21, same-day sibling of the BF16 repo) carries `quantization_config`:
  `quant_method fp8`, `fmt e4m3`, `activation_scheme dynamic`,
  `weight_block_size [128,128]`, vision tower excluded (`modules_to_not_convert`), MTP
  1 layer + vocab 248320 + 64 layers unchanged from BF16. Expected 3.8 repo pair:
  `Qwen/Qwen3.8-27B` + `Qwen/Qwen3.8-27B-FP8` same-day — the FP8 sibling is the
  FP8-ST-leg artifact and it is block-128, exactly the class the merged loader gates
  (`MEMRA_FP8_BLK_GPU` byte-exact dequant; cuBLASLt refuses block-scaled FP8 on sm_120
  per research/fp8st-20260803/P1-VERDICT.md).
- GDN / MTP-head / vocab for 3.8-27B: **still unknown, nothing leaked** — the "What is
  NOT known" section below stands unchanged.

## Confirmed public signals (fetched 2026-08-03, receipts below)

- **Official announcement exists**: @Alibaba_Qwen on X, 2026-08-03 (~7h before this note):
  "Next week, the open weights of Qwen3.8-Max will be released, and Qwen3.8-27B is also
  going open-weights" — Qwen3.8-Max described as 2.4T params, coding/cowork focus.
  https://x.com/Alibaba_Qwen/status/2084100707423289643 (via websearch snippet; X not
  fetched directly).
- TechNode 2026-08-03 corroborates: Qwen3.8 launched (API live on Alibaba's platform),
  Qwen3.8-Max open-source next week, Qwen3.8-27B to be released as open-source.
  https://technode.com/2026/08/03/alibaba-launches-qwen3-8-with-2-4-trillion-parameters/
  (fetched full page 2026-08-03; cites National Business Daily
  https://www.nbd.com.cn/articles/2026-08-03/4530010.html).
- unsloth (@UnslothAI) already announced intent: "Qwen3.8-27B is coming! Will run locally on
  17GB RAM/VRAM setups" — implies they will ship GGUF quants quickly, as they did for 3.6
  (our unsloth-27b artifact lineage). https://x.com/UnslothAI/status/2084110664789024769.
- Timeline matches the expected ~week of 2026-08-10.

## What is NOT known (do not invent)

- **Architecture: unknown.** No config.json, no model card, no paper as of this note. Whether
  3.8-27B keeps the 3.6-27B hybrid (GDN linear-attn 3:1, head_dim 256, vocab 248320, MTP 1
  layer) is unverified. The prep assumption "same architecture" is an expectation, not
  evidence — the runbook's arch-diff step is the verification.
- Whether an FP8/NVFP4 official quant ships alongside BF16: unknown (3.6 precedent: NVIDIA
  published an official NVFP4 repo ~5 weeks after release; unsloth NVFP4 came earlier).
- Whether the MTP head ships in-checkpoint: unknown (3.6 shipped it as mtp.* bf16 tensors
  inside the modelopt export and as text_config.mtp_num_hidden_layers=1).
- 17GB claim from unsloth suggests a ~27B dense at 4-5 bpw — consistent with same-size-class,
  but it is a tweet, not a spec.
- One contrary signal for calibration: an r/LocalLLaMA thread (1 week old, pre-announcement)
  claimed the small-dense team was disbanded and 3.6 27B/35B might be the last of the line
  — now refuted by the official 27B announcement, kept here as a reminder that rumor-class
  sources ran both directions this week.

## Where it will appear + artifact grab order

1. **HF org `Qwen`** — expected repo name `Qwen/Qwen3.8-27B` (pattern: Qwen/Qwen3.6-27B;
   verify the exact casing when live). Poll: `huggingface-cli repo info Qwen/Qwen3.8-27B`
   or the org page https://huggingface.co/Qwen (sort by created).
2. **Qwen blog** https://qwen.ai/blog + **@Alibaba_Qwen on X** — the announcement usually
   lands minutes before/after the HF repo goes public and carries the arch summary + bench
   table (grab the arch claims for the arch-diff cross-check).
3. Grab order (runbook §1): `config.json` + `tokenizer.json` + `tokenizer_config.json` +
   `chat_template.jinja` FIRST (arch-diff unblocks on ~20 MB), full safetensors in the
   background, `generation_config.json` (EOS semantics).
4. Official GGUF: check for a `Qwen/Qwen3.8-27B-GGUF` sibling repo (3.6 had none — expect
   unsloth to be first again: watch https://huggingface.co/unsloth for `Qwen3.8-27B-GGUF` /
   `-NVFP4`). Also watch `nvidia/` for a modelopt NVFP4 export (the 3.6 NVFP4 lineage).
5. Qwen3.8-Max (2.4T) is a separate, non-target release — ignore for this lane; it lands
   "next week" too and will dominate the news cycle. Do not confuse Max coverage with 27B
   availability.

## Trip-wires while waiting

- If the 27B page appears with `model_type` != `qwen3_5` family: expect converter work
  before anything runs (runbook §3) — budget shifts by however long the conversion subclass
  takes.
- If reasoning-levels/hybrid-thinking knobs appear (Qwen3.8-Max is advertised with
  agentic/coding focus): chat-template diff matters more than usual — ranks derive with
  template ON (DRAFT-REGIME law 1).
- llama.cpp upstream support speed is a signal, not a dependency: our converter is the house
  fork; check `ggml-org/llama.cpp` PRs mentioning qwen3.8 for arch intel even before config
  drops.
