# Greedy Degeneration and Speculative Decoding Evaluation: Protocol Survey

**Question:** How should speculative-decoding acceptance and throughput be measured on long generations, given greedy decoding's degeneration-into-repetition failure mode (which inflates acceptance)?

**Context:** This survey documents the field's actual practice and formal foundations for measuring spec-decode performance under greedy/low-temperature settings, where neural text degeneration can artificially inflate acceptance rates by locking draft and target models into the same repetition loop.

---

## 1. Degeneration Theory: Holtzman et al. 2020

**Citation:** Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, Yejin Choi. "The Curious Case of Neural Text Degeneration." ICLR 2020. arXiv:1904.09751 [cs.CL].

### Key Findings

**The likelihood paradox:** Holtzman et al. establish that "using likelihood as a decoding objective leads to text that is bland and strangely repetitive," despite likelihood working well as a training objective. The paper shows "maximization is an inappropriate decoding objective for open-ended text generation."

**Self-reinforcing mechanism:** Maximization-based decoding (greedy and beam search) creates a positive feedback loop. Once a model emits a pattern, "that pattern becomes part of the context for the next step, which can make the same continuation even more likely" (Sebastian Raschka, FAQ summary of Holtzman findings). The paper identifies "surprising distributional differences between human text and machine text" and shows decoding strategy alone "can dramatically effect the quality of machine text."

**Failure mode:** Under greedy decoding, "once the model enters a repetitive state, the expected escape time is infinite" (formalized via Markov chain analysis in follow-up work by arXiv:2512.04419v1, Section 4.3). Repetition only terminates when hitting external limits like `max_tokens`.

### Quantitative Context

**Follow-up empirical work (arXiv:2512.04419v1, 2025):** In production vLLM deployments with long generations:
- Repetition occurred in **75-80% of batch runs** under greedy decoding
- Per-transaction latency increased from ~28 min to 40-160 min (43% to 471% overhead)
- Termination occurred only at `max_tokens` limit
- Three compounding effects: context repetition boosts probability, self-reinforcement increases monotonically with repetition count, and greedy decoding has no mechanism to escape once the loop starts

**No quantitative onset threshold vs context length:** Holtzman et al. and follow-up work do not establish a predictable "degeneration onset" as a function of context or generation length. The phenomenon is path-dependent and prompt-specific, not a deterministic phase transition.

---

## 2. Rejection-Sampling Speculative Decoding: Formal Foundations

### Leviathan et al. 2023 (Google)

**Citation:** Yaniv Leviathan, Matan Kalman, Yossi Matias. "Fast Inference from Transformers via Speculative Decoding." ICML 2023. arXiv:2211.17192 [cs.LG].

**Acceptance rule (Section 2.3, Algorithm 1):**

Sample draft token x ~ q(x). Accept if q(x) ≤ p(x). If q(x) > p(x), reject with probability `1 - p(x)/q(x)`. On rejection, resample from the adjusted distribution:

```
p'(x) = norm(max(0, p(x) - q(x)))
```

In pseudocode: draw `r ~ U(0,1)` and reject when `r > p(x)/q(x)`.

**Distribution-equality theorem (Section 2.3, Appendix A.1):**

"Tokens produced by speculative sampling are distributed identically to those sampled from p(x) alone." Proof decomposes P(x=x') into acceptance and rejection branches:
- Acceptance: `q(x') * min(1, p(x')/q(x')) = min(q(x'), p(x'))`
- Rejection (from adjusted dist): `p(x') - min(q(x'), p(x'))`
- Sum: `min` terms cancel → `P(x=x') = p(x')`

The acceptance rate β (Theorem 3.5): `β = 1 - D_LK(p,q) = Σ_x min(p(x), q(x))`.

**Determinism/reproducibility:**

The paper does not discuss random seed reproducibility. The guarantee is **distributional:** "the output distribution is guaranteed to stay the same" and "without any changes to the outputs" (meaning the output distribution, not bit-for-bit equality). Temperature=0 (greedy) is cast as standard sampling from a distribution with non-max entries zeroed (Section 2.2). Results show higher α and speedups at temp=0, but the paper treats deterministic decoding as a special case of sampling, not as requiring separate verification.

### Chen et al. 2023 (DeepMind)

**Citation:** Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau, Laurent Sifre, John Jumper. "Accelerating Large Language Model Decoding with Speculative Sampling." arXiv:2302.01318 [cs.CL]. Submitted Feb 2, 2023.

**Modified rejection sampling with numerics:** Chen et al. introduce "a novel modified rejection sampling scheme which preserves the distribution of the target model **within hardware numerics**" (abstract, emphasis added). The paper acknowledges floating-point precision and distributed-serving constraints but does not publish the exact modification formula on the abstract page. The core claim: "achieving a 2-2.5x decoding speedup in a distributed setup" on Chinchilla 70B "with no loss of sample quality."

**Key distinction from Leviathan:** Chen et al. emphasize numerics-aware distribution preservation and distributed serving of very large models (70B), while Leviathan et al. present the theoretical foundation. Both use equivalent rejection-based acceptance.

---

## 3. How Spec-Decode Papers Measure Acceptance

### EAGLE-1 (arXiv:2401.15077v3, ICML 2024)

**Citation:** Yuhui Li, Fangyun Wei, Chao Zhang, Hongyang Zhang. "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty." arXiv:2401.15077 [cs.LG].

**Datasets:** Four task types via MT-bench (multi-turn dialogue), HumanEval (code generation), GSM8K (mathematical reasoning), and Alpaca (instruction following). MT-bench emphasized because "it has been employed by the current state-of-the-art, including Lookahead and Medusa."

**Models:** Vicuna (7B, 13B, 33B), LLaMA2-Chat (7B, 13B, 70B), Mixtral 8x7B Instruct.

**Temperature settings:** Both temp=0 and temp=1 tested. "EAGLE demonstrates better acceleration at temperature=0 compared to temperature=1" (Section 4.2). Greedy results emphasized; temp=1 comparisons limited because "Lookahead is confined to greedy decoding, and the non-greedy generation of Medusa does not guarantee lossless performance."

**Batch size:** Batch size 1 (standard from prior speculative-sampling work).

**Metrics:**
- **Walltime speedup ratio:** actual test speedup vs vanilla autoregressive
- **Average acceptance length τ:** average tokens accepted per forward pass
- **Acceptance rate α:** ratio of accepted to generated tokens **for chain drafts only** (not applicable to tree drafts due to multiple candidates per position)

**Generation lengths:** Not specified in the paper.

**Repetition/degeneration handling:** None mentioned. Output quality is "both unnecessary and meaningless" to evaluate because the method is lossless by construction. No repetition penalties or loop detection.

### EAGLE-2 (arXiv:2406.16858v2, EMNLP 2024)

**Citation:** Yuhui Li, Fangyun Wei, Chao Zhang, Hongyang Zhang. "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees." arXiv:2406.16858 [cs.CL].

**Evaluation:** "Three series of LLMs and six tasks" (abstract). Specific dataset names not provided in abstract. Achieves 3.05x-4.26x speedup, 20-40% faster than EAGLE-1.

**Key innovation:** Context-aware dynamic draft tree (replacing EAGLE-1's static tree) using draft-model confidence scores to approximate acceptance rates.

**Temperature, generation lengths, repetition:** Not specified in abstract.

### EAGLE-3 (arXiv:2503.01840v3, NeurIPS 2025)

**Citation:** Yuhui Li, Fangyun Wei, Chao Zhang, Hongyang Zhang. "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test." arXiv:2503.01840 [cs.CL].

**Evaluation:** "Both chat models and reasoning models, evaluated on five tasks" (abstract). Specific datasets, temperature, generation lengths not provided. Reports "up to 6.5x speedup" and "1.38x throughput improvement at batch size 64" under SGLang.

**Repetition:** Not mentioned.

### Medusa (arXiv:2401.10774v1, Jan 2024)

**Citation:** Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D. Lee, Deming Chen, Tri Dao. "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads." arXiv:2401.10774 [cs.LG].

**Datasets:**
- Training: ShareGPT (~60k samples for Vicuna); self-distillation on ShareGPT + UltraChat (~100k samples)
- Evaluation: MT-Bench (quality scores), Alpaca-eval (tree pruning calibration)

**Temperature:** Typical acceptance evaluated at "a fixed temperature of 0.7." Ablation shows "at temperature 0 typical acceptance reverts to greedy decoding" and "an increased temperature will correspondingly result in longer accepted sequences."

**Batch size:** Focus on "batch size of one" (local hosting scenario).

**Metrics:**
- **Acceleration rate:** average tokens decoded per decoding step (1.0 = baseline)
- **Overhead:** per-step latency of Medusa vs vanilla model
- **Speedup:** wall-time rate = acceleration rate / overhead

**Acceptance:** Typical acceptance scheme (typicality threshold ε, entropy parameter α). First token greedy and accepted unconditionally; longest accepted prefix among candidates chosen.

**Results:** Medusa-1 >2.2x speedup; Medusa-2 2.3-3.6x. Acceptance rates (Medusa-2): 3.47 (Vicuna-7B), 3.14 (Zephyr-7B), 3.51 (Vicuna-13B), 3.01 (Vicuna-33B).

**Generation lengths:** Not specified. Training used sequence length 4096 (Vicuna) or 2048 (self-distillation), but per-response max_tokens not stated.

**Repetition/degeneration:** Not analyzed. Quality assessed solely via MT-Bench scores; no repetition metrics. Paper notes "direct sampling from a language model may lead to incoherent or nonsensical results" (motivating typical acceptance), but does not measure repetition in Medusa outputs.

### DeepSeek-V3 MTP (arXiv:2412.19437v1, Dec 2024)

**Citation:** DeepSeek-AI. "DeepSeek-V3 Technical Report." arXiv:2412.19437 [cs.CL].

**MTP design:** Multi-token prediction depth D=1 (each token predicts one future token). Loss weight λ=0.3 (first 10T tokens), λ=0.1 (remaining 4.8T tokens). At inference, MTP modules can be "repurposed for speculative decoding."

**Acceptance rate (from external sources, as Section 5.4.3 was truncated in fetch):**
- Medium.com summary (Bing, Dec 2025): "acceptance rate of MTP1 is above 80%"
- Hugging Face nebius/MTP-DeepSeek-V3-0324 card (Feb 2026): "acceptance metrics reported above were measured under standard rejection sampling"
- GitHub llama.cpp discussion #11455 (Mar 2026): "acceptance rate of the second token prediction ranges between 85% and 90% across various generation topics"
- Real speedup: "1.8x improvement" (machinelearningatscale.substack.com, Feb 2025)

**Evaluation protocol details:** Not available in truncated abstract fetch. The acceptance-rate numbers (85-90%) are widely cited but the underlying evaluation protocol (datasets, generation lengths, temperature) requires the full paper.

**Repetition:** Not mentioned in available excerpts.

---

## 4. How Engines Bench Spec-Decode

### vLLM

**Documentation:** https://docs.vllm.ai/en/latest/features/speculative_decoding/

**Acceptance measurement:** The docs do not describe how acceptance rate is logged. A `synthetic_acceptance_rate` config exists for testing: "Average acceptance rate to target when rejection_sample_method is synthetic." Rejection methods: strict, probabilistic, or synthetic.

**Metrics reported:** No specific acceptance-rate or speedup metrics documented. Users directed to offline example script and "benchmark CLI guide." Method comparison table gives qualitative ratings (e.g., EAGLE "High gain" at low QPS).

**Temperature:** No benchmark temperature defaults stated. Docs explicitly say "temperature and top_p are sampling parameters, not --speculative-config fields." Constraint: probabilistic acceptance (temp>0 draft sampling) "is not yet supported" with heterogeneous vocabularies.

**Correctness guarantees:** vLLM validates that rejection-sampled outputs "align with the target distribution" and confirms "greedy sampling with speculative decoding matches greedy sampling without it." However, "does not currently guarantee stable token log probabilities (logprobs)" across runs due to floating-point precision and batch-size effects.

**Repetition/loop handling:** Not mentioned. Suffix decoding has bounding params: `suffix_decoding_max_spec_factor` caps speculative length as multiple of prefix-match length; `suffix_decoding_min_token_prob` (default 0.1) sets minimum probability to speculate a token.

**Practical note (DigitalOcean guide, Jan 2026):** "A benchmark run at temperature=0 tells you nothing about what happens at temperature=0.8 in production." At temp=0 (greedy), "the model always picks the single most likely next token" (fully deterministic); at higher temps "the model starts choosing from a wider range of possible tokens."

### TensorRT-LLM

**Documentation:** https://nvidia.github.io/TensorRT-LLM/advanced/speculative-decoding.html

**Support:** Speculative sampling via draft model, Medusa, n-gram, ReDrafter, EAGLE (1/2/3), and MTP. ReDrafter implements "logits prediction, beam search, and draft token acceptance inside the TensorRT engine."

**Benchmarking:** Example tutorial (Triton Inference Server guide) states benchmarks "reflect the true performance gains of speculative decoding in real-world, low-concurrency scenarios" but does not specify temperature or generation length.

**Performance note (Baseten blog, Dec 2024):** "Speculative decoding performance degrades when the model is given more creative freedom (e.g. high temperature, high top_k, top_p)" — higher temperature reduces draft-target alignment and lowers acceptance.

**Recent deployment:** NVIDIA blog (Jan 2025) reports "over 3x speedup in total token throughput." GB200/B200 example shows running GPT-OSS-120B with EAGLE-3 spec-decode. Perf benchmarking docs note: "Allow the GPU to dynamically adjust... temperature. While locking clocks at maximum frequency might seem beneficial, it can sometimes lead to thermal throttling."

**Repetition:** Not mentioned in docs.

### llama.cpp

**Documentation:** https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md

**Acceptance measurement:**

Prints per-implementation statistics:
```
draft acceptance rate = 0.57576 (  171 accepted /   297 generated)
```

Per-implementation counters:
- `#gen tokens`: tokens produced (including rejected)
- `#acc tokens`: tokens accepted by main model
- `#gen drafts` / `#acc drafts`: drafts generated vs accepted (partially)

Each implementation (ngram_simple, ngram_mod, ngram_map_k, draft) prints its own line.

**Temperature/sampling defaults:** Not specified in docs. Speculative-specific defaults:
- `--spec-draft-p-min` (minimum greedy probability, default 0.00)
- `--spec-draft-p-split` (split probability, default 0.10)
- `--spec-draft-n-max` default 3, `--spec-draft-n-min` default 0

Benchmarking: SPEED-Bench client "can compare a baseline run against a speculative-decoding run," but no sampler config listed.

**Repetition handling:** None described. Instead, repetition is **exploited** by n-gram methods:
- N-gram approaches "rely on patterns that have already appeared in the generated text"
- `ngram-map-k4v` suggested "if there are a lot of longer repetitions"
- `ngram-mod` applications include code iteration, summarization, and reasoning models "when they have to repeat their thinking in the final answer"
- No mention of repetition penalties or degeneration safeguards

**Real-world empirical finding (HuggingFace discussion, Apr 2026):** On Qwen3.6-35B-A3B + RTX 3090, user tested 19 spec-decode configurations (ngram-cache, ngram-mod, draft with vocab-matched Qwen3.5-0.8B). **Result:** "no variant achieves net speedup on Ampere + A3B MoE." Draft acceptance rate reached 100% on easy/low-temp prompts (confirmed via verbose output: "draft acceptance rate = 1.00000 (115 accepted / 115 generated)"), but overhead exceeded gains.

---

## 5. Repetition Penalty: Keskar et al. 2019 (CTRL)

**Citation:** Nitish Shirish Keskar, Bryan McCann, Lav R. Varshney, Caiming Xiong, Richard Socher. "CTRL: A Conditional Transformer Language Model for Controllable Generation." arXiv:1909.05858 [cs.CL]. Sep 2019.

**Formal definition (as characterized in follow-up work, arXiv:2504.20131v3):**

Applies "a fixed logit penalty that encourages the model to use new tokens." Mechanism: for any token that has already appeared in the context, scale its logit down by a fixed multiplier (typically >1, e.g., 1.2 as standard value suggested by Keskar et al.).

**How it works:**
- Binary penalty: a token is penalized the same whether it appeared once or many times
- No recency decay or windowing ("cannot forget tokens")
- Multiplicative scaling: strength values {1, 1.1, 1.2, 1.25, 1.3, 1.5} tested in practice (arXiv:2504.20131v3)

**Quality distortions documented (arXiv:2504.20131v3, Oct 2025):**

1. **Ineffective at preventing degeneration:** "Does not actually succeed in preventing degenerate repetitions," with degenerate repetition rates ~4% at low temperature (model/task dependent) — "disqualifying for any kind of serious application."

2. **Over-penalization at high strengths:** If set too high, "the sampler becomes unable to use fundamental, necessary, but frequent tokens, such as spaces or periods, resulting in poor completions."

3. **Relative performance:** "Works significantly better than the frequency penalty" and provides "some modest relief," but incomplete solution.

**Spec-decode verification under repetition penalty:**

**No published work found.** Searched "repetition penalty" + "speculative decoding" across arxiv/docs — no paper verifies spec-decode distribution preservation or acceptance behavior when penalized decoding is active. The surveyed spec-decode papers (EAGLE 1/2/3, Medusa, Leviathan, Chen, DeepSeek MTP) never mention repetition penalty in their evaluation protocols.

**Why this matters:** Repetition penalty modifies the target distribution p(x) at decode time. Standard rejection sampling assumes p(x) is the raw model logits. If p(x) includes penalty, both draft and target must apply identical penalty for distribution preservation — but no published work validates this, and llama.cpp/vLLM docs do not clarify whether penalties apply to draft, target, or both.

---

## 6. Deterministic Loop-Breaking Alternatives

### No-Repeat-Ngram Blocking (Paulus et al. 2017, Klein et al. 2017)

**Citations:**
- Romain Paulus, Caiming Xiong, Richard Socher. "A Deep Reinforced Model for Abstractive Summarization." arXiv:1705.04304 (2017).
- Guillaume Klein, Yoon Kim, Yuntian Deng, Jean Senellart, Alexander M. Rush. "OpenNMT: Open-Source Toolkit for Neural Machine Translation." arXiv:1701.02810 (2017).

**Mechanism (HuggingFace blog, Mar 2020):**

"Manually setting the probability of next words that could create an already seen n-gram to 0" — so any token completing a repeated n-gram is excluded from the candidate set. Implemented in `transformers` via `no_repeat_ngram_size` parameter (e.g., 2 blocks any repeated bigram).

**Quality trade-offs:**

- **Effective at removing repetition:** Blog notes result "looks much better!" after eliminating beam-search loops.
- **Must be used with care:** For topics with naturally recurring phrases (e.g., "New York"), a 2-gram penalty means the city name "would only appear once in the whole text!"
- **Tuning required:** "Finding a good trade-off between inhibiting repetition and repeating cycles of identical n-grams requires a lot of finetuning."

**Spec-decode use:** No published work found using no-repeat-ngram blocking in spec-decode evaluation. The technique is **deterministic** (unlike stochastic repetition penalty), so it would preserve bit-for-bit reproducibility at temp=0 if both draft and target apply identical n-gram blocking.

### Bounded Generation Lengths per Prompt Class

**Practice in evaluation harnesses:**

**EleutherAI lm-evaluation-harness:** GitHub issues (e.g., #1426, Feb 2024) discuss "truncation strategy" for few-shot samples, but no explicit repetition-loop truncation documented in main codebase. The framework supports filter pipelines (lm_eval/filters/__init__.py) but standard operation is to generate to task-specific length.

**HELM (Holistic Evaluation of Language Models):** No specific repetition-loop handling documented in search results. Standard practice: each benchmark has a max_tokens setting appropriate to the task (e.g., HumanEval code completion uses modest max_tokens since solutions are typically <1000 tokens; long-form generation tasks may use 2048-4096).

**Practical protocol:** Most benchmarks implicitly avoid long-generation degeneration by:
1. Choosing tasks with naturally bounded outputs (code completion, single-paragraph answers)
2. Setting max_tokens to match expected output length plus safety margin
3. Not measuring generation quality past max_tokens (so loops that hit the limit are truncated but not separately flagged)

### Loop-Detection Truncation

**Recent work acknowledging greedy-loop problem:**

**arXiv:2509.24328v1 (Speculative Verification, Apr 2026, Table 5):** "We exclude greedy decoding because it often produces lower-quality outputs (e.g., repetitive tokens) and is less representative of how LLMs are typically served." Their protocol: use temp>0 with model-specific recommended settings (Qwen: top-k=20, top-p=0.8, temp=0.7, repetition_penalty=1.05; Llama: top-p=0.9, temp=0.6).

**OpenReview forum (Liquid.ai blog, 2 days ago, "Reducing Doom Loops with Final Token Preference Optimization"):** Training-time intervention to reduce doom-looping rate from 10.2% to 1.4% in LFM2.5-2.6B. Acknowledgment: "loops especially at low temperatures or with greedy decoding" are a recognized production problem.

**No standard loop-detection eval protocol found:** None of the surveyed spec-decode papers (EAGLE, Medusa, MTP) use explicit n-gram repeat detectors during evaluation, despite degeneration being a known issue. The field's current approach: avoid the problem via temp>0 or bounded lengths, not detect and truncate.

---

## 7. Synthesis: What the Field Would Do with the bw24 Constraint

**Constraint:** Measure spec-decode acceptance and throughput on **long generations under greedy (temp=0) decoding**, where degeneration can inflate acceptance by locking draft and target into the same loop.

### Three Candidate Protocols

#### (a) Bounded-Length Greedy on Fixed Prompt Sets

**What it is:** Run greedy decoding with max_tokens set to clip generation before loops typically start. Choose prompt sets where correct answers are naturally short (e.g., HumanEval code completion, GSM8K math problems with solutions <500 tokens).

**Precedent:**
- **Strong:** All surveyed spec-decode papers use this implicitly. EAGLE-1/2/3 eval on MT-bench/HumanEval/GSM8K/Alpaca; Medusa on MT-bench/Alpaca; DeepSeek MTP cites acceptance rates without stating lengths but likely uses standard benchmarks.
- **Explicit temp=0 results:** EAGLE-1 emphasizes "EAGLE demonstrates better acceleration at temperature=0 compared to temperature=1" and provides Figure 8 showing acceptance length τ at temp=0 on MT-bench (per ar5iv.labs.arxiv.org/html/2401.15077, Table 5).
- **llama.cpp real-world:** User reports 100% acceptance on "low-temperature / easy prompts" (RTX 3090 + Qwen3.6-35B, Apr 2026), confirming that bounded easy tasks at temp=0 can yield high α without hitting degeneration.

**Publishability:** **Strong.** This is the field's standard practice. You can directly compare to published EAGLE/Medusa/MTP numbers if using the same benchmarks (MT-bench, HumanEval, GSM8K) and reporting temp=0 separately from temp>0.

**Caveats:**
- Does not measure long-context or open-ended generation performance
- Acceptance rates may not generalize to production workloads with longer outputs
- Requires stating max_tokens explicitly (e.g., "MT-bench with max_tokens=512" or whatever EAGLE used, though they don't publish it)

#### (b) Rejection-Sampling Verify with Seeded RNG

**What it is:** Run speculative decoding with temp>0 (e.g., temp=0.7 or 0.9) using a fixed random seed for reproducibility. Verify that the output distribution matches baseline autoregressive at the same seed by:
1. Generating N completions with spec-decode (seed s)
2. Generating N completions with baseline (seed s)
3. Comparing token-level exact match rate (should be 100% if RNG is deterministic and rejection sampling is correct)

**Precedent:**
- **Theoretical:** Leviathan et al. 2023 and Chen et al. 2023 **prove distribution equality** and emphasize "without any changes to the outputs" (Leviathan) and "preserves the distribution of the target model within hardware numerics" (Chen).
- **Practical verification:** vLLM docs state "greedy sampling with speculative decoding matches greedy sampling without it" and samples "align with the target distribution." However, vLLM also warns "does not currently guarantee stable token log probabilities (logprobs)" due to floating-point precision and batch effects — so **bit-for-bit equality is not guaranteed even at temp=0**.
- **Seeded RNG in spec-decode papers:** None of the surveyed papers (EAGLE, Medusa, MTP) describe seeded-RNG verification experiments. The distribution-equality theorem is taken as given from Leviathan/Chen foundational work.

**Publishability:** **Medium.** Seeded-RNG verification is **novel** as an empirical robustness check but not necessary to publish spec-decode work (since the theorem already proves distributional equality). You could include it as a "verification" subsection showing your implementation is correct, but it's not a substitute for standard acceptance-rate benchmarks.

**Caveats:**
- Temp>0 avoids degeneration but yields **lower acceptance rates** (all sources note spec-decode gains degrade at higher temp)
- Floating-point non-determinism (especially in distributed/batched settings) may break exact bit-for-bit match even with seed
- Does not address the original constraint (greedy long-generation measurement)

#### (c) Deterministic Penalized/No-Repeat-Ngram Greedy

**What it is:** Run greedy decoding (temp=0) with **no-repeat-ngram blocking** (e.g., `no_repeat_ngram_size=3`) applied identically to both draft and target models. This deterministically prevents loops while preserving greedy's determinism. Measure acceptance and throughput on long generations (e.g., 2048-4096 tokens) on open-ended prompts.

**Precedent:**
- **No-repeat-ngram in general LLM eval:** HuggingFace blog (Mar 2020) describes it as standard technique for beam search. Widely used in summarization and translation benchmarks (Paulus 2017, Klein 2017 origin).
- **In spec-decode evaluation:** **NONE FOUND.** Zero published spec-decode papers mention using no-repeat-ngram or any repetition penalty in their evaluation protocols.
- **Why not used:** Likely because (1) standard benchmarks avoid the problem via bounded lengths, and (2) modifying the sampling distribution complicates the "lossless" claim (though no-repeat-ngram is lossless in the sense that it's a deterministic truncation of the original distribution).

**Publishability:** **Low to Medium.** This is **novel** — no precedent in spec-decode literature. You would need to:
1. **Justify why it's necessary:** Explain that long-generation greedy measurement is required for [your use case], and that degeneration would artificially inflate both baseline and spec-decode acceptance.
2. **Validate that it's fair:** Show that no-repeat-ngram is applied **identically** to draft and target (so distribution preservation still holds under the modified distribution).
3. **Compare to unpenalized baseline:** Report both penalized-greedy and temp=0.7 results to show where your work fits in the existing landscape.

**Caveats:**
- **Quality distortion:** No-repeat-ngram can harm outputs on topics with naturally recurring phrases (HuggingFace blog example: "New York" only once). Requires careful n-gram size tuning.
- **Not standard practice:** Reviewers may ask why you didn't use temp>0 like everyone else.
- **Unclear how engines handle it:** llama.cpp and vLLM docs do not clarify whether repetition penalties apply to draft, target, or both. You'd need to verify in code or implement it yourself.

---

## 8. Recommendations for bw24

### Publishable Protocol (Precedent-Based)

**Tier 1: Bounded-length greedy on standard benchmarks**

Use MT-bench + HumanEval + GSM8K at **temp=0 with explicit max_tokens** (e.g., 512 for MT-bench, 1024 for HumanEval, 512 for GSM8K — tune based on baseline generation lengths). Report:
- Acceptance length τ (mean accepted tokens per draft)
- Acceptance rate α (accepted / generated, for non-tree methods)
- Walltime speedup vs baseline
- Explicitly state temp=0 and max_tokens in methodology

**Why this works:** Direct comparison to EAGLE-1 Table 5 (temp=0 results on MT-bench), Medusa (temp=0.7 but you can report both), and llama.cpp community benchmarks. Avoids degeneration by clipping before loops start.

**Tier 2: Temperature sweep with distribution verification**

Report acceptance and speedup at **temp ∈ {0.0, 0.7, 1.0}** on the same benchmarks. At temp>0, include a small-scale (N=100 prompts) exact-match verification showing your spec-decode outputs match baseline bit-for-bit (with fixed seed). This proves implementation correctness and shows how acceptance degrades with temperature.

**Why this works:** Standard practice in recent work (e.g., arXiv:2509.24328v1 excludes greedy, uses temp=0.7 with repetition_penalty=1.05). Adds rigor via verification.

### Novel Protocol (If Long-Generation Greedy is Required)

**If your use case genuinely requires long-generation greedy measurement** (e.g., because target application is code refactoring with 4K-token outputs at temp=0), then:

1. **Lead with standard protocol (Tier 1 above)** to anchor your work in existing practice.

2. **Add a "Long-Generation Robustness" section** measuring acceptance on 2048-4096 token generations at temp=0 with **no-repeat-ngram blocking** (e.g., n=4). Report:
   - Acceptance and speedup under penalized-greedy
   - Comparison to temp=0.7 unpenalized on the same prompts
   - Explicit statement that no-repeat-ngram is applied identically to draft and target

3. **Justify the novelty:** Explain that (a) standard benchmarks implicitly avoid degeneration via bounded lengths, (b) long-generation greedy is underexplored due to degeneration risk, and (c) your deterministic loop-breaking method enables fair measurement without inflating acceptance via shared loops.

4. **Validate that it's conservative:** Show that your acceptance numbers under penalized-greedy are **lower** than unpenalized-greedy (because the penalty reduces draft-target alignment on repeated tokens), proving you're not gaming the metric.

**Risk:** Reviewers may still prefer temp>0 results as more representative of real-world usage. Mitigate by reporting **both** protocols and framing penalized-greedy as a "robustness check" rather than the primary claim.

---

## 9. References

### Foundational Papers

1. Holtzman et al. 2020: "The Curious Case of Neural Text Degeneration." ICLR 2020. arXiv:1904.09751 [cs.CL].

2. Leviathan et al. 2023: "Fast Inference from Transformers via Speculative Decoding." ICML 2023. arXiv:2211.17192 [cs.LG].

3. Chen et al. 2023: "Accelerating Large Language Model Decoding with Speculative Sampling." arXiv:2302.01318 [cs.CL].

### Spec-Decode Implementations

4. EAGLE-1: Yuhui Li et al. "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty." ICML 2024. arXiv:2401.15077 [cs.LG].

5. EAGLE-2: Yuhui Li et al. "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees." EMNLP 2024. arXiv:2406.16858 [cs.CL].

6. EAGLE-3: Yuhui Li et al. "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test." NeurIPS 2025. arXiv:2503.01840 [cs.CL].

7. Medusa: Tianle Cai et al. "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads." arXiv:2401.10774 [cs.LG]. Jan 2024.

8. DeepSeek-V3: DeepSeek-AI. "DeepSeek-V3 Technical Report." arXiv:2412.19437 [cs.CL]. Dec 2024.

### Repetition Mitigation

9. Keskar et al. 2019: "CTRL: A Conditional Transformer Language Model for Controllable Generation." arXiv:1909.05858 [cs.CL].

10. Paulus et al. 2017: "A Deep Reinforced Model for Abstractive Summarization." arXiv:1705.04304 [cs.CL].

11. Klein et al. 2017: "OpenNMT: Open-Source Toolkit for Neural Machine Translation." arXiv:1701.02810 [cs.CL].

12. arXiv:2504.20131v3 (Oct 2025): "LZ Penalty: An information-theoretic repetition penalty for autoregressive language models."

13. arXiv:2512.04419v1 (Dec 2025): "Solving LLM Repetition Problem in Production: A Comprehensive Study of Multiple Solutions."

14. arXiv:2509.24328v1 (Apr 2026): "Speculative Verification: Exploiting Information Gain for Speculative Decoding."

### Engine Documentation

15. vLLM: https://docs.vllm.ai/en/latest/features/speculative_decoding/

16. TensorRT-LLM: https://nvidia.github.io/TensorRT-LLM/advanced/speculative-decoding.html

17. llama.cpp: https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md

18. HuggingFace: "How to generate text: using different decoding methods for language generation with Transformers." March 2020. https://huggingface.co/blog/how-to-generate

---

**End of survey. Commit with `docs:` prefix per protocol.**
