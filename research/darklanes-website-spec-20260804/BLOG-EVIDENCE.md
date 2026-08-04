# Blog-post source material — the receipts behind each candidate post (2026-08-05)

Companion to SPEC.md §8. Every post below is written from material that already exists in
the memra repo — the writer's job is narrative, not new evidence. Paths relative to repo
root (branch restructure/public-split).

## A. The nvcc miscompile catch (candidate: LAUNCH POST)

Commit `93e1b6a1` (2026-08-04); write-up `research/fp8ship-20260804/official/RESULTS.md`
("FINDING FIRST — nvcc 13.0.88 miscompiles the ARM B' dequant kernel").

- nvcc **13.0.88** at -O3/sm_120a drops the first of two adjacent byte stores
  (`dst[0]=lo; dst[1]=hi;` under `if (lane==0)`) in the FP8 block-128 dequant kernel —
  the LOW byte of the Q8_0 `d` f16 scale zeroed. The same commit was green on nvcc 13.1.
- Caught by the `kernel-check` `[fp8-blk-gpu]` bit-parity arm on FIRST run (3968/680/8 bad
  bytes on three shapes); byte-in-block histogram localized every bad byte to offset 0.
- Real-world consequence had it shipped: first-token logits max_abs diff 3.04, greedy
  stream diverged at token 14. "The bit-identity gate did exactly its job."
- 40-line isolated repro committed: `research/fp8ship-20260804/official/miscompile/
  halfstore_repro.cu` (8/8 blocks fail) + `halfstore_fix.cu` (0/8). Fix: one aligned u16
  store in `crates/memra-engine/cu/fp8_blk_dequant.cu`.
- The law extracted: "parity gates re-run per TOOLCHAIN, not per commit."
- Related compiler lore for the post's depth: ptxas C7514/15/17/19 wgmma
  form-sensitivity findings pinned in ARCHITECTURE-H100.md:446, 1372-1398; FLAGS.md:433.

## B. Dogfood F4 — the founder's own agent found two serving bugs (2026-08-04)

Commits 8032ab01 → 6f51d4a1 → 95df4f3c → merged c716954b; receipts
`research/sampledspec-20260804/`.

- Bug 1: `#[serde(default)] temperature: f32` = 0.0 ⇒ every client omitting temperature
  got greedy argmax instead of OpenAI's documented 1.0. Live symptom: the owner's daily
  agent (sends no temperature) locked into ~10 identical tool-call cycles.
- Bug 2: fixing temperature was NOT enough — `#[serde(default)] seed: u64` pinned omitted
  seed to 0, a valid FIXED seed; the loop survived. Differential proof on the pre-fix
  binary is in the commit body (4/4 byte-identical with omitted params; distinct with
  explicit seeds).
- Fixes: default_temperature=1.0; `seed: Option<u64>` + `fresh_seed()` (nanosecond clock
  x atomic counter through SplitMix64 finalizer, never 0). Explicit temp 0 / seed 0
  honored exactly — every determinism gate sends both explicitly.
- The meta-lesson that makes it a post: **every golden in the repo ran temp=0, which
  routes around the sampler chain entirely — a broken sampled-spec path is INVISIBLE to
  argmax goldens** (sample_check.rs top-of-file doc). New gate: sampled-spec composition
  arm (accept-walk output distribution vs target p, 20k draws, L-inf + total variation).
- Verification battery on the owner's exact config: ALL PASS; sampled-spec 73.92 tok/s
  (1.72x plain-sampled), 266 spec bursts, acceptance 0.59.

## C. "We bit-audit every kernel" — the exactness discipline

- README: "Exactness is the contract... speed never changes what the model says."
  "A MISMATCH voids every number after it."
- CONTRIBUTING.md:38: "A kernel that reduces in different floating-point order can flip
  an argmax at tight logit margins — has silently broken 'faster' kernels before."
- The concat-prime-exact defect (merge b3a5465f, receipts
  `research/concat-prime-exact-20260802/findings.jsonl`): the m-dependent F32 router GEMV
  changed EXPERT SELECTION by co-arrival — 121/760 (16%) of (layer,token) pairs picked
  different expert sets depending on batch size. Fixed with m-invariant router twins;
  new contract: greedy serving is isolated-identical under concurrent load at defaults.
- Honest-limits section for credibility: first-token cross-config drift, ~7% (10/144)
  on near-tie prompts, bounded, `MEMRA_PRIME_TOKENWISE=1` escape (docs/SERVING.md).
- Post argument: determinism is not a benchmark aesthetic — it is the property that makes
  evals reproducible, agent loops debuggable, and caching auditable.

## D. The darklanes QoS model (the brand-namesake post)

- Mechanism (crates/memra-lanes/src/lib.rs): interactive/judge/harvest; shed at
  ADMISSION, never inside the engine ("the engine's own queue is where the tail dies");
  interactive never preempted; per-lane prefill budgets replace vLLM's single global
  chunked-prefill knob (the knob that taxed baseline p99 11.6 → 40.6 ms at zero parasite
  load).
- Numbers (`research/qos-p95-20260802/RESULTS.md`): a c=96 bulk tenant inflates
  interactive p95 4.2x (1.74 → 7.33 s) on a lane-blind fleet; lanes on → 3.69 s at −11%
  bulk; SLO dial 25 ms → contended p95 2.16 s ≈ alone, bulk pays −67%.
- Endurance (`research/fleet-endurance-20260803/SUMMARY.txt`): 140 min, 464,870 requests,
  0 errors, 0 sheds, +0.045% drift, greedy hash identical on all 8 replicas pre/post.
- Post argument: mixed workloads on shared GPUs are the actual economics of small fleets;
  hard p95 for interactive + harvest lanes for batch = the price/latency trade made
  explicit and receipted.

## E. Spec-decode economics on consumer-priced silicon

- `research/pro6000-prod-20260804/`: 27B-class NVFP4+MTP spec 186.7 tok/s bare / 170.6
  serve c=1; 421 agg c=8; TTFT 0.182 s cold / 3 ms warm; Q8RP +57% at c16/32 on 96GB.
- FP8 official checkpoint: embedded-MTP spec 128.06 tok/s (2.61x plain), out of the box
  from the checkpoint's own mtp.safetensors; +own-trim drafter 136.75
  (research/fp8ship-20260804/).
- 5090 reference: 9B MTP spec 2.30/1.74/1.59x llama.cpp by prompt class (README, 2026-08-02).
- Economics frame: consumer/workstation Blackwell $/token vs datacenter cards
  (`research/hw-buy-20260802/`: 2x5090 $0.47/Mtok; H100 TCO-negative on the small SKU —
  cite carefully, it's an internal estimate).
- Post argument: single-stream speed on cheap silicon changes which workloads are
  economical to serve at hard-guarantee latency.

## F. Launch-announcement material (Qwen3.8 day-one)

- `docs/qwen38-bringup-runbook.md`: release-watch → published-as-supported in one day if
  the arch-diff is clean; two legs (FP8-ST exact arm = prod direction; GGUF NVFP4+MTP =
  spec leg); deployment bar ≥1.1x e2e vs llama.cpp before "published as supported."
- The launch post can only be written AFTER the drop lands and gates pass — the site
  ships with posts A–E ready and F slotted.
