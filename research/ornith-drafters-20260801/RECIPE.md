# Ornith/KAT own-gen trimmed drafters — recipe + receipts (2026-08-01)

Lane: `lane/ornith-drafters` (from `restructure/public-split`). Owner bar: every deployed
memra model gets the own-gen trimmed drafter treatment ("require the mtp head trimed like
other models before they are deployed"). The three onboarded models
(`research/onboard-ornith-20260801/`) ship NO NextN/MTP head, so this lane applies the
standing draft regime (`docs/DRAFT-REGIME.md`) in its donor-block variant.

## Pipeline archaeology — the exact regime as reconstructed from the repo

Sources: `docs/DRAFT-REGIME.md` (the three laws), `tools/make-trimmed-draft.sh`,
`tools/extract_mtp_draft.py`, `tools/trim_draft_head.py`,
`crates/memra-engine/src/bin/frspec_owngen.rs`, jsonl rows 337/342/344/345
(`research/tune-data/rig5090.jsonl`), gemma corpus log
(`research/gemma4-bringup/mtp-step1-ranks.log`), board protocol
(`tools/full-board-bench.sh:205-211`).

1. **Own-gen ranks** (GPU): `frspec-owngen <target.gguf> owngen-ranks-32768.gguf 32768
   --ngen 512 research/gemma4-bringup/corpus-prompts` — the in-repo canonical 254-prompt
   pack (36 chat / 59 code / 3 e2e / 6 tool / 150 wiki, post-coverage-law), greedy
   (temp 0), chat template ON, ≈130k generated tokens (floor 4×topN=131,072; the
   supported builds ran 108-110k and the gemma builds 129,974 with the small-corpus
   warning — accepted). Rank counting is over the TARGET's OWN generations; prompts are
   prompts only. This lane adds `--corpus-out <ids.txt> --limit N` (committed to
   `frspec_owngen.rs`) so the corpus runs as bounded flock chunks on the shared rig;
   greedy decoding makes chunked ≡ single-run.
2. **Extract** (CPU): `tools/extract_mtp_draft.py <donor.gguf> draft-full.gguf` —
   byte-verbatim NextN block + head + embd. Donor = the serving GGUF itself for models
   that carry the head; for this batch the DONOR is the same-backbone supported artifact
   (below). Never re-convert from HF safetensors (converter drafts collapsed to 35-39%
   acceptance — jsonl row 337, route deprecated).
3. **Trim** (CPU): `tools/trim_draft_head.py draft-full.gguf <target-ranks.txt>
   draft-trim.gguf 32768` — byte-level row gather of `output.weight` to the top-32768
   target-own ranks + embedded `d2t` i32 map.
4. **Requant** (CPU): `llama-quantize --allow-requantize --output-tensor-type nvfp4
   --token-embedding-type q5_k draft-trim.gguf <out> Q4_K_M` — hqmtp order (quantize
   AFTER trim); NVFP4 head measured zero acceptance cost, Q4_K_M block measured faster
   AND higher acceptance than Q8_0.
   Steps 2-4 are one command: `tools/make-trimmed-draft.sh <donor.gguf> <ranks.txt>
   <out.gguf> 32768` (`MEMRA_CONVERT_PY` now defaults to `python3`; needs numpy +
   `/data/projects/llama.cpp/gguf-py`).
5. **Gates** (GPU, per drafter, target model + `MEMRA_MTP_DRAFT=<draft>`):
   - `run-spec` K=1..8 self-consistency PASS (greedy spec ≡ plain generate) AND
     acceptance > 0.
   - Acceptance table K=2..4 on the board prompt classes (`research/e2e/prompts/`
     p1-code-short / p2-code-medium / p3-agentic-long, `MEMRA_NGEN=256`, raw prompt —
     the `full-board-bench.sh` spec-cell protocol). Greedy acceptance is deterministic
     per (prompt, K); tok/s is not.
   - e2e spec-vs-plain ratio, interleaved x3 (every run-spec invocation runs plain
     generate then spec in the same process — one invocation = one interleaved pair;
     3 invocations at the model's serving K per class).

## Donor-block variant (why not the target's own bytes)

All three targets ship without the NextN block (Ornith-9B: 32 blocks / 427 tensors vs
the reference's 33 / 668; the 35B trio: 40 vs 41 blocks — metadata receipts in
`research/onboard-ornith-20260801/metadata/`). The draft block therefore comes from the
config-identical same-backbone donor; the RANKS (the vocab+distribution artifact — law 1)
still come from each target's own generations. The loader
(`MtpHead::load_draft`, `crates/memra-engine/src/hybrid.rs`) asserts the trunk interface
(n_embd, head_dim, n_head, n_head_kv) and takes token_embd from the serving model;
verification keeps spec exact, so donor drift can only cost acceptance. Precedent: the
the q35 fleet-box gate ran a nextn=0 GGUF with the sidecar draft, K=1..8 PASS
(`research/qwen-adaptive-k-20260801/nextn0-blocker.txt`).

| target (serving GGUF) | arch class | donor (NextN source) | serving K |
|---|---|---|---|
| `/data/ai-ml/hf-models/ornith-1.0-9b-gguf/ornith-1.0-9b-Q8_0.gguf` | qwen35 dense 9B | `/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf` (blk.32) | 3 (q9 recipe) |
| `/data/ai-ml/hf-models/ornith-1.0-35b-gguf/ornith-1.0-35b-Q4_K_M.gguf` | qwen35moe 35B-A3B | `/data/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf` (blk.40) | 2 (q35 recipe) |
| `/data/ai-ml/hf-models/kat-coder-v25-dev-gguf/Kwaipilot_KAT-Coder-V2.5-Dev-IQ4_XS.gguf` | qwen35moe 35B-A3B | `/data/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf` (blk.40) | 2 (q35 recipe) |

Artifacts (on /data next to each model, manifests+shas committed here, never weights):
`owngen-ranks-32768.gguf(.txt)` + `draft-{ornith9b,ornith35b,katcoder}-owntrim-nvfp4head-q4blk.gguf`.

## Supported-family reference (sanity bars, board logs `research/tune-data/fullboard-logs/`)

| model | K | p1-code-short | p2-code-medium | p3-agentic-long |
|---|---|---|---|---|
| q9 (Qwen3.5-9B NVFP4, own draft) | 3 | 70.3% acc / 1.91x | 54.6% / 1.59x | 49.5% / 1.42x |
| q35 (Qwen3.6-35B IQ4_XS, own draft) | 2 | 80.6% / 1.56x | 65.3% / 1.35x | 63.3% / 1.27x |

Donor-block drafters are expected BELOW these bars (the NextN block was trained for the
donor's hidden states, not the post-train's) — the adopt/reject verdict is the e2e
spec-vs-plain ratio per model, not parity with the reference.

## Status

See `STATUS.md` (updated per completed stage; corpus manifests in `corpus/`, gate logs in
`gates/`, acceptance tables in `acceptance/`).
