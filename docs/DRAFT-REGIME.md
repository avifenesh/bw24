# The draft regime — how every bw24 model gets its speculative draft

This is the DEFAULT, applied identically to every supported model. It replaced the
old per-model mix of external drafts, separate trim files, and env knobs with **one
draft file per model, zero flags** (measured sweep on the 27B champion itself:
p1 +3.4%, p2 +2.5%, p3 +0.9% over the previous board config — 2026-07-18, jsonl).

## The three laws (all measured, all violated at cost before being written down)

1. **Per-model, every time.** Rank files and draft heads are vocab+distribution
   artifacts of the EXACT serving model. Derive fresh ranks from the model's OWN
   generations for every model and every requant of a model. Foreign ranks measured
   −12 acceptance pts on an identical tokenizer; corpus text is prompts only, never
   the counted distribution; ranks also inherit their corpus MIX (wiki-heavy ranks
   lose ~12 pts on code prompts). Corpus floor: ≥4× topN generated tokens. Derive with
   the CHAT TEMPLATE ON when you serve chat (frspec-owngen's default; `--raw` is for
   pure-continuation serving) — a raw-derived rank set left a chat cell with 10.9%
   structurally-unproposable tokens (every one a guaranteed rejection; −15 acceptance
   pts, 31B 2026-07-19). The rank corpus must cover every prompt CLASS you serve —
   coverage is the whole game: an oracle control (exact escapees injected into the
   trim set) flipped a −17% cell to +2% at identical acceptance, so a trim wins any
   cell whose emitted distribution it covers and loses any cell it doesn't. A
   class×K sweep confirmed the sign NEVER flips with K, only with class coverage.
   On very large vocabs (gemma 262k) a finite own-gen corpus cannot guarantee
   coverage — the serve-time ADAPTIVE TRIM closes the gap (gemma: on by default
   with ranks; below).
2. **Byte-verbatim extraction.** The draft block comes out of the serving GGUF's own
   bytes (`tools/extract_mtp_draft.py`) — external draft ≡ embedded head, proven at
   acceptance parity. Never re-convert the MTP block from the HF checkpoint:
   converter-produced drafts collapsed to ~35-39% acceptance with no tensor-level
   difference findable (open mystery; route deprecated).
3. **Quantize AFTER trimming, judge by e2e tok/s.** Head → NVFP4 (measured zero
   acceptance cost vs q5_K at ~¼ the bytes — the hqmtp order), block → Q4_K_M
   (measured faster AND higher acceptance than Q8_0: cheaper rounds waste fewer
   drafts). The verdict metric for any draft/trim decision is END-TO-END TOK/S
   under the board protocol; acceptance is a diagnostic for why, never the decision.

## Build one (two commands, any supported model)

```bash
# 1. ranks from the model's own generations (~30-60 min GPU; built-in mixed prompt
#    pack, or point it at your own prompts / a HF dataset with hfds:owner/name)
./target/release/frspec-owngen model.gguf ranks.gguf 32768

# 2. extract + trim + quantize -> the draft file
tools/make-trimmed-draft.sh model.gguf ranks.gguf.txt draft.gguf 32768 [imatrix.gguf]
```

Serve: `BW24_MTP_DRAFT=draft.gguf ./target/release/bw24-server` (or run-spec).
Validate before trusting: `frspec-owngen model.gguf out.gguf --validate` A/Bs
baseline-vs-trimmed spec e2e and prints a GOOD/WASH/BAD verdict.

## Prebuilt drafts

Every board model's draft (built by exactly this pipeline, from exactly the published
model bytes) ships at [huggingface.co/Avifenesh/bw24-bench](https://huggingface.co/Avifenesh/bw24-bench)
with per-file provenance (source model, rank corpus, commands). Use ours for the board
models; build your own (commands above) for any other model, requant, or finetune —
a finetune's distribution moved, so its draft must too (law 1).

## Gemma variant

Gemma drafters are already standalone byte-verbatim GGUFs (law 2 by provenance); the trim
applies at LOAD instead of at build: `BW24_GEMMA_DRAFT_RANKS=<ranks.txt>` (the `.txt`
sidecar frspec-owngen emits). Laws 1 and 3 apply unchanged — own-gen ranks per model,
adopt on e2e only.

With ranks set, the **adaptive trim** is on by default (`BW24_GEMMA_TRIM_ADAPT`,
512 spare head slots): coverage escapes self-identify at serve time — they arrive as
verify corrections and ride in with the prompt — so the loop writes their head rows
into the spare slots and persists the ids to `<ranks>.learned` (pre-filled on the next
load: first-miss cost is once per serve lifetime, not per request). This closed the
one cell the static trim lost: 31B chat −17% → +2.5% vs untrimmed, and trim ≥
untrimmed on every measured cell (both models, N=2 interleaved, 2026-07-19).
Measured verdicts: 26B trim adopted, 31B trim adopted on BOTH cells (depth +5.2%,
chat +2.5% warm), E4B stays untrimmed (small head, trim buys nothing — structural).

## Regime checklist for a new supported model

- [ ] own-gen ranks derived from the published artifact (`frspec-owngen`)
- [ ] draft built via `make-trimmed-draft.sh` (byte-verbatim + NVFP4 head + Q4_K_M block)
- [ ] e2e A/B vs no-draft and vs any prior draft, board protocol (interleaved, N≥2,
      power pinned, window validated) — adopt only on e2e win
- [ ] draft + ranks uploaded to the HF bench repo with provenance in the README
- [ ] board row + README model table cross-link the HF file
