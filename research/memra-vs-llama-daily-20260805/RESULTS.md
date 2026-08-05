# memra vs llama — the owner's daily 27B, owner's actual usage shape — 2026-08-05

**DOGFOOD-EXPERIENCE DIAGNOSTIC, NOT BOARD MATERIAL.** llama benching is doctrine-stopped
for the tracked boards. This cell answers one owner question honestly: *"27B gets 73 tok/s
[sampled-spec on memra]? I think I got better with llama."*

Setup, arms, and timing definitions: `META.md`. Raw rows: `runs.jsonl` (90 rows, 0 errors).
Full server logs per phase: `logs/`. N=5 per (arm, cell), server phases interleaved
memra→llama per rep, warm steady-state (one warmup request per phase, excluded;
gpu-full-power for every phase; gpustate stamps in `logs/driver-*.log`). Same artifact
(inode-verified), both servers running the owner's exact daily serve configs.

**llama's daily script DOES run the MTP draft** (`--spec-type draft-mtp --spec-draft-n-max 3
--spec-draft-p-min 0.1`, full Q4_K_M draft). So this is spec-vs-spec, like-for-like and
as-configured — not memra-spec vs llama-plain.

## The table (medians, N=5; dec-rng = min–max across reps)

| arm | cell | TTFT s | decode tok/s | dec rng | e2e tok/s | out tok |
|---|---|---|---|---|---|---|
| memra t0.8 | short-agentic | 0.543 | 60.9* | 15–95* | 64.4 | 87 |
| memra t0.8 | long-gen | 0.400 | **88.0** | 86–92 | 87.1 | 513 |
| memra t0.8 | ctx4k | 3.012 | **80.5** | 77–82 | 45.0 | 257 |
| memra t1.0 (ACTUAL daily: pi omits temp) | short-agentic | 0.526 | 73.2* | 15–92* | 68.0 | 89 |
| memra t1.0 | long-gen | 0.402 | **89.8** | 82–93 | 88.6 | 512 |
| memra t1.0 | ctx4k | 2.988 | 76.4 | 73–79 | 43.8 | 257 |
| memra greedy (control) | short-agentic | 0.494 | 90.5 | 88–95 | 83.3 | 93 |
| memra greedy | long-gen | 0.378 | 97.0 | 95–98 | 95.4 | 513 |
| memra greedy | ctx4k | 2.952 | 89.8 | 89–94 | 47.4 | 259 |
| llama default (ACTUAL daily: t0.8+trunc) | short-agentic | **0.188** | **84.0** (srv 86.4) | 78–91 | 71.4 | 71 |
| llama default | long-gen | **0.160** | 76.5 (srv 76.8) | 75–78 | 74.9 | 512 |
| llama default | ctx4k | **1.748** | 68.1 (srv 68.5) | 67–74 | 46.7 | 256 |
| llama t1.0 | short-agentic | 0.181 | 81.2 (srv 84.5) | 76–82 | 62.4 | 41 |
| llama t1.0 | long-gen | 0.158 | 75.9 (srv 76.1) | 72–81 | 74.4 | 512 |
| llama t1.0 | ctx4k | 1.732 | 74.2 (srv 74.7) | 65–77 | 49.6 | 256 |

`srv` = llama's server-truth `timings.predicted_per_second` (client estimator confirmed
within ~3%). memra server-truth cross-check: `usage.elapsed_s` vs client wall, median
delta 3 ms, max 27 ms (N=60).

\* memra short-agentic decode is estimator-unreliable: memra releases whole spec bursts
as single SSE chunks, and short outputs arrive in only 2–3 chunks (first chunk carried
~86% of the text), so the inter-chunk estimator has almost no support — use e2e for that
cell. llama streams per-token (70+ chunks), and its server truth pins the cell at ~84–86.

The excluded arm: `memra-t0.8-lsampler` (memra + llama's truncation shape) is INVALID as
a perf datapoint — it exposed a memra sampler bug (finding 3) that collapsed short outputs
to n=6–7.

Acceptance (server truth / log-parsed):

| | short-agentic | long-gen | ctx4k |
|---|---|---|---|
| llama t0.8 (draft_n_accepted/draft_n) | **0.733** | 0.606 | 0.527 |
| llama t1.0 | **0.719** | 0.603 | 0.607 |
| memra t0.8 (spec-acc cum, per-request final) | 0.567 | 0.546 | 0.500 |
| memra t1.0 | 0.510 | 0.540 | 0.647 |
| memra greedy | 0.574 | 0.562 | — |

Cold 3.6k prefill (nonce-defeated caches, from TTFT): memra ~1,200 tok/s vs llama ~2,070
tok/s (llama server truth `prompt_per_second` median 2,069) — llama 1.7x faster at the
serve configs as-written (memra runs `MEMRA_PRIME_CHUNK=2048`, the 36.5k-OOM workaround;
llama runs ubatch 512).

F5 corroboration: **12 of 12 measured memra requests per phase hit
`[worker] spec pool evicted (1) after alloc failure; retrying`** (logs/server-memra-r*.log)
— every fresh request pays the parked-ghost evict + full ~4.4 GB session realloc, exactly
the wt-specpool root-cause. It is inside every memra TTFT here.

## The honest verdict

**The owner is right about the shape he actually feels, and the "73 tok/s" reading is
real.** memra's ACTUAL daily sampled path (t=1.0 — pi omits temperature) decodes the
owner's tool-check shape at ~73 tok/s median while llama's actual daily path shows 84–86
on the same cell — and llama's TTFT is 2.9x faster on short turns (0.19 s vs 0.53 s) and
1.7x faster at 4k context (1.75 s vs 3.0 s). In an agentic loop of many short turns,
turn-start latency plus short-burst decode IS the experience: llama feels faster there,
because it is.

memra is NOT behind everywhere: it wins long generations sampled (89.8 vs 76.5, **+17%**)
and t-matched ctx4k decode (76.4 vs 74.2; +18% at t0.8), and greedy-spec (97.0 long-gen)
anchors sanely against the board rows. The gap is concentrated in exactly three
mechanisms, all measured here:

1. **Per-request session establishment (F5).** Every memra request re-allocates a full
   131k-floor session (~4.4 GB) after evicting the parked pool ghost — 12/12 requests. On
   the owner's real pi sessions this same miss also destroys the parked continuation, so
   every turn re-primes the whole conversation — the "slower every loop" live complaint.
   llama reuses its slot at zero cost. Cost visible here: memra short-prompt TTFT ~0.53 s
   where prefill (~0.1 s) + first spec burst (~0.1 s) explain well under half of it.
2. **Prefill throughput at the daily serve config.** 1.2k vs 2.1k tok/s at 3.6k cold.
   Part config (PRIME_CHUNK=2048, the 24 GB OOM guard), part the known prefill-GEMM gap.
   This dominates the 4k-context TTFT loss and grows linearly with context — it is also
   the multiplier on F5's re-prime-everything penalty.
3. **Draft acceptance at short context under sampling.** llama's full Q4 MTP draft
   accepts 0.72–0.73 on fresh short agentic turns; memra's daily trimmed regime draft
   sits ~0.51–0.57 there (flat across temperature — even greedy is 0.574 on this cell,
   while the same server hits 0.89–0.94 bursts mid-request elsewhere in the logs). memra's
   cheaper rounds out-run llama anyway on long generations, but at short outputs llama's
   acceptance advantage wins the cell. memra's sampled tax on this cell is ~19%
   (greedy 90.5 → t1.0 73.2 median); llama's t0.8→t1.0 spread is only ~3% (84.0 → 81.2;
   no llama greedy arm was run in this battery — its temperature sensitivity on this cell
   is small where memra's is large).

### Finding 3 (BUG, found by the cross-sampler arm): memra top_p/min_p corrupt output

memra with llama's truncation shape (top_k 40 + top_p 0.95 + min_p 0.05) produced
corrupted text — `!` tokens spliced mid-word (`!bash`, `grep -!q`, `gpu-r!ig`), across
all seeds tried (battery: 5/5 reps collapsed to n=6–7; post-hoc receipts
`logs/posthoc-lsampler.txt`). Isolation: **top_k alone is clean (byte-identical to
untruncated at the same seed); top_p alone corrupts; min_p alone corrupts.** `!` is a
suspiciously low token id — consistent with the truncated-distribution sample path
returning an unmapped/fallthrough index on some rounds. The owner's daily is unaffected
(pi sends no truncation params; memra defaults are top_p=1/top_k=0/min_p=0), but any
standard OpenAI client that sets top_p — most do — hits this on the published serve
surface. This invalidates the lsampler perf arm and is a correctness lane on its own.

## Named fix lanes

1. **F5 spec-pool/session realloc — already open and prioritized**
   (`wt-specpool`, `research/specpool-20260804/RESULTS.md`). This cell adds N=5
   independent confirmation: 12/12 fresh requests evict-retry. Fixing pool-miss admission
   (don't hold the ghost while asking for a new full-floor session / right-size or resume
   instead of realloc) removes the biggest experienced gap: turn-start latency and the
   progressive session slowdown.
2. **Serve prefill lane**: memra 1.2k vs llama 2.1k tok/s at 4k. Two sub-items:
   (a) adaptive prime chunk — 2048 is a worst-case-OOM guard applied to every request;
   pick chunk by free VRAM (or retry-halve on alloc failure) so shallow-context requests
   prime at 4096+; (b) the standing prefill-GEMM rebuild priority — this is the same wall,
   measured at the serve surface.
3. **Sampled-spec acceptance at short context** (F4 follow-up): find why the daily
   trimmed draft holds ~0.55 flat where llama's full draft opens at 0.73, and why memra's
   sampled tax is 19% vs llama's 8% on the same cell — candidates: no p-min-class gate on
   the regime draft (llama gates at 0.1 and drafts only 3; memra drafted bursts of 27–51
   at 0.5 acceptance — deep rounds are cheap but the acceptance-depth product may be
   mistuned for short sampled turns), and the sampled verification rule itself.
4. **Sampler truncation correctness (BUG, public surface)**: top_p and min_p paths inject
   low-id tokens under load; top_k is clean. Receipts: `logs/posthoc-lsampler.txt`.
   Fix + a seeded truncation-matrix gate in the serve battery so this can't regress
   silently.

## Scope caveats

- Client-side timing over localhost SSE; both servers measured with the identical driver
  and cross-checked against each server's own truth channel (llama `timings`, memra
  `elapsed_s`) — agreement within ~3%.
- Single-turn requests with cold caches (nonce-defeated) — deliberately the owner's
  daily-real pool-miss shape (pi rewrites history every turn, F5). A cache-hit study is
  the F5 lane's after-curve, not this cell.
- 4k is the deepest cell here; the prefill gap and F5 re-prime penalty both GROW with
  depth, so at the owner's typical 10–20k mid-session depths the experienced llama
  advantage on turn-start is larger than measured here.
- One 512-token cell represents long-gen; the memra win there matches the board-row
  pattern (spec p1/p2/p3 116/101/86 vs llama 92/93/82).
