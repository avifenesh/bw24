# Hy3 MTP K=1 acceptance profile across serving classes — PP-2 spec-config input (Mumbai H100, 2026-08-01/02)

Answers the question the K-sweep (`research/hy3-spec-20260802/SUMMARY.md`) left for the
PP-2 spike: the d1736 8.5% vs chat 47.6% split proved acceptance is prompt-content-dependent,
so what is the ACCEPTANCE PROFILE across realistic serving classes at K=1 (the only K that
matters — the nextn=1 head never chains)? Six classes, K=1, NGEN=128, greedy, chat-templated.

## Protocol

- Box: Mumbai darklanes-bench H100 80GB (shared; **every** GPU-touching process under
  `flock /tmp/gpu-h100.lock`). GPU otherwise idle in every pre/post bracket (in-log),
  temps 35-40 C, same-night regime, all runs 20:03:36Z - 23:22:04Z Aug 1 (GPU span 3 h 18 m, 20 runs).
- Build: REUSED the K-sweep lane tree `/opt/dlami/nvme/hy3-spec-sweep/memra` at
  `2b9a6aa6` (kernel-check 206/206 ALL GREEN on the K-sweep receipts, same binary,
  byte-untouched). Branch base `restructure/public-split` tip is `c654329f`, 21 commits
  ahead; the delta does not touch `spec.rs`/`run_spec.rs` (it is concat batch-prime
  isolation + resident-if-fits residency work in decode/hybrid paths), and the K-sweep
  proved acceptance on this artifact is bit-identical across builds. Reusing the K-sweep
  binary keeps every floor number here directly comparable to the K-sweep's
  2.49 tok/s / 0.84x rows.
- Artifact: `/opt/dlami/nvme/models/hy3-layer103p5-bw24-runtime` (manifest sha
  `b8bdd684…` re-verified), bytes untouched; staged dirs left exactly per
  `research/hy3-hopper-20260801/box-state.md`.
- One run = ONE fresh `run-spec` process: `MEMRA_CHAT=1 MEMRA_NGEN=128 MEMRA_SPEC_K=1
  MEMRA_PROMPT_FILE=<class>.txt` — model load, warmup, plain greedy oracle, then spec
  K=1; plain-vs-spec arms therefore interleave within every process (same cache regime,
  same clock), and same-class reps are separated by the full round-robin (~1 h apart).
  All tok/s are gen-only (run-spec's in-API prime-subtract timer).
- Self-consistency: **PASS on every arm of every run** (run-spec asserts exactness;
  the drivers abort the batch on any nonzero exit). Acceptance counts are bit-identical
  across all reps of every class — greedy decode on this artifact is run-to-run
  deterministic, so rep count for acceptance only confirms determinism.
- Storage caveat (the first-light 1.16x lesson, re-learned in-lane): the first three runs
  of the session hit a cold NVMe page cache — their plain arms (0.43/1.92/2.09 tok/s) are
  cold-denominator artifacts and are **excluded from ratio medians** (kept in the per-rep
  table, labeled COLD). Longer-prompt classes were warm from run 4 onward (their plain
  arms reproduce the K-sweep floor 2.49 +-0.03 exactly).
- One run was operator-killed: `code-review-medium-r3` (first attempt, 22:16:55Z) — a
  redundant queued runner (`run-topup2.sh`) started the same tag as the consolidated
  `run-topup3.sh`; its process tree was killed ~1 min in (header-only log removed) and
  the class re-ran cleanly under topup3. Not a gate failure; noted for evidence
  discipline. Root cause: ssh-heredoc launcher cmdlines self-matched the `pgrep -f`
  queue-wait patterns; consolidated runner scripts are `scp`-uploaded instead.

## Prompt pack (all in `prompts/`, chat-templated single user turn)

| class | file | source | prompt tokens |
|---|---|---|---|
| chat-QA short | chat-qa-short.txt | the K-sweep chat probe (first-light prompt) | 25 |
| chat-prose medium | chat-prose-medium.txt | `research/gemma4-bringup/corpus-prompts/chat-000.txt` | 38 |
| code-gen | code-gen-short.txt | `research/e2e/prompts/p1-code-short.txt` (drafter-gate p1) | 42 |
| code-review (longer ctx) | code-review-medium.txt | `research/e2e/prompts/p2-code-medium.txt` (drafter-gate p2) | 1799 |
| agentic/tool | agentic-tool.txt | `research/q27-mtp-20260801/prompt-agentic-500w.txt` | 655 |
| summarization | summarize-medium.txt | composed: summarize instruction + `research/e2e/prompts/board-2048.txt` | 2096 |

"chat-prose medium" is medium in generation shape (explanatory prose), not prompt length —
the pack's prompt-length axis is carried by agentic (655) / code-review (1799) /
summarize (2096). The drafter-gate p3 (23 KB agentic transcript) was priced out at the
spill prefill cost (~100 s/kilotoken measured fresh-process); the 655-token tool prompt
carries the agentic class.

## Result — per-class profile (K=1, NGEN=128, greedy; medians over warm reps)

### Per-class medians
| class | prompt tok | N(acc)/N(warm) | acceptance = rounds-accept% | plain tok/s | spec tok/s | spec/plain @floor | PP-2 ceiling 1+r | PP-2 est 1+r/2 |
|---|---|---|---|---|---|---|---|---|
| chat-qa-short | 25 | 5/4 | 46.0% | 4.04 | 2.96 | 0.73x | 1.46x | 1.23x |
| chat-prose-medium | 38 | 3/2 | 43.8% | 2.68 | 3.02 | 1.13x | 1.44x | 1.22x |
| code-gen-short | 42 | 3/2 | 75.3% | 3.56 | 3.33 | 0.94x | 1.75x | 1.38x |
| code-review-medium | 1799 | 3/3 | 64.9% | 2.46 | 3.07 | 1.25x | 1.65x | 1.32x |
| agentic-tool | 655 | 3/3 | 64.9% | 2.49 | 3.02 | 1.21x | 1.65x | 1.32x |
| summarize-medium | 2096 | 3/3 | 44.3% | 2.46 | 2.63 | 1.07x | 1.44x | 1.22x |

Per-rep rows (incl. the labeled cold/disturbed reps): `table.md`.

At K=1, rounds == drafted and a round accepts 0 or 1 token, so rounds-with-accept
fraction == acceptance rate — the two requested columns are the same number at this K
(they diverge only for K>1, where this head accepts exactly zero chained drafts).

Reference row from the K-sweep (same box, same build, raw continuation): synthetic
d1736 story, 1818 tokens — acceptance 8.5%, floor ratio 0.84x, N=3.

## Mechanism at the floor — spec round cost is ~constant, plain step cost is not

Across all classes the spec verify round costs ~0.49-0.55 s at the spill floor
(2-position verify batch stages a 2-position expert union: staging-bound, nearly
ctx-independent), while the plain step cost varies with class: ~0.25 s at 25-token ctx
up to ~0.41 s at 1.8-2.1k ctx. Spec at the floor pays iff
`plain_step > round_cost / (1 + r)`:

- Short-ctx classes (fast plain step): spec LOSES or is neutral at the floor even at
  75% acceptance (code-gen 0.90-0.98x; chat-qa unstable — see below).
- Medium/long-ctx classes (plain step at the 2.46-2.49 floor): spec is neutral-to-positive
  at the floor on real content — agentic 1.21x (x3 reps, tight), summarize 1.07x (x3,
  tight), code-review spread 0.97-1.25x (fresh-process staging variance).

This refines the K-sweep verdict: "spec OFF at the 1-GPU floor" was calibrated on the
8.5%-acceptance synthetic d1736; on realistic 44-75%-acceptance content the floor is
roughly break-even at medium/long ctx. It does NOT overturn the K-sweep default (see
caveats), but it means the floor penalty of spec ON is small for long-ctx classes,
while short-ctx serving still wants it OFF at the floor.

Caveats on the floor ratios (NOT $/Mtok-grade):
1. Arms run plain-then-spec inside one process; greedy-exactness means the spec arm
   re-walks the same generation with the SLRU expert cache pre-warmed by the plain arm —
   the ratio is an upper bound on spec at the floor. (Same protocol as the K-sweep and
   probe A, so numbers are comparable within the lane.)
2. At 25-token ctx the 128-token window starts on an empty SLRU and is sensitive to the
   storage state a fresh process encounters: the chat-qa plain arm ranged 0.43-4.04 tok/s
   over five fresh processes. On a quiet warm box it is bit-tight (plain 4.03-4.04, ratio
   0.73x, x3); the 1.56x rep (r3) ran immediately after the operator-kill disturbed the
   storage state and is a labeled outlier in `table.md`.

## PP-2 resident-verify guidance (the deliverable)

Model used (the formula): at K=1 with the nextn=1 head (no chaining, proven), each spec
round drafts 1 token, verifies a 2-position batch, and emits `1 + b` tokens with
`E[b] = r` (= the class acceptance rate above). With the bank fully resident across
2x80GB the verify batch stops staging experts; assume a 2-position verify pass costs
~= 1 plain resident decode step, and charge draft-forward + verify overhead with the
prescribed discount:

    S_ceiling(r)  = 1 + r                       (free draft, verify == 1 step)
    S_est(r, K=1) = 1 + r * K/(K+1) = 1 + r/2   (honest: half the acceptance gain
                                                 covers draft forward + 2-pos overhead)

Per class:

| class | r (K=1) | S_ceiling | S_est | PP-2 spec verdict |
|---|---|---|---|---|
| code-gen | 75.3% | 1.75x | 1.38x | **ON — first to bench** |
| code-review (1.8k ctx) | 64.9% | 1.65x | 1.32x | **ON** |
| agentic/tool | 64.9% | 1.65x | 1.32x | **ON** |
| chat-QA short | 46.0% | 1.46x | 1.23x | ON (margin thinner) |
| summarization | 44.3% | 1.44x | 1.22x | ON (margin thinner) |
| chat-prose | 43.8% | 1.44x | 1.22x | ON (margin thinner) |
| (synthetic d1736 ref) | 8.5% | 1.09x | 1.04x | OFF — do not calibrate on it |

Guidance for the PP-2 spike:
- **Wire spec K=1 into the PP-2 config and bench it — every realistic class clears
  S_est >= 1.2x**, ordered code-gen > code-review == agentic > chat/summarize. No
  realistic class in this pack is expected to lose at PP-2 under this model.
- The decision variable PP-2 must measure is the actual verify-batch overhead phi in
  `S = (1+r)/(1+phi)`: this profile prices r (stable, deterministic, content-driven);
  it cannot price phi in the resident regime (the floor's staging term vanishes there).
  If a 2-position resident decode step costs meaningfully more than a 1-position step,
  S_est shrinks toward 1 — measure, don't assume.
- Do NOT reuse the synthetic d1736 story for PP-2 spec decisions: 8.5% is a synthetic
  under-statement (run_spec.rs's own warning), 5-9x below every realistic class measured.
- K stays 1: the K-sweep proved chained drafts accept exactly zero on this head; nothing
  in this profile changes that.

## Receipts

- `logs/<class>-r{1,2,3}.log` (+ `chat-qa-short-r{4,5}.log`) — full raw run-spec logs,
  GPU brackets + exit codes in-log; `logs/master.log` — batch timeline.
- `table.md` — machine-parsed per-rep + median tables (`parse-profile.py logs`).
- `prompts/` — the exact six prompt files; `accept-profile-driver.sh`, `run-all.sh`,
  `run-topup{,2,3,4}.sh` — on-box drivers (copies of `/opt/dlami/nvme/hy3-accept-profile/`'s;
  topup/topup2 are the superseded queue-wait runners documented above).
- Box left clean: no GPU processes, staged artifact + K-sweep dirs untouched; lane
  additions live only under `/opt/dlami/nvme/hy3-accept-profile/` (mirrored here).
